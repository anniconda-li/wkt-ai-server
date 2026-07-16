import asyncio
import hashlib
import os
from pathlib import Path
import shutil
import struct
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import ai_protocol
import main
from ai_ws import AiWsConnection, AiWsHub
from ai_ws_store import AiWsSessionStore, AiWsStoreError, get_ai_ws_store
from opus_packets import _build_ogg_page
from opus_packets import decode_ogg_to_device_wav
from rop1 import Rop1Audio, Rop1File, build_rop1_bytes, encode_wav_to_rop1, mux_rop1_to_ogg, parse_rop1_bytes
from wav_utils import validate_device_wav
from wai1_protocol import (
    PACKET_CAMERA_UPLOAD,
    PACKET_REPLY_OPUS,
    PACKET_VOICE_UPLOAD,
    Wai1Error,
    Wai1Packet,
    decode_wai1_packet,
    encode_wai1_packet,
)
from wav_utils import write_silence_wav


SAMPLE_JPEG = Path("samples/camera/yingguo_jade_eagle_esp32.jpg").read_bytes()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_aop1(frame_count: int = 2) -> bytes:
    packet = b"\xf8\xff\xfe"
    return struct.pack(
        "<4sBBHIHHII", b"AOP1", 1, 1, 24, 16000, 320, 20,
        frame_count, frame_count * 320,
    ) + b"".join(struct.pack("<H", len(packet)) + packet for _ in range(frame_count))


class Wai1BinaryTests(unittest.TestCase):
    def test_exact_4096_byte_packet_round_trip(self) -> None:
        payload = bytes(range(256)) * 16
        encoded = encode_wai1_packet(Wai1Packet(PACKET_VOICE_UPLOAD, 1, 7, 0, 4096, payload))
        self.assertEqual(len(encoded), 32 + 4096)
        self.assertEqual(decode_wai1_packet(encoded).payload, payload)

    def test_crc_offset_total_and_length_errors(self) -> None:
        encoded = bytearray(encode_wai1_packet(Wai1Packet(PACKET_CAMERA_UPLOAD, 2, 0, 0, 3, b"abc")))
        encoded[-1] ^= 1
        with self.assertRaisesRegex(Wai1Error, "CRC32"):
            decode_wai1_packet(bytes(encoded))
        with self.assertRaises(Wai1Error):
            encode_wai1_packet(Wai1Packet(PACKET_CAMERA_UPLOAD, 2, 0, 2, 3, b"ab"))
        valid = encode_wai1_packet(Wai1Packet(PACKET_CAMERA_UPLOAD, 2, 0, 0, 3, b"abc"))
        with self.assertRaises(Wai1Error):
            decode_wai1_packet(valid[:-1])


class AiWsStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.store = AiWsSessionStore(root / "state.sqlite3", root / "files")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_start_is_idempotent_for_voice_and_camera(self) -> None:
        voice = self.store.start(device_id="d1", request_id="voice-1", kind="voice", total=10, sha256=sha(b"x" * 10), content_type="opus_packets_v1")
        again = self.store.start(device_id="d1", request_id="voice-1", kind="voice", total=10, sha256=sha(b"x" * 10), content_type="opus_packets_v1")
        camera = self.store.start(device_id="d1", request_id="camera-1", kind="camera", total=128, sha256=sha(b"y" * 128), content_type="image/jpeg")
        self.assertEqual(voice.session_id, again.session_id)
        self.assertEqual(camera.stream_id, 2)

    def test_duplicate_chunk_after_lost_ack_is_not_written_twice(self) -> None:
        data = b"x" * 5000
        session = self.store.start(device_id="d1", request_id="retry", kind="voice", total=len(data), sha256=sha(data))
        first, action = self.store.append(session.session_id, device_id="d1", kind="voice", offset=0, total=len(data), payload=data[:4096])
        repeated, repeated_action = self.store.append(session.session_id, device_id="d1", kind="voice", offset=0, total=len(data), payload=data[:4096])
        self.assertEqual((action, repeated_action), ("accepted", "duplicate"))
        self.assertEqual(first.received, repeated.received)
        self.assertEqual(repeated.temp_path.stat().st_size, 4096)

    def test_disconnect_resume_offset_and_identity_isolation(self) -> None:
        data = b"z" * 5000
        one = self.store.start(device_id="d1", request_id="same", kind="voice", total=len(data), sha256=sha(data))
        self.store.append(one.session_id, device_id="d1", kind="voice", offset=0, total=len(data), payload=data[:4096])
        resumed = self.store.start(device_id="d1", request_id="same", kind="voice", total=len(data), sha256=sha(data))
        other = self.store.start(device_id="d2", request_id="same", kind="voice", total=len(data), sha256=sha(data))
        self.assertEqual(resumed.received, 4096)
        self.assertNotEqual(one.session_id, other.session_id)
        self.assertIsNone(self.store.get(one.session_id, device_id="d2"))

    def test_offset_and_total_conflicts_report_server_offset(self) -> None:
        data = b"x" * 5000
        session = self.store.start(device_id="d1", request_id="offset", kind="voice", total=len(data), sha256=sha(data))
        self.store.append(session.session_id, device_id="d1", kind="voice", offset=0, total=len(data), payload=data[:100])
        with self.assertRaises(AiWsStoreError) as ahead:
            self.store.append(session.session_id, device_id="d1", kind="voice", offset=200, total=len(data), payload=b"x")
        self.assertEqual(ahead.exception.next_offset, 100)
        with self.assertRaises(AiWsStoreError) as changed:
            self.store.append(session.session_id, device_id="d1", kind="voice", offset=100, total=len(data) + 1, payload=b"x")
        self.assertEqual(changed.exception.code, "total_mismatch")

        conflict = self.store.start(device_id="d1", request_id="conflict", kind="voice", total=len(data), sha256=sha(data))
        self.store.append(conflict.session_id, device_id="d1", kind="voice", offset=0, total=len(data), payload=data[:100])
        with self.assertRaises(AiWsStoreError):
            self.store.append(conflict.session_id, device_id="d1", kind="voice", offset=0, total=len(data), payload=b"y" * 100)
        self.assertEqual(self.store.get(conflict.session_id).state, "failed")

    def test_cancel_is_idempotent_and_cleanup_has_ttl_boundary(self) -> None:
        data = b"x" * 10
        session = self.store.start(device_id="d1", request_id="cancel", kind="voice", total=10, sha256=sha(data))
        self.assertEqual(self.store.cancel(session.session_id, device_id="d1").status, "cancelled")
        self.assertEqual(self.store.cancel(session.session_id, device_id="d1").status, "cancelled")
        self.assertGreaterEqual(self.store.ttl_seconds, 600)

        clock = [1000.0]
        root = Path(self.temp.name) / "expiry"
        expiring = AiWsSessionStore(root / "state.sqlite3", root / "files", clock=lambda: clock[0])
        stale = expiring.start(device_id="d2", request_id="stale", kind="voice", total=10, sha256=sha(data))
        clock[0] += expiring.ttl_seconds + 1
        self.assertEqual(expiring.cleanup(), 1)
        self.assertIsNone(expiring.get(stale.session_id))


class Rop1Tests(unittest.TestCase):
    def test_rop1_header_packets_sha_and_duration(self) -> None:
        audio = Rop1Audio(2, 600, 104, 40, (b"\xf8\xff\xfe", b"\xf8\xff\xfe"))
        data = build_rop1_bytes(audio)
        parsed = parse_rop1_bytes(data)
        self.assertEqual(parsed, audio)
        self.assertEqual(len(hashlib.sha256(data).hexdigest()), 64)
        self.assertEqual(parsed.duration_ms, 38)

    def test_ffmpeg_ogg_metadata_is_converted_to_rop1_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wav = root / "reply.wav"
            destination = root / "reply.rop1"
            write_silence_wav(wav, duration_seconds=0.2)
            pre_skip = 312
            packet = b"\xf8\xff\xfe"
            vendor = b"test"
            head = struct.pack("<8sBBHIhB", b"OpusHead", 1, 1, pre_skip, 16000, 0, 0)
            tags = b"OpusTags" + struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", 0)

            def fake_run(command, **kwargs):
                ogg_path = Path(command[-1])
                pages = [
                    _build_ogg_page([head], serial=1, sequence=0, granule_position=0, header_type=2),
                    _build_ogg_page([tags], serial=1, sequence=1, granule_position=0, header_type=0),
                    _build_ogg_page([packet] * 10, serial=1, sequence=2, granule_position=pre_skip + 9600, header_type=4),
                ]
                ogg_path.write_bytes(b"".join(pages))
                return SimpleNamespace(returncode=0, stderr="")

            with patch("rop1.subprocess.run", side_effect=fake_run):
                result = encode_wav_to_rop1(wav, destination)
            self.assertEqual(result.audio.pcm_samples, 3200)
            self.assertLessEqual(abs(result.audio.duration_ms - 200), 20)
            self.assertEqual(result.sha256, sha(destination.read_bytes()))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for ROP1 playback test")
    def test_rop1_round_trip_playback_duration_matches_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            rop1 = root / "reply.rop1"
            ogg = root / "reply.ogg"
            decoded = root / "decoded.wav"
            write_silence_wav(source, duration_seconds=0.42)
            encoded = encode_wav_to_rop1(source, rop1)
            mux_rop1_to_ogg(rop1, ogg)
            decode_ogg_to_device_wav(ogg, decoded)
            info = validate_device_wav(decoded)
            self.assertLessEqual(abs(info["duration_seconds"] * 1000 - encoded.audio.duration_ms), 20)


class AiWebSocketIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {
            "AI_WS_DB_PATH": str(self.root / "ws.sqlite3"),
            "AI_WS_TEMP_DIR": str(self.root / "ws-files"),
            "CAMERA_IDEMPOTENCY_DB_PATH": str(self.root / "camera.sqlite3"),
        })
        self.env.start()
        ai_protocol.ai_sessions.clear()
        self.client_context = TestClient(main.app)
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.env.stop()
        self.temp.cleanup()

    def connect(self, device: str = "ws-device"):
        return self.client.websocket_connect(f"/ai/ws?device={device}&protocol=wai1")

    def test_hello_ping_and_voice_start_idempotency(self) -> None:
        payload = build_aop1()
        start = {"type": "voice_start", "request_id": "voice-idempotent", "language": "zh", "input_format": "opus_packets_v1", "total": len(payload), "sha256": sha(payload)}
        with self.connect() as websocket:
            self.assertEqual(websocket.receive_json()["protocol"], "wai1")
            websocket.send_json({"type": "ping", "seq": 9})
            self.assertEqual(websocket.receive_json(), {"type": "pong", "seq": 9})
            websocket.send_json(start)
            first = websocket.receive_json()
            websocket.send_json(start)
            second = websocket.receive_json()
        self.assertEqual(first["session"], second["session"])

    def test_binary_upload_ack_duplicate_and_reconnect_resume(self) -> None:
        payload = build_aop1(900)
        start = {"type": "voice_start", "request_id": "voice-resume", "language": "zh", "input_format": "opus_packets_v1", "total": len(payload), "sha256": sha(payload)}
        with self.connect() as websocket:
            websocket.receive_json(); websocket.send_json(start)
            started = websocket.receive_json()
            packet = encode_wai1_packet(Wai1Packet(PACKET_VOICE_UPLOAD, 1, 0, 0, len(payload), payload[:4096]))
            websocket.send_bytes(packet)
            self.assertEqual(websocket.receive_json()["next_offset"], 4096)
            websocket.send_bytes(packet)
            self.assertEqual(websocket.receive_json()["next_offset"], 4096)
        with self.connect() as websocket:
            websocket.receive_json(); websocket.send_json(start)
            resumed = websocket.receive_json()
        self.assertEqual(resumed["session"], started["session"])
        self.assertEqual(resumed["next_offset"], 4096)

    def test_camera_finish_survives_disconnect_and_session_resume(self) -> None:
        release = threading.Event()

        async def camera_result(session, image_bytes):
            await asyncio.to_thread(release.wait, 2)
            return 200, {"status": "ready", "device_id": session.device_id, "image": {"size_bytes": len(image_bytes)}, "recognition": {"mode": "mock"}, "latest_artifact_id": None, "latest_image_id": "img", "latest_vision_description": "mock", "upload_generation": 1}

        start = {"type": "camera_start", "request_id": "camera-background", "content_type": "image/jpeg", "total": len(SAMPLE_JPEG), "sha256": sha(SAMPLE_JPEG)}
        with patch.object(main, "process_ai_ws_camera", new=camera_result):
            with self.connect() as websocket:
                websocket.receive_json(); websocket.send_json(start)
                started = websocket.receive_json()
                for sequence, offset in enumerate(range(0, len(SAMPLE_JPEG), 4096)):
                    chunk = SAMPLE_JPEG[offset:offset + 4096]
                    websocket.send_bytes(encode_wai1_packet(Wai1Packet(PACKET_CAMERA_UPLOAD, 2, sequence, offset, len(SAMPLE_JPEG), chunk)))
                    websocket.receive_json()
                websocket.send_json({"type": "camera_finish", "session": started["session"], "sha256": sha(SAMPLE_JPEG)})
                self.assertEqual(websocket.receive_json()["status"], "uploaded")
            release.set()
            deadline = time.time() + 3
            while time.time() < deadline and get_ai_ws_store().get(started["session"]).state != "completed":
                time.sleep(0.02)
            with self.connect() as websocket:
                websocket.receive_json(); websocket.send_json({"type": "session_resume", "session": started["session"]})
                result = websocket.receive_json()
        self.assertEqual(result["status"], "text_ready")
        self.assertEqual(result["latest_image_id"], "img")

    def test_voice_finish_continues_after_disconnect_and_resumes_final_result(self) -> None:
        payload = build_aop1(3)

        async def fake_pipeline(session_id: str) -> None:
            runtime = ai_protocol.get_ai_session(session_id)
            ai_protocol.set_status(runtime, "asr_running")
            await asyncio.sleep(0.02)
            runtime.asr_text = "这是什么文物"
            runtime.answer_text = "这是应国玉鹰。"
            ai_protocol.set_status(runtime, "text_ready")
            ai_protocol.set_status(runtime, "tts_running")
            write_silence_wav(runtime.reply_wav_path, duration_seconds=0.2)
            runtime.reply_wav_size = runtime.reply_wav_path.stat().st_size
            ai_protocol.set_status(runtime, "audio_ready")

        def fake_rop1(source: Path, destination: Path) -> Rop1File:
            audio = Rop1Audio(1, 320, 104, 0, (b"\xf8\xff\xfe",))
            data = build_rop1_bytes(audio)
            destination.write_bytes(data)
            return Rop1File(destination, len(data), sha(data), audio)

        start = {"type": "voice_start", "request_id": "voice-background", "language": "zh", "input_format": "opus_packets_v1", "total": len(payload), "sha256": sha(payload)}
        with patch("ai_ws.process_ai_session", side_effect=fake_pipeline), patch("ai_ws.encode_wav_to_rop1", side_effect=fake_rop1):
            with self.connect() as websocket:
                websocket.receive_json(); websocket.send_json(start)
                started = websocket.receive_json()
                websocket.send_bytes(encode_wai1_packet(Wai1Packet(PACKET_VOICE_UPLOAD, 1, 0, 0, len(payload), payload)))
                websocket.receive_json()
                websocket.send_json({"type": "voice_finish", "session": started["session"], "sha256": sha(payload)})
                self.assertEqual(websocket.receive_json()["status"], "uploaded")
            deadline = time.time() + 3
            while time.time() < deadline and get_ai_ws_store().get(started["session"]).state != "completed":
                time.sleep(0.02)
            with self.connect() as websocket:
                websocket.receive_json(); websocket.send_json({"type": "session_resume", "session": started["session"]})
                final_result = websocket.receive_json()
                ready = websocket.receive_json()
        self.assertEqual(final_result["status"], "audio_ready")
        self.assertEqual(final_result["answer_text"], "这是应国玉鹰。")
        self.assertEqual(ready["type"], "reply_ready")

    def test_voice_text_ready_is_pushed_before_audio_when_statuses_change_without_yield(self) -> None:
        payload = build_aop1(3)

        async def fake_pipeline(session_id: str) -> None:
            runtime = ai_protocol.get_ai_session(session_id)
            runtime.asr_text = "这是什么文物"
            runtime.answer_text = "这是应国玉鹰。"
            ai_protocol.set_status(runtime, "text_ready")
            ai_protocol.set_status(runtime, "tts_running")
            write_silence_wav(runtime.reply_wav_path, duration_seconds=0.2)
            runtime.reply_wav_size = runtime.reply_wav_path.stat().st_size
            ai_protocol.set_status(runtime, "audio_ready")

        def fake_rop1(source: Path, destination: Path) -> Rop1File:
            audio = Rop1Audio(1, 320, 104, 0, (b"\xf8\xff\xfe",))
            data = build_rop1_bytes(audio)
            destination.write_bytes(data)
            return Rop1File(destination, len(data), sha(data), audio)

        start = {
            "type": "voice_start",
            "request_id": "voice-text-first",
            "language": "zh",
            "input_format": "opus_packets_v1",
            "total": len(payload),
            "sha256": sha(payload),
        }
        with (
            patch("ai_ws.process_ai_session", side_effect=fake_pipeline),
            patch("ai_ws.encode_wav_to_rop1", side_effect=fake_rop1),
            self.connect() as websocket,
        ):
            websocket.receive_json()
            websocket.send_json(start)
            started = websocket.receive_json()
            websocket.send_bytes(
                encode_wai1_packet(
                    Wai1Packet(PACKET_VOICE_UPLOAD, 1, 0, 0, len(payload), payload)
                )
            )
            websocket.receive_json()
            websocket.send_json(
                {
                    "type": "voice_finish",
                    "session": started["session"],
                    "sha256": sha(payload),
                }
            )

            messages = []
            while not any(message.get("type") == "reply_ready" for message in messages):
                messages.append(websocket.receive_json())

        statuses = [
            message["status"]
            for message in messages
            if message.get("type") == "result" and message.get("kind") == "voice"
        ]
        text_ready_index = statuses.index("text_ready")
        audio_ready_index = statuses.index("audio_ready")
        self.assertLess(text_ready_index, audio_ready_index)
        text_ready = next(
            message
            for message in messages
            if message.get("type") == "result" and message.get("status") == "text_ready"
        )
        self.assertEqual(text_ready["answer_text"], "这是应国玉鹰。")

    def test_reply_get_supports_stop_and_wait_and_reconnect_offset(self) -> None:
        store = get_ai_ws_store()
        session = store.start(device_id="reply-device", request_id="reply", kind="voice", total=1, sha256=sha(b"x"))
        reply = self.root / "reply.rop1"
        reply.write_bytes(bytes(range(250)) * 20)
        session = store.set_reply(session.session_id, path=reply, total=reply.stat().st_size, sha256=sha(reply.read_bytes()), duration_ms=1000)
        with self.connect("reply-device") as websocket:
            websocket.receive_json(); websocket.send_json({"type": "reply_get", "session": session.session_id, "offset": 0})
            first = decode_wai1_packet(websocket.receive_bytes())
            self.assertEqual((first.packet_type, len(first.payload)), (PACKET_REPLY_OPUS, 4096))
        with self.connect("reply-device") as websocket:
            websocket.receive_json(); websocket.send_json({"type": "reply_get", "session": session.session_id, "offset": 4096})
            second = decode_wai1_packet(websocket.receive_bytes())
            websocket.send_json({"type": "reply_ack", "session": session.session_id, "next_offset": second.next_offset})
            complete = websocket.receive_json()
        self.assertEqual(second.offset, 4096)
        self.assertEqual(complete["type"], "reply_complete")

    def test_cancel_is_idempotent_over_websocket(self) -> None:
        payload = build_aop1()
        with self.connect() as websocket:
            websocket.receive_json()
            websocket.send_json({"type": "voice_start", "request_id": "cancel-ws", "language": "zh", "input_format": "opus_packets_v1", "total": len(payload), "sha256": sha(payload)})
            session = websocket.receive_json()["session"]
            for _ in range(2):
                websocket.send_json({"type": "cancel", "session": session})
                self.assertEqual(websocket.receive_json()["type"], "cancelled")

    def test_protocol_does_not_use_public_ip(self) -> None:
        payload = build_aop1()
        with self.connect("nat-device") as websocket:
            websocket.receive_json(); websocket.send_json({"type": "voice_start", "request_id": "nat", "language": "zh", "input_format": "opus_packets_v1", "total": len(payload), "sha256": sha(payload)})
            session = websocket.receive_json()["session"]
        stored = get_ai_ws_store().get(session)
        self.assertEqual(stored.device_id, "nat-device")
        self.assertFalse(hasattr(stored, "client_ip"))


class AiWsConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_connection_replaces_same_device_only(self) -> None:
        hub = AiWsHub()
        old = AiWsConnection(SimpleNamespace(), "d1", hub)
        new = AiWsConnection(SimpleNamespace(), "d1", hub)
        other = AiWsConnection(SimpleNamespace(), "d2", hub)
        await hub.register(old); await hub.register(other); await hub.register(new)
        self.assertIs(hub.connections["d1"], new)
        self.assertIs(hub.connections["d2"], other)
        self.assertEqual((await old.send_queue.get())[0], "close")

    async def test_slow_device_queue_is_bounded(self) -> None:
        with patch.dict(os.environ, {"AI_WS_SEND_QUEUE_SIZE": "2"}):
            connection = AiWsConnection(SimpleNamespace(), "slow", AiWsHub())
        self.assertTrue(connection.enqueue_json({"n": 1}))
        self.assertTrue(connection.enqueue_json({"n": 2}))
        self.assertFalse(connection.enqueue_json({"n": 3}))
        self.assertLessEqual(connection.send_queue.qsize(), 2)


if __name__ == "__main__":
    unittest.main()
