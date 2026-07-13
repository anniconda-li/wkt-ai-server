import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import ai_protocol
import main
from opus_packets import (
    OpusPacketsError,
    decode_ogg_to_device_wav,
    mux_aop1_to_ogg,
    parse_aop1_bytes,
)
from wav_utils import validate_device_wav
from wav_utils import write_silence_wav


def build_aop1(packets: list[bytes], pcm_samples: int | None = None) -> bytes:
    pcm_samples = pcm_samples or len(packets) * 320
    header = struct.pack(
        "<4sBBHIHHII",
        b"AOP1",
        1,
        1,
        24,
        16000,
        320,
        20,
        len(packets),
        pcm_samples,
    )
    return header + b"".join(struct.pack("<H", len(packet)) + packet for packet in packets)


class OpusPacketsTests(unittest.TestCase):
    def test_parse_aop1_and_reject_trailing_bytes(self) -> None:
        payload = build_aop1([b"\xf8\xff\xfe"] * 3)
        parsed = parse_aop1_bytes(payload)
        self.assertEqual(parsed.frame_count, 3)
        self.assertEqual(parsed.pcm_samples, 960)
        self.assertEqual(parsed.duration_seconds, 0.06)
        with self.assertRaises(OpusPacketsError):
            parse_aop1_bytes(payload + b"unexpected")

    def test_parse_aop1_rejects_truncated_and_invalid_packets(self) -> None:
        payload = build_aop1([b"\xf8\xff\xfe"] * 2)
        with self.assertRaises(OpusPacketsError):
            parse_aop1_bytes(payload[:-1])

        invalid_length = bytearray(payload)
        struct.pack_into("<H", invalid_length, 24, 1276)
        with self.assertRaises(OpusPacketsError):
            parse_aop1_bytes(bytes(invalid_length))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for Opus decode test")
    def test_muxed_ogg_is_decodable_to_device_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "request.aop1"
            ogg = root / "request.ogg"
            wav = root / "request.wav"
            source.write_bytes(build_aop1([b"\xf8\xff\xfe"] * 10))

            parsed = mux_aop1_to_ogg(source, ogg)
            decode_ogg_to_device_wav(ogg, wav)

            self.assertEqual(ogg.read_bytes()[:4], b"OggS")
            self.assertEqual(parsed.frame_count, 10)
            info = validate_device_wav(wav)
            self.assertEqual(info["sample_rate"], 16000)
            self.assertAlmostEqual(info["duration_seconds"], 0.2, places=2)

    def test_ffmpeg_decode_is_limited_to_one_thread(self) -> None:
        completed = SimpleNamespace(returncode=0, stderr="")
        with patch("opus_packets.subprocess.run", return_value=completed) as run:
            decode_ogg_to_device_wav(Path("request.ogg"), Path("request.wav"))

        command = run.call_args.args[0]
        self.assertEqual(command.count("-threads"), 2)
        for index, value in enumerate(command):
            if value == "-threads":
                self.assertEqual(command[index + 1], "1")
        filter_index = command.index("-filter_threads")
        self.assertEqual(command[filter_index + 1], "1")


class OpusUploadApiTests(unittest.TestCase):
    def setUp(self) -> None:
        ai_protocol.ai_sessions.clear()

    def test_start_and_chunk_upload_negotiate_opus(self) -> None:
        payload = build_aop1([b"\xf8\xff\xfe"] * 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                ai_protocol, "AI_ROOT", Path(temp_dir) / "uploads" / "ai"
            ):
                client = TestClient(main.app)
                start = client.post(
                    "/ai/start",
                    json={
                        "device": "opus-device",
                        "language": "zh",
                        "audio_format": "opus_packets_v1",
                    },
                )
                self.assertEqual(start.status_code, 200, start.text)
                start_body = start.json()
                self.assertTrue(start_body["ok"])
                self.assertEqual(start_body["audio_format"], "opus_packets_v1")
                self.assertEqual(start_body["chunk_size"], 32768)

                upload = client.post(
                    "/ai/upload",
                    params={
                        "session": start_body["session"],
                        "device": "opus-device",
                        "index": 0,
                        "offset": 0,
                        "total": len(payload),
                    },
                    content=payload,
                    headers={"Content-Type": "application/vnd.wkt.opus-packets"},
                )
                self.assertEqual(upload.status_code, 200, upload.text)
                session = ai_protocol.get_ai_session(start_body["session"])
                self.assertEqual(session.request_audio_path.read_bytes(), payload)

    def test_legacy_wav_start_and_upload_remain_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "request.wav"
            write_silence_wav(source, duration_seconds=0.25)
            payload = source.read_bytes()
            with patch.object(ai_protocol, "AI_ROOT", root / "uploads" / "ai"):
                client = TestClient(main.app)
                start = client.post(
                    "/ai/start",
                    json={"device": "legacy-wav", "language": "zh"},
                )
                self.assertEqual(start.status_code, 200, start.text)
                start_body = start.json()
                self.assertEqual(start_body["audio_format"], "pcm_wav")

                upload = client.post(
                    "/ai/upload",
                    params={
                        "session": start_body["session"],
                        "device": "legacy-wav",
                        "index": 0,
                        "offset": 0,
                        "total": len(payload),
                    },
                    content=payload,
                    headers={"Content-Type": "application/octet-stream"},
                )
                self.assertEqual(upload.status_code, 200, upload.text)
                session = ai_protocol.get_ai_session(start_body["session"])
                self.assertEqual(session.request_audio_path.read_bytes(), payload)

    def test_unknown_audio_format_is_rejected(self) -> None:
        client = TestClient(main.app)
        response = client.post(
            "/ai/start",
            json={"device": "bad-format", "audio_format": "mp3"},
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
