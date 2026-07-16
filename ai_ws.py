from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import json
import logging
import os
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

import ai_protocol
from ai_protocol import AI_AUDIO_FORMAT_OPUS, AiSession, create_ai_session, process_ai_session
from ai_ws_store import AiWsSession, AiWsSessionStore, AiWsStoreError, get_ai_ws_store
from opus_packets import OpusPacketsError, parse_aop1_file
from rop1 import ROP1_TARGET_BITRATE, encode_wav_to_rop1
from sessions import normalize_device_id
from vision import CameraUploadError, validate_jpeg_upload
from wai1_protocol import (
    PACKET_CAMERA_UPLOAD,
    PACKET_REPLY_OPUS,
    PACKET_VOICE_UPLOAD,
    WAI1_MAX_PAYLOAD,
    Wai1Error,
    Wai1Packet,
    decode_wai1_packet,
    encode_wai1_packet,
)


logger = logging.getLogger("wkt_ai_server.ai_ws")
HEARTBEAT_MS = 10_000
IDLE_TIMEOUT_SECONDS = 30
CameraProcessor = Callable[[AiWsSession, bytes], Awaitable[tuple[int, dict[str, Any]]]]


def _queue_size() -> int:
    try:
        return max(2, int(os.getenv("AI_WS_SEND_QUEUE_SIZE", "32")))
    except ValueError:
        return 32


def _error_payload(
    code: str,
    message: str,
    *,
    session: str | None = None,
    stream_id: int | None = None,
    next_offset: int | None = None,
    retryable: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "error",
        "code": code,
        "message": message,
        "retryable": retryable,
    }
    if session is not None:
        payload["session"] = session
    if stream_id is not None:
        payload["stream_id"] = stream_id
    if next_offset is not None:
        payload["next_offset"] = next_offset
    return payload


class AiWsConnection:
    def __init__(self, websocket: WebSocket, device_id: str, hub: "AiWsHub") -> None:
        self.websocket = websocket
        self.device_id = device_id
        self.hub = hub
        self.send_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=_queue_size())
        self.streams: dict[int, str] = {}
        self.reply_waiting: dict[str, int] = {}
        self.reply_stopped: set[str] = set()
        self.closed = False

    def enqueue_json(self, payload: dict[str, Any]) -> bool:
        return self._enqueue(("json", payload))

    def enqueue_bytes(self, payload: bytes) -> bool:
        return self._enqueue(("bytes", payload))

    def _enqueue(self, item: tuple[str, Any]) -> bool:
        if self.closed:
            return False
        try:
            self.send_queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            logger.warning(
                "ai.ws.send_queue_full device=%s capacity=%d",
                self.device_id,
                self.send_queue.maxsize,
            )
            self.request_close(1013, "send queue full")
            return False

    def request_close(self, code: int, reason: str) -> None:
        if self.closed:
            return
        try:
            self.send_queue.put_nowait(("close", (code, reason)))
        except asyncio.QueueFull:
            try:
                self.send_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self.send_queue.put_nowait(("close", (code, reason)))
            except asyncio.QueueFull:
                pass

    async def writer(self) -> None:
        try:
            while True:
                kind, payload = await self.send_queue.get()
                if kind == "json":
                    await self.websocket.send_json(payload)
                elif kind == "bytes":
                    await self.websocket.send_bytes(payload)
                else:
                    code, reason = payload
                    await self.websocket.close(code=code, reason=reason)
                    return
        except (WebSocketDisconnect, RuntimeError):
            return
        finally:
            self.closed = True


class AiWsHub:
    def __init__(self) -> None:
        self.connections: dict[str, AiWsConnection] = {}
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    async def register(self, connection: AiWsConnection) -> None:
        async with self._lock:
            previous = self.connections.get(connection.device_id)
            self.connections[connection.device_id] = connection
        if previous is not None and previous is not connection:
            previous.request_close(4001, "replaced by a newer AI connection")
            logger.info("ai.ws.replaced device=%s", connection.device_id)

    async def unregister(self, connection: AiWsConnection) -> None:
        async with self._lock:
            if self.connections.get(connection.device_id) is connection:
                self.connections.pop(connection.device_id, None)

    def publish(self, device_id: str, payload: dict[str, Any]) -> bool:
        connection = self.connections.get(device_id)
        return connection.enqueue_json(payload) if connection is not None else False

    def keep_task(self, session_id: str, task: asyncio.Task[None]) -> bool:
        existing = self.tasks.get(session_id)
        if existing is not None and not existing.done():
            task.cancel()
            return False
        self.tasks[session_id] = task

        def discard(completed: asyncio.Task[None]) -> None:
            if self.tasks.get(session_id) is completed:
                self.tasks.pop(session_id, None)

        task.add_done_callback(discard)
        return True


ai_ws_hub = AiWsHub()


def _voice_result(session: AiWsSession, runtime: AiSession | None = None) -> dict[str, Any]:
    status = session.status
    if runtime is not None:
        status = runtime.status
        if status in {"llm_running"}:
            status = "asr_running"
        elif status in {"audio_failed"}:
            status = "failed"
        elif status == "audio_ready" and session.reply_total <= 0:
            status = "tts_running"
    return {
        "type": "result",
        "session": session.session_id,
        "kind": "voice",
        "status": status,
        "asr_text": runtime.asr_text if runtime is not None else str((session.result or {}).get("asr_text", "")),
        "answer_text": runtime.answer_text if runtime is not None else str((session.result or {}).get("answer_text", "")),
        "error": runtime.error if runtime is not None else session.error,
    }


def _reply_ready(session: AiWsSession) -> dict[str, Any]:
    return {
        "type": "reply_ready",
        "session": session.session_id,
        "format": "rop1",
        "total": session.reply_total,
        "sha256": session.reply_sha256,
        "duration_ms": session.reply_duration_ms,
        "sample_rate": 16000,
        "channels": 1,
        "frame_ms": 20,
        "bitrate": ROP1_TARGET_BITRATE,
    }


def _camera_result(session: AiWsSession) -> dict[str, Any]:
    payload = session.result or {}
    message: dict[str, Any] = {
        "type": "result",
        "session": session.session_id,
        "kind": "camera",
        "status": session.status,
        "error": session.error,
    }
    if payload:
        message["camera_status"] = payload.get("status")
        message.update({key: value for key, value in payload.items() if key != "status"})
    return message


class AiWsProtocol:
    def __init__(
        self,
        connection: AiWsConnection,
        store: AiWsSessionStore,
        camera_processor: CameraProcessor,
    ) -> None:
        self.connection = connection
        self.store = store
        self.camera_processor = camera_processor

    def send_error(self, exc: AiWsStoreError | Wai1Error, *, session: str | None = None, stream_id: int | None = None) -> None:
        self.connection.enqueue_json(
            _error_payload(
                exc.code, exc.message, session=session, stream_id=stream_id,
                next_offset=getattr(exc, "next_offset", None), retryable=exc.retryable,
            )
        )

    async def handle_text(self, raw: str) -> bool:
        try:
            message = json.loads(raw)
            if not isinstance(message, dict) or not isinstance(message.get("type"), str):
                raise AiWsStoreError("invalid_message", "JSON message must be an object with a type")
            message_type = message["type"]
            if message_type == "ping":
                self.connection.enqueue_json({"type": "pong", "seq": message.get("seq")})
            elif message_type in {"voice_start", "camera_start"}:
                await self.start(message_type.removesuffix("_start"), message)
            elif message_type in {"voice_finish", "camera_finish"}:
                await self.finish(message_type.removesuffix("_finish"), message)
            elif message_type == "session_resume":
                await self.resume(message)
            elif message_type == "reply_get":
                await self.reply_get(message)
            elif message_type == "reply_ack":
                await self.reply_ack(message)
            elif message_type == "cancel":
                await self.cancel(message)
            elif message_type == "stop_audio":
                await self.stop_audio(message)
            else:
                raise AiWsStoreError("invalid_message", f"unsupported JSON message type: {message_type}")
            return True
        except json.JSONDecodeError:
            self.send_error(AiWsStoreError("invalid_message", "message is not valid JSON"))
        except (AiWsStoreError, Wai1Error) as exc:
            self.send_error(exc, session=str(message.get("session")) if isinstance(locals().get("message"), dict) and message.get("session") else None)
        except Exception as exc:
            logger.exception("ai.ws.message_failed device=%s error=%s", self.connection.device_id, type(exc).__name__)
            self.connection.enqueue_json(_error_payload("internal_error", "AI WebSocket request failed"))
        return False

    async def start(self, kind: str, message: dict[str, Any]) -> None:
        request_id = str(message.get("request_id") or "")
        sha256 = str(message.get("sha256") or "")
        try:
            total = int(message.get("total"))
        except (TypeError, ValueError):
            raise AiWsStoreError("invalid_message", "total must be an integer")
        if kind == "voice" and message.get("input_format") != "opus_packets_v1":
            raise AiWsStoreError("invalid_audio", "input_format must be opus_packets_v1")
        if kind == "camera" and message.get("content_type") != "image/jpeg":
            raise AiWsStoreError("invalid_jpeg", "content_type must be image/jpeg")
        session = await asyncio.to_thread(
            self.store.start,
            device_id=self.connection.device_id,
            request_id=request_id,
            kind=kind,
            total=total,
            sha256=sha256,
            language=str(message.get("language") or "zh"),
            content_type=str(message.get("content_type") or message.get("input_format") or ""),
        )
        self.connection.streams[session.stream_id] = session.session_id
        self.connection.enqueue_json(
            {
                "type": f"{kind}_started",
                "request_id": request_id,
                "session": session.session_id,
                "stream_id": session.stream_id,
                "next_offset": session.received,
                "max_payload": WAI1_MAX_PAYLOAD,
            }
        )

    async def handle_binary(self, raw: bytes) -> bool:
        packet: Wai1Packet | None = None
        try:
            packet = decode_wai1_packet(raw)
            if packet.packet_type == PACKET_REPLY_OPUS:
                raise Wai1Error("invalid_stream", "reply_opus is server-to-device only")
            session_id = self.connection.streams.get(packet.stream_id)
            if session_id is None:
                raise Wai1Error("invalid_stream", "stream_id is not active on this connection")
            kind = "voice" if packet.packet_type == PACKET_VOICE_UPLOAD else "camera"
            session, action = await asyncio.to_thread(
                self.store.append,
                session_id,
                device_id=self.connection.device_id,
                kind=kind,
                offset=packet.offset,
                total=packet.total,
                payload=packet.payload,
            )
            logger.info(
                "ai.ws.upload.%s device=%s session=%s kind=%s offset=%d bytes=%d next_offset=%d",
                action, self.connection.device_id, session_id, kind, packet.offset,
                len(packet.payload), session.received,
            )
            self.connection.enqueue_json(
                {
                    "type": "upload_ack", "kind": kind, "session": session_id,
                    "stream_id": packet.stream_id, "next_offset": session.received,
                }
            )
            return True
        except (AiWsStoreError, Wai1Error) as exc:
            self.send_error(
                exc,
                session=self.connection.streams.get(packet.stream_id) if packet else None,
                stream_id=packet.stream_id if packet else None,
            )
            return False

    def _message_session(self, message: dict[str, Any], kind: str | None = None) -> AiWsSession:
        session_id = str(message.get("session") or "")
        session = self.store.get(session_id, device_id=self.connection.device_id)
        if session is None:
            raise AiWsStoreError("session_not_found", "session does not exist")
        if kind is not None and session.kind != kind:
            raise AiWsStoreError("invalid_stream", "session kind does not match message type")
        return session

    async def finish(self, kind: str, message: dict[str, Any]) -> None:
        session = self._message_session(message, kind)
        sha256 = str(message.get("sha256") or "")
        session, action = await asyncio.to_thread(
            self.store.prepare_finish,
            session.session_id,
            device_id=self.connection.device_id,
            sha256=sha256,
        )
        if action == "claimed":
            if session.temp_path is None:
                raise AiWsStoreError("internal_error", "upload temporary file is missing")
            if kind == "voice":
                try:
                    await asyncio.to_thread(parse_aop1_file, session.temp_path)
                except OpusPacketsError as exc:
                    await asyncio.to_thread(self.store.update_status, session.session_id, "failed", error=str(exc), state="failed")
                    raise AiWsStoreError("invalid_audio", str(exc)) from exc
                task = asyncio.create_task(self._process_voice(session))
            else:
                try:
                    image_bytes = await asyncio.to_thread(session.temp_path.read_bytes)
                    await asyncio.to_thread(validate_jpeg_upload, image_bytes, "image/jpeg")
                except (OSError, CameraUploadError) as exc:
                    await asyncio.to_thread(self.store.update_status, session.session_id, "failed", error=str(exc), state="failed")
                    raise AiWsStoreError("invalid_jpeg", str(exc)) from exc
                task = asyncio.create_task(self._process_camera(session, image_bytes))
            ai_ws_hub.keep_task(session.session_id, task)
        self.connection.enqueue_json(
            _voice_result(session) if kind == "voice" else _camera_result(session)
        )
        if session.reply_total:
            self.connection.enqueue_json(_reply_ready(session))

    async def _on_voice_status(
        self,
        stored_session_id: str,
        status: str,
        asr_text: str,
        answer_text: str,
        error: str | None,
    ) -> None:
        if status == "audio_ready":
            status = "tts_running"
        elif status == "llm_running":
            status = "asr_running"
        elif status == "audio_failed":
            status = "failed"
        result = {"asr_text": asr_text, "answer_text": answer_text}
        state = "failed" if status == "failed" else "cancelled" if status == "cancelled" else None
        stored = await asyncio.to_thread(
            self.store.update_status,
            stored_session_id,
            status,
            result=result,
            error=error,
            state=state,
        )
        published = ai_ws_hub.publish(
            stored.device_id,
            {
                "type": "result",
                "session": stored.session_id,
                "kind": "voice",
                "status": status,
                "asr_text": asr_text,
                "answer_text": answer_text,
                "error": error,
            },
        )
        logger.info(
            "ai.ws.voice.status device=%s session=%s status=%s answer_chars=%d published=%s",
            stored.device_id,
            stored.session_id,
            status,
            len(answer_text),
            published,
        )

    async def _process_voice(self, stored: AiWsSession) -> None:
        if stored.temp_path is None:
            return
        runtime = create_ai_session(stored.device_id, stored.language, AI_AUDIO_FORMAT_OPUS, session_id=stored.session_id)
        root = stored.temp_path.parent
        runtime.request_audio_path = stored.temp_path
        runtime.request_wav_path = root / "request.wav"
        runtime.request_ogg_path = root / "request.ogg"
        runtime.reply_wav_path = root / "reply.wav"
        runtime.total_size = stored.total
        runtime.received_chunks = {0: stored.total}
        loop = asyncio.get_running_loop()
        pending_status_task: asyncio.Task[None] | None = None

        def schedule_status(snapshot: tuple[str, str, str, str | None]) -> None:
            nonlocal pending_status_task
            previous = pending_status_task

            async def publish_after_previous() -> None:
                if previous is not None:
                    await previous
                await self._on_voice_status(stored.session_id, *snapshot)

            pending_status_task = asyncio.create_task(publish_after_previous())

        def status_callback(current: AiSession) -> None:
            snapshot = (
                current.status,
                current.asr_text,
                current.answer_text,
                current.error,
            )
            loop.call_soon_threadsafe(schedule_status, snapshot)

        async def drain_status_updates() -> None:
            await asyncio.sleep(0)
            if pending_status_task is not None:
                await pending_status_task

        runtime.status_callback = status_callback
        try:
            await process_ai_session(runtime.session_id)
            await drain_status_updates()
            if runtime.status == "audio_ready" and runtime.reply_wav_path is not None:
                rop1_file = await asyncio.to_thread(
                    encode_wav_to_rop1, runtime.reply_wav_path, root / "reply.rop1"
                )
                latest = self.store.get(stored.session_id)
                if latest is None or latest.status == "cancelled" or runtime.cancel_requested or runtime.audio_stopped:
                    rop1_file.path.unlink(missing_ok=True)
                    return
                completed = await asyncio.to_thread(
                    self.store.set_reply,
                    stored.session_id,
                    path=rop1_file.path,
                    total=rop1_file.total,
                    sha256=rop1_file.sha256,
                    duration_ms=rop1_file.audio.duration_ms,
                )
                ai_ws_hub.publish(completed.device_id, _voice_result(completed, runtime))
                ai_ws_hub.publish(completed.device_id, _reply_ready(completed))
            elif runtime.status in {"no_speech", "cancelled", "failed", "audio_failed"}:
                final_status = "failed" if runtime.status in {"failed", "audio_failed"} else runtime.status
                state = "failed" if final_status == "failed" else "cancelled" if final_status == "cancelled" else "completed"
                await asyncio.to_thread(
                    self.store.update_status,
                    stored.session_id,
                    final_status,
                    result={"asr_text": runtime.asr_text, "answer_text": runtime.answer_text},
                    error=runtime.error,
                    state=state,
                )
        except Exception as exc:
            logger.exception("ai.ws.voice.failed session=%s error=%s", stored.session_id, type(exc).__name__)
            failed = await asyncio.to_thread(self.store.update_status, stored.session_id, "failed", error=str(exc), state="failed")
            ai_ws_hub.publish(failed.device_id, _voice_result(failed, runtime))
        finally:
            runtime.status_callback = None
            await drain_status_updates()

    async def _process_camera(self, stored: AiWsSession, image_bytes: bytes) -> None:
        try:
            status_code, payload = await self.camera_processor(stored, image_bytes)
            latest = self.store.get(stored.session_id)
            if latest is None or latest.status == "cancelled":
                return
            if status_code != 200:
                failed = await asyncio.to_thread(self.store.update_status, stored.session_id, "failed", result=payload, error=str(payload.get("detail") or "camera processing failed"), state="failed")
                ai_ws_hub.publish(failed.device_id, {"type": "result", "session": failed.session_id, "kind": "camera", "status": "failed", "error": failed.error})
                return
            completed = await asyncio.to_thread(self.store.update_status, stored.session_id, "text_ready", result=payload, state="completed")
            message = {"type": "result", "session": completed.session_id, "kind": "camera", "status": "text_ready", "camera_status": payload.get("status")}
            message.update({key: value for key, value in payload.items() if key != "status"})
            ai_ws_hub.publish(completed.device_id, message)
        except Exception as exc:
            logger.exception("ai.ws.camera.failed session=%s error=%s", stored.session_id, type(exc).__name__)
            failed = await asyncio.to_thread(self.store.update_status, stored.session_id, "failed", error=str(exc), state="failed")
            ai_ws_hub.publish(failed.device_id, {"type": "result", "session": failed.session_id, "kind": "camera", "status": "failed", "error": failed.error})

    async def resume(self, message: dict[str, Any]) -> None:
        session = self._message_session(message)
        self.connection.streams[session.stream_id] = session.session_id
        if session.kind == "voice":
            runtime = ai_protocol.ai_sessions.get(session.session_id)
            self.connection.enqueue_json(_voice_result(session, runtime))
            if session.reply_total:
                self.connection.enqueue_json(_reply_ready(session))
        else:
            self.connection.enqueue_json(_camera_result(session))

    async def _send_reply_chunk(self, session: AiWsSession, offset: int) -> None:
        if session.reply_path is None or session.reply_total <= 0 or session.reply_sha256 is None:
            raise AiWsStoreError("invalid_stream", "reply audio is not ready")
        if session.status == "cancelled":
            raise AiWsStoreError("session_cancelled", "session is cancelled")
        if offset < 0 or offset > session.reply_total:
            raise AiWsStoreError("offset_mismatch", "reply offset is outside the ROP1 file", next_offset=0)
        if session.session_id in self.connection.reply_stopped:
            raise AiWsStoreError("session_cancelled", "reply download was stopped")
        if offset == session.reply_total:
            self.connection.enqueue_json({"type": "reply_complete", "session": session.session_id, "total": session.reply_total, "sha256": session.reply_sha256})
            self.connection.reply_waiting.pop(session.session_id, None)
            return
        try:
            with session.reply_path.open("rb") as handle:
                handle.seek(offset)
                payload = handle.read(WAI1_MAX_PAYLOAD)
        except OSError as exc:
            raise AiWsStoreError("internal_error", f"failed to read ROP1 reply: {exc}") from exc
        sequence = offset // WAI1_MAX_PAYLOAD
        packet = encode_wai1_packet(Wai1Packet(PACKET_REPLY_OPUS, 1, sequence, offset, session.reply_total, payload))
        self.connection.reply_waiting[session.session_id] = offset + len(payload)
        self.connection.enqueue_bytes(packet)

    async def reply_get(self, message: dict[str, Any]) -> None:
        session = self._message_session(message, "voice")
        try:
            offset = int(message.get("offset"))
        except (TypeError, ValueError):
            raise AiWsStoreError("invalid_message", "reply offset must be an integer")
        self.connection.reply_stopped.discard(session.session_id)
        await self._send_reply_chunk(session, offset)

    async def reply_ack(self, message: dict[str, Any]) -> None:
        session = self._message_session(message, "voice")
        try:
            next_offset = int(message.get("next_offset"))
        except (TypeError, ValueError):
            raise AiWsStoreError("invalid_message", "next_offset must be an integer")
        expected = self.connection.reply_waiting.get(session.session_id)
        if expected is None or next_offset != expected:
            raise AiWsStoreError("offset_mismatch", "reply ACK does not match the outstanding chunk", next_offset=expected or 0, retryable=True)
        await self._send_reply_chunk(session, next_offset)

    async def cancel(self, message: dict[str, Any]) -> None:
        session = self._message_session(message)
        await asyncio.to_thread(self.store.cancel, session.session_id, device_id=self.connection.device_id)
        runtime = ai_protocol.ai_sessions.get(session.session_id)
        if runtime is not None:
            ai_protocol.cancel_ai_session(session.session_id, self.connection.device_id)
        self.connection.reply_stopped.add(session.session_id)
        self.connection.enqueue_json({"type": "cancelled", "session": session.session_id})

    async def stop_audio(self, message: dict[str, Any]) -> None:
        session = self._message_session(message, "voice")
        self.connection.reply_stopped.add(session.session_id)
        self.connection.reply_waiting.pop(session.session_id, None)
        runtime = ai_protocol.ai_sessions.get(session.session_id)
        if runtime is not None:
            ai_protocol.stop_ai_audio(session.session_id, self.connection.device_id)
        self.connection.enqueue_json({"type": "audio_stopped", "session": session.session_id})


async def serve_ai_websocket(
    websocket: WebSocket,
    *,
    device: str,
    protocol: str,
    camera_processor: CameraProcessor,
    store: AiWsSessionStore | None = None,
) -> None:
    await websocket.accept()
    if protocol != "wai1":
        await websocket.send_json(_error_payload("unsupported_protocol", "protocol query parameter must be wai1"))
        await websocket.close(code=1002)
        return
    device_id = normalize_device_id(device)
    connection = AiWsConnection(websocket, device_id, ai_ws_hub)
    handler = AiWsProtocol(connection, store or get_ai_ws_store(), camera_processor)
    writer_task = asyncio.create_task(connection.writer())
    await ai_ws_hub.register(connection)
    connection.enqueue_json({"type": "hello", "protocol": "wai1", "max_payload": WAI1_MAX_PAYLOAD, "heartbeat_ms": HEARTBEAT_MS, "reply_format": "rop1"})
    logger.info("ai.ws.connected device=%s", device_id)
    loop = asyncio.get_running_loop()
    last_valid_message = loop.time()
    try:
        while not connection.closed:
            idle_remaining = IDLE_TIMEOUT_SECONDS - (loop.time() - last_valid_message)
            if idle_remaining <= 0:
                connection.request_close(1001, "AI WebSocket idle timeout")
                break
            try:
                incoming = await asyncio.wait_for(websocket.receive(), timeout=idle_remaining)
            except asyncio.TimeoutError:
                connection.request_close(1001, "AI WebSocket idle timeout")
                break
            if incoming["type"] == "websocket.disconnect":
                break
            if incoming.get("text") is not None:
                valid = await handler.handle_text(str(incoming["text"]))
            elif incoming.get("bytes") is not None:
                valid = await handler.handle_binary(bytes(incoming["bytes"]))
            else:
                handler.send_error(AiWsStoreError("invalid_message", "empty WebSocket message"))
                valid = False
            if valid:
                last_valid_message = loop.time()
    except WebSocketDisconnect:
        pass
    finally:
        await ai_ws_hub.unregister(connection)
        connection.request_close(1000, "connection closed")
        try:
            await asyncio.wait_for(writer_task, timeout=1)
        except (asyncio.TimeoutError, WebSocketDisconnect, RuntimeError):
            writer_task.cancel()
        logger.info("ai.ws.disconnected device=%s", device_id)
