import asyncio
from dataclasses import dataclass
import hashlib
import logging
import os
import re
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from starlette.requests import ClientDisconnect

from ai_protocol import (
    AI_CHUNK_SIZE,
    AiProtocolError,
    ai_result_info,
    cancel_ai_session,
    create_ai_session,
    finish_ai_upload,
    get_ai_session,
    read_ai_result_chunk,
    stop_ai_audio,
    write_ai_upload_chunk,
)
from camera_idempotency import (
    CameraIdempotencyCapacityError,
    CameraIdempotencyStore,
    CameraRequestIdentity,
    CameraStoredOutcome,
    camera_idempotency_poll_interval_seconds,
    camera_idempotency_wait_timeout_seconds,
    get_camera_idempotency_store,
)
from camera_chunk_upload import (
    MAX_CHUNK_BYTES,
    CameraChunkError,
    CameraChunkIdentity,
    CameraChunkSession,
    CameraChunkStore,
    camera_chunk_cleanup_interval_seconds,
    get_camera_chunk_store,
)
from artifacts import ArtifactNotFoundError, get_artifact, list_artifacts
from llm import validate_llm_config
from router import chat_stream
from sessions import (
    clear_session,
    list_session_summaries,
    normalize_device_id,
    session_snapshot,
    set_artifact_context,
    set_image_context,
)
from vision import (
    MAX_IMAGE_BYTES,
    CameraUploadError,
    build_manual_vision_result,
    build_unrecognized_vision_result,
    save_camera_image,
    validate_jpeg_upload,
)
from vision_llm import (
    VisionConfigError,
    VisionRecognitionError,
    is_vision_configured,
    recognize_artifact_from_image,
)


def get_log_level() -> int:
    return getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)


logging.basicConfig(
    level=get_log_level(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger().setLevel(get_log_level())
logger = logging.getLogger("wkt_ai_server.main")

app = FastAPI(
    title="wkt-ai-server",
    description="Walkie-talkie AI voice, ASR, TTS, WAV/Opus chunking, and camera analysis service.",
    version="0.1.0",
)

_camera_chunk_cleanup_task: asyncio.Task[None] | None = None


async def camera_chunk_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(camera_chunk_cleanup_interval_seconds())
        try:
            await asyncio.to_thread(get_camera_chunk_store().cleanup)
        except Exception as exc:
            logger.error("camera.chunk.cleanup_failed error=%s", type(exc).__name__)


@app.on_event("startup")
async def start_camera_chunk_cleanup() -> None:
    global _camera_chunk_cleanup_task
    _camera_chunk_cleanup_task = asyncio.create_task(camera_chunk_cleanup_loop())


@app.on_event("shutdown")
async def stop_camera_chunk_cleanup() -> None:
    global _camera_chunk_cleanup_task
    if _camera_chunk_cleanup_task is None:
        return
    _camera_chunk_cleanup_task.cancel()
    try:
        await _camera_chunk_cleanup_task
    except asyncio.CancelledError:
        pass
    _camera_chunk_cleanup_task = None


def elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000


class CameraUploadReadError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        reason: str,
        detail: str,
        *,
        received_bytes: int = 0,
        expected_bytes: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.reason = reason
        self.detail = detail
        self.received_bytes = received_bytes
        self.expected_bytes = expected_bytes


@dataclass(frozen=True)
class CameraUploadHeaders:
    content_length: int
    request_id: str | None
    declared_sha256: str | None


@dataclass(frozen=True)
class CameraUploadBody:
    image_bytes: bytes
    sha256: str
    received_bytes: int


@dataclass(frozen=True)
class CameraLogContext:
    device_id: str
    request_id: str
    content_length: int
    bytes_received: int
    sha256: str
    total_start: float
    offset: int = -1
    chunk_size: int = 0
    next_offset: int | None = None


@dataclass(frozen=True)
class CameraChunkRequestMetadata:
    device_id: str
    request_id: str
    offset: int
    total: int
    content_length: int
    image_sha256: str
    chunk_sha256: str


CAMERA_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
CAMERA_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_camera_processing_tasks: set[asyncio.Task[CameraStoredOutcome]] = set()


def positive_env_number(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("config.invalid_number name=%s value=%r default=%s", name, raw_value, default)
        return default
    if value <= 0:
        logger.warning("config.non_positive_number name=%s value=%r default=%s", name, raw_value, default)
        return default
    return value


def parse_camera_upload_headers(request: Request) -> CameraUploadHeaders:
    raw_value = request.headers.get("content-length")
    if raw_value is None:
        raise CameraUploadReadError(
            411,
            "content_length_required",
            "Content-Length header is required",
        )
    try:
        content_length = int(raw_value)
    except ValueError as exc:
        raise CameraUploadReadError(
            400,
            "invalid_content_length",
            "invalid Content-Length header",
        ) from exc
    if content_length < 0:
        raise CameraUploadReadError(
            400,
            "invalid_content_length",
            "invalid Content-Length header",
            expected_bytes=content_length,
        )
    if content_length > MAX_IMAGE_BYTES:
        raise CameraUploadReadError(
            413,
            "content_length_too_large",
            f"camera upload exceeds {MAX_IMAGE_BYTES} byte limit",
            expected_bytes=content_length,
        )
    request_id_header = request.headers.get("x-request-id")
    request_id = request_id_header.strip() if request_id_header is not None else None
    if request_id is not None and not CAMERA_REQUEST_ID_PATTERN.fullmatch(request_id):
        raise CameraUploadReadError(
            400,
            "invalid_request_id",
            "X-Request-ID must contain 1-128 letters, digits, dot, underscore, colon or hyphen",
            expected_bytes=content_length,
        )

    sha256_header = request.headers.get("x-content-sha256")
    declared_sha256 = sha256_header.strip() if sha256_header is not None else None
    if declared_sha256 is not None and not CAMERA_SHA256_PATTERN.fullmatch(declared_sha256):
        raise CameraUploadReadError(
            400,
            "invalid_content_sha256",
            "X-Content-SHA256 must be 64 lowercase hexadecimal characters",
            expected_bytes=content_length,
        )
    if request_id is not None and declared_sha256 is None:
        raise CameraUploadReadError(
            400,
            "content_sha256_required",
            "X-Content-SHA256 is required when X-Request-ID is provided",
            expected_bytes=content_length,
        )
    if declared_sha256 is not None and request_id is None:
        raise CameraUploadReadError(
            400,
            "request_id_required",
            "X-Request-ID is required when X-Content-SHA256 is provided",
            expected_bytes=content_length,
        )
    return CameraUploadHeaders(
        content_length=content_length,
        request_id=request_id,
        declared_sha256=declared_sha256,
    )


async def read_camera_upload_body(
    request: Request,
    headers: CameraUploadHeaders,
) -> CameraUploadBody:
    expected_bytes = headers.content_length
    idle_timeout = positive_env_number("CAMERA_UPLOAD_IDLE_TIMEOUT_SECONDS", 8.0)
    received_bytes = 0
    body = bytearray()
    digest = hashlib.sha256()
    stream = request.stream().__aiter__()

    while True:
        try:
            chunk = await asyncio.wait_for(anext(stream), timeout=idle_timeout)
        except StopAsyncIteration:
            break
        except TimeoutError as exc:
            raise CameraUploadReadError(
                408,
                "idle_timeout",
                "camera upload timed out while waiting for request body",
                received_bytes=received_bytes,
                expected_bytes=expected_bytes,
            ) from exc
        except ClientDisconnect as exc:
            raise CameraUploadReadError(
                400,
                "client_disconnected",
                "camera upload disconnected before request body completed",
                received_bytes=received_bytes,
                expected_bytes=expected_bytes,
            ) from exc

        if not chunk:
            continue
        received_bytes += len(chunk)
        if received_bytes > MAX_IMAGE_BYTES:
            raise CameraUploadReadError(
                413,
                "body_too_large",
                f"camera upload exceeds {MAX_IMAGE_BYTES} byte limit",
                received_bytes=received_bytes,
                expected_bytes=expected_bytes,
            )
        if expected_bytes is not None and received_bytes > expected_bytes:
            raise CameraUploadReadError(
                400,
                "content_length_exceeded",
                "camera upload body exceeds declared Content-Length",
                received_bytes=received_bytes,
                expected_bytes=expected_bytes,
            )
        body.extend(chunk)
        digest.update(chunk)

    if received_bytes != expected_bytes:
        raise CameraUploadReadError(
            400,
            "content_length_mismatch",
            "camera upload body length does not match Content-Length",
            received_bytes=received_bytes,
            expected_bytes=expected_bytes,
        )
    actual_sha256 = digest.hexdigest()
    return CameraUploadBody(
        image_bytes=bytes(body),
        sha256=actual_sha256,
        received_bytes=received_bytes,
    )


def camera_log(
    event: str,
    context: CameraLogContext,
    *,
    result: str,
    stage_start: float,
    level: int = logging.INFO,
    extra: str = "",
) -> None:
    logger.log(
        level,
        "%s device=%s request_id=%s content_length=%d bytes_received=%d sha256=%s "
        "offset=%d chunk_size=%d received=%d total=%d next_offset=%d "
        "stage_ms=%.1f total_ms=%.1f result=%s%s",
        event,
        context.device_id,
        context.request_id,
        context.content_length,
        context.bytes_received,
        context.sha256,
        context.offset,
        context.chunk_size,
        context.bytes_received,
        context.content_length,
        context.next_offset if context.next_offset is not None else context.bytes_received,
        elapsed_ms(stage_start),
        elapsed_ms(context.total_start),
        result,
        extra,
    )


def camera_error_payload(
    reason: str,
    message: str,
    *,
    received_bytes: int,
    expected_bytes: int | None,
) -> dict[str, object]:
    return {
        "detail": {
            "reason": reason,
            "message": message,
            "received_bytes": received_bytes,
            "expected_bytes": expected_bytes,
        }
    }


def camera_outcome_response(outcome: CameraStoredOutcome) -> object:
    if 200 <= outcome.status_code < 300:
        return outcome.payload
    return JSONResponse(status_code=outcome.status_code, content=outcome.payload)


def camera_chunk_error_payload(
    error_code: str,
    message: str,
    *,
    next_offset: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "accepted": False,
        "error_code": error_code,
        "message": message,
    }
    if next_offset is not None:
        payload["next_offset"] = next_offset
    return payload


def camera_chunk_error_response(exc: CameraChunkError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=camera_chunk_error_payload(
            exc.error_code,
            exc.message,
            next_offset=exc.next_offset,
        ),
    )


def camera_chunk_log(
    event: str,
    *,
    device_id: str,
    request_id: str,
    offset: int,
    chunk_size: int,
    received: int,
    total: int,
    next_offset: int,
    stage_start: float,
    result: str,
    level: int = logging.INFO,
) -> None:
    logger.log(
        level,
        "%s device=%s request_id=%s offset=%d chunk_size=%d received=%d total=%d "
        "next_offset=%d stage_ms=%.1f result=%s",
        event,
        device_id,
        request_id,
        offset,
        chunk_size,
        received,
        total,
        next_offset,
        elapsed_ms(stage_start),
        result,
    )


def parse_required_integer(value: str | None, name: str) -> int:
    if value is None or not value.strip():
        raise CameraChunkError(400, f"{name}_required", f"{name} query parameter is required")
    try:
        return int(value)
    except ValueError as exc:
        raise CameraChunkError(400, f"invalid_{name}", f"{name} must be an integer") from exc


def parse_camera_chunk_metadata(
    request: Request,
    *,
    device: str | None,
    request_id: str | None,
    offset: str | None,
    total: str | None,
) -> CameraChunkRequestMetadata:
    raw_device = (device or "").strip()
    if not raw_device:
        raise CameraChunkError(400, "device_required", "device query parameter is required")
    normalized_request_id = (request_id or "").strip()
    if not CAMERA_REQUEST_ID_PATTERN.fullmatch(normalized_request_id):
        raise CameraChunkError(
            400,
            "invalid_request_id",
            "request_id must contain 1-128 letters, digits, dot, underscore, colon or hyphen",
        )
    parsed_offset = parse_required_integer(offset, "offset")
    parsed_total = parse_required_integer(total, "total")
    if parsed_offset < 0:
        raise CameraChunkError(400, "invalid_offset", "offset cannot be negative")
    if parsed_total <= 0:
        raise CameraChunkError(400, "invalid_total", "total must be positive")
    if parsed_total > MAX_IMAGE_BYTES:
        raise CameraChunkError(413, "total_too_large", "total exceeds the JPEG size limit")

    raw_content_length = request.headers.get("content-length")
    if raw_content_length is None:
        raise CameraChunkError(400, "content_length_required", "Content-Length header is required")
    try:
        content_length = int(raw_content_length)
    except ValueError as exc:
        raise CameraChunkError(400, "invalid_content_length", "Content-Length must be an integer") from exc
    if content_length <= 0:
        raise CameraChunkError(400, "invalid_content_length", "chunk Content-Length must be positive")
    if content_length > MAX_CHUNK_BYTES:
        raise CameraChunkError(413, "chunk_too_large", "chunk exceeds the 4096 byte limit")
    if parsed_offset + content_length > parsed_total:
        raise CameraChunkError(
            413,
            "chunk_exceeds_total",
            "chunk would make received bytes exceed total",
        )
    if parsed_offset + content_length < parsed_total and content_length != MAX_CHUNK_BYTES:
        raise CameraChunkError(
            400,
            "non_final_chunk_size",
            "every non-final chunk must contain exactly 4096 bytes",
        )

    content_type = (request.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    if content_type != "application/octet-stream":
        raise CameraChunkError(
            400,
            "invalid_content_type",
            "chunk Content-Type must be application/octet-stream",
        )
    image_sha256 = (request.headers.get("x-image-sha256") or "").strip()
    chunk_sha256 = (request.headers.get("x-chunk-sha256") or "").strip()
    if not CAMERA_SHA256_PATTERN.fullmatch(image_sha256):
        raise CameraChunkError(
            400,
            "invalid_image_sha256",
            "X-Image-SHA256 must be 64 lowercase hexadecimal characters",
        )
    if not CAMERA_SHA256_PATTERN.fullmatch(chunk_sha256):
        raise CameraChunkError(
            400,
            "invalid_chunk_sha256",
            "X-Chunk-SHA256 must be 64 lowercase hexadecimal characters",
        )
    return CameraChunkRequestMetadata(
        device_id=normalize_device_id(raw_device),
        request_id=normalized_request_id,
        offset=parsed_offset,
        total=parsed_total,
        content_length=content_length,
        image_sha256=image_sha256,
        chunk_sha256=chunk_sha256,
    )


async def read_camera_chunk_body(
    request: Request,
    expected_bytes: int,
) -> bytes:
    idle_timeout = positive_env_number("CAMERA_UPLOAD_IDLE_TIMEOUT_SECONDS", 8.0)
    body = bytearray()
    stream = request.stream().__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(anext(stream), timeout=idle_timeout)
        except StopAsyncIteration:
            break
        except TimeoutError as exc:
            raise CameraChunkError(408, "idle_timeout", "chunk upload timed out waiting for body data") from exc
        except ClientDisconnect as exc:
            raise CameraChunkError(400, "client_disconnected", "chunk upload disconnected before completion") from exc
        if not chunk:
            continue
        body.extend(chunk)
        if len(body) > MAX_CHUNK_BYTES:
            raise CameraChunkError(413, "chunk_too_large", "chunk exceeds the 4096 byte limit")
        if len(body) > expected_bytes:
            raise CameraChunkError(400, "content_length_exceeded", "chunk exceeds Content-Length")
    if len(body) != expected_bytes:
        raise CameraChunkError(
            400,
            "content_length_mismatch",
            "actual chunk body length does not match Content-Length",
        )
    return bytes(body)


async def require_empty_request_body(request: Request) -> None:
    raw_content_length = request.headers.get("content-length")
    if raw_content_length is not None:
        try:
            content_length = int(raw_content_length)
        except ValueError as exc:
            raise CameraChunkError(400, "invalid_content_length", "Content-Length must be zero") from exc
        if content_length != 0:
            raise CameraChunkError(400, "body_not_empty", "request body must be empty")
    stream = request.stream().__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(
                anext(stream),
                timeout=positive_env_number("CAMERA_UPLOAD_IDLE_TIMEOUT_SECONDS", 8.0),
            )
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            raise CameraChunkError(408, "idle_timeout", "timed out waiting for request body") from exc
        except ClientDisconnect:
            return
        if chunk:
            raise CameraChunkError(400, "body_not_empty", "request body must be empty")


def normalize_chunk_finish_outcome(outcome: CameraStoredOutcome) -> CameraStoredOutcome:
    if 200 <= outcome.status_code < 300 or outcome.payload.get("accepted") is False:
        return outcome
    detail = outcome.payload.get("detail")
    message = str(detail) if detail else "camera processing failed"
    error_code = "vision_recognition_failed" if outcome.status_code == 502 else "camera_processing_failed"
    return CameraStoredOutcome(
        outcome.status_code,
        camera_chunk_error_payload(error_code, message),
    )


async def persist_camera_chunk_outcome(
    chunk_store: CameraChunkStore,
    session: CameraChunkSession,
    outcome: CameraStoredOutcome,
    *,
    stage_start: float,
) -> CameraStoredOutcome:
    try:
        await asyncio.to_thread(chunk_store.complete, session.identity, outcome)
    except Exception as exc:
        camera_chunk_log(
            "camera.upload.failed",
            device_id=session.identity.device_id,
            request_id=session.identity.request_id,
            offset=session.received,
            chunk_size=0,
            received=session.received,
            total=session.identity.total,
            next_offset=session.received,
            stage_start=stage_start,
            result=f"result_persist_failed:{type(exc).__name__}",
            level=logging.ERROR,
        )
        return CameraStoredOutcome(
            500,
            camera_chunk_error_payload(
                "result_persist_failed",
                "failed to persist camera finish result",
            ),
        )
    camera_chunk_log(
        "camera.upload.failed",
        device_id=session.identity.device_id,
        request_id=session.identity.request_id,
        offset=session.received,
        chunk_size=0,
        received=session.received,
        total=session.identity.total,
        next_offset=session.received,
        stage_start=stage_start,
        result=str(outcome.payload.get("error_code") or "failed"),
        level=logging.WARNING,
    )
    return outcome


def keep_camera_task(task: asyncio.Task[CameraStoredOutcome]) -> None:
    _camera_processing_tasks.add(task)
    task.add_done_callback(_camera_processing_tasks.discard)


class ChatRequest(BaseModel):
    message: str
    device: str = "default"


class ArtifactContextRequest(BaseModel):
    artifact_id: str
    vision_description: str | None = None
    image_id: str | None = None


class AiStartRequest(BaseModel):
    device: str
    language: str = "zh"
    audio_format: str = "pcm_wav"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sessions")
async def sessions() -> list[dict[str, object]]:
    return list_session_summaries()


@app.get("/artifacts")
async def artifacts() -> list[dict[str, object]]:
    return list_artifacts()


@app.get("/artifacts/{artifact_id}")
async def artifact_detail(artifact_id: str) -> dict[str, object]:
    try:
        return get_artifact(artifact_id)
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc


@app.get("/sessions/{device_id}")
async def get_device_session(device_id: str) -> dict[str, object]:
    return session_snapshot(device_id)


@app.post("/sessions/{device_id}/clear")
async def clear_device_session(device_id: str) -> dict[str, str]:
    clear_session(device_id)
    return {"status": "cleared", "device_id": normalize_device_id(device_id)}


@app.post("/sessions/{device_id}/artifact-context")
async def set_device_artifact_context(
    device_id: str, request: ArtifactContextRequest
) -> dict[str, object]:
    start = perf_counter()
    artifact_id = request.artifact_id.strip()
    if not artifact_id:
        raise HTTPException(status_code=400, detail="artifact_id cannot be empty")

    try:
        artifact = get_artifact(artifact_id)
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc

    session = set_artifact_context(
        device_id=normalize_device_id(device_id),
        artifact_id=artifact_id,
        vision_description=request.vision_description,
        image_id=request.image_id,
    )
    logger.info(
        "session.artifact_context.done device=%s artifact_id=%s image_id=%s total_ms=%.1f",
        session.device_id,
        artifact_id,
        request.image_id,
        elapsed_ms(start),
    )
    return {
        "status": "ready",
        "device_id": session.device_id,
        "latest_artifact_id": session.latest_artifact_id,
        "latest_artifact_name": artifact["name"],
        "latest_image_id": session.latest_image_id,
        "latest_vision_description": session.latest_vision_description,
        "upload_generation": session.upload_generation,
    }


async def execute_camera_workflow(
    context: CameraLogContext,
    image_bytes: bytes,
    content_type: str,
    *,
    artifact_id: str,
    vision_description: str | None,
    use_vision: bool,
) -> CameraStoredOutcome:
    artifact = None
    if artifact_id:
        try:
            artifact = get_artifact(artifact_id)
        except ArtifactNotFoundError:
            camera_log(
                "camera.upload.failed",
                context,
                result="artifact_not_found",
                stage_start=perf_counter(),
            )
            return CameraStoredOutcome(404, {"detail": "artifact not found"})

    save_start = perf_counter()
    try:
        saved_image = await asyncio.to_thread(
            save_camera_image,
            device_id=context.device_id,
            image_bytes=image_bytes,
            content_type=content_type,
        )
    except CameraUploadError as exc:
        camera_log(
            "camera.upload.failed",
            context,
            result="save_validation_failed",
            stage_start=save_start,
            extra=f" error={exc!r}",
        )
        return CameraStoredOutcome(400, {"detail": str(exc)})
    camera_log(
        "camera.upload.saved",
        context,
        result="saved",
        stage_start=save_start,
        extra=f" image_id={saved_image['image_id']}",
    )

    recognition_start = perf_counter()
    camera_log(
        "camera.recognition.start",
        context,
        result="started",
        stage_start=recognition_start,
    )
    if artifact is not None:
        recognition = build_manual_vision_result(artifact, vision_description)
        session = set_artifact_context(
            device_id=context.device_id,
            artifact_id=artifact_id,
            vision_description=str(recognition["vision_description"]),
            image_id=str(saved_image["image_id"]),
        )
    elif use_vision and is_vision_configured():
        try:
            recognition = await recognize_artifact_from_image(
                image_bytes=image_bytes,
                content_type=str(saved_image["content_type"]),
            )
        except (VisionConfigError, VisionRecognitionError) as exc:
            camera_log(
                "camera.upload.failed",
                context,
                result="vision_recognition_failed",
                stage_start=recognition_start,
                extra=f" error={exc!r}",
            )
            return CameraStoredOutcome(502, {"detail": str(exc)})
        recognized_artifact_id = recognition.get("artifact_id")
        if recognized_artifact_id:
            session = set_artifact_context(
                device_id=context.device_id,
                artifact_id=str(recognized_artifact_id),
                vision_description=str(recognition.get("vision_description") or ""),
                image_id=str(saved_image["image_id"]),
            )
        else:
            session = set_image_context(
                device_id=context.device_id,
                image_id=str(saved_image["image_id"]),
                vision_description=recognition.get("vision_description"),
            )
    else:
        recognition = build_unrecognized_vision_result(vision_description)
        session = set_image_context(
            device_id=context.device_id,
            image_id=str(saved_image["image_id"]),
            vision_description=recognition["vision_description"],
        )

    camera_log(
        "camera.recognition.done",
        context,
        result="recognized" if recognition.get("artifact_id") else "unrecognized",
        stage_start=recognition_start,
        extra=f" mode={recognition['mode']}",
    )
    payload: dict[str, Any] = {
        "status": "ready",
        "device_id": session.device_id,
        "image": saved_image,
        "recognition": recognition,
        "latest_artifact_id": session.latest_artifact_id,
        "latest_image_id": session.latest_image_id,
        "latest_vision_description": session.latest_vision_description,
        "upload_generation": session.upload_generation,
    }
    camera_log(
        "camera.upload.done",
        context,
        result="ready",
        stage_start=context.total_start,
        extra=f" image_id={saved_image['image_id']}",
    )
    return CameraStoredOutcome(200, payload)


async def safe_execute_camera_workflow(*args: Any, **kwargs: Any) -> CameraStoredOutcome:
    context = args[0]
    try:
        return await execute_camera_workflow(*args, **kwargs)
    except Exception as exc:
        camera_log(
            "camera.upload.failed",
            context,
            result="internal_error",
            stage_start=perf_counter(),
            level=logging.ERROR,
            extra=f" error={type(exc).__name__}",
        )
        return CameraStoredOutcome(500, {"detail": "camera upload processing failed"})


async def wait_for_camera_outcome(
    store: CameraIdempotencyStore,
    identity: CameraRequestIdentity,
    context: CameraLogContext,
) -> CameraStoredOutcome:
    wait_start = perf_counter()
    timeout = camera_idempotency_wait_timeout_seconds()
    poll_interval = camera_idempotency_poll_interval_seconds()
    while elapsed_ms(wait_start) < timeout * 1000:
        claim = await asyncio.to_thread(store.get, identity)
        if claim is None:
            camera_log(
                "camera.upload.failed",
                context,
                result="idempotency_record_expired",
                stage_start=wait_start,
            )
            return CameraStoredOutcome(503, {"detail": "idempotency record expired"})
        if claim.action == "conflict":
            camera_log(
                "camera.upload.conflict",
                context,
                result="conflict",
                stage_start=wait_start,
            )
            return CameraStoredOutcome(409, {"detail": "X-Request-ID conflicts with another upload"})
        if claim.outcome is not None:
            camera_log(
                "camera.upload.idempotent_hit",
                context,
                result="completed",
                stage_start=wait_start,
            )
            return claim.outcome
        await asyncio.sleep(poll_interval)

    camera_log(
        "camera.upload.failed",
        context,
        result="idempotency_wait_timeout",
        stage_start=wait_start,
    )
    return CameraStoredOutcome(504, {"detail": "timed out waiting for original camera upload"})


async def execute_and_store_camera_outcome(
    store: CameraIdempotencyStore,
    identity: CameraRequestIdentity,
    owner_token: str,
    context: CameraLogContext,
    image_bytes: bytes,
    content_type: str,
    *,
    artifact_id: str,
    vision_description: str | None,
    use_vision: bool,
) -> CameraStoredOutcome:
    outcome = await safe_execute_camera_workflow(
        context,
        image_bytes,
        content_type,
        artifact_id=artifact_id,
        vision_description=vision_description,
        use_vision=use_vision,
    )
    for attempt in range(3):
        try:
            if await asyncio.to_thread(store.complete, identity, owner_token, outcome):
                return outcome
        except Exception:
            if attempt == 2:
                camera_log(
                    "camera.upload.failed",
                    context,
                    result="idempotency_persist_exception",
                    stage_start=perf_counter(),
                    level=logging.ERROR,
                )
            else:
                await asyncio.sleep(0.05 * (attempt + 1))
    camera_log(
        "camera.upload.failed",
        context,
        result="idempotency_persist_failed",
        stage_start=perf_counter(),
        level=logging.ERROR,
    )
    return CameraStoredOutcome(500, {"detail": "failed to persist camera upload result"})


async def process_idempotent_camera_upload(
    store: CameraIdempotencyStore,
    identity: CameraRequestIdentity,
    context: CameraLogContext,
    image_bytes: bytes,
    content_type: str,
    *,
    artifact_id: str,
    vision_description: str | None,
    use_vision: bool,
) -> CameraStoredOutcome:
    claim_start = perf_counter()
    try:
        claim = await asyncio.to_thread(store.claim, identity)
    except CameraIdempotencyCapacityError:
        camera_log(
            "camera.upload.failed",
            context,
            result="idempotency_capacity_full",
            stage_start=claim_start,
        )
        return CameraStoredOutcome(503, {"detail": "camera idempotency capacity is full"})

    if claim.action == "conflict":
        camera_log(
            "camera.upload.conflict",
            context,
            result="conflict",
            stage_start=claim_start,
        )
        return CameraStoredOutcome(409, {"detail": "X-Request-ID conflicts with another upload"})
    if claim.outcome is not None:
        camera_log(
            "camera.upload.idempotent_hit",
            context,
            result="completed",
            stage_start=claim_start,
        )
        return claim.outcome
    if claim.action == "wait":
        camera_log(
            "camera.upload.idempotent_hit",
            context,
            result="waiting",
            stage_start=claim_start,
        )
        return await wait_for_camera_outcome(store, identity, context)

    if claim.owner_token is None:
        return CameraStoredOutcome(500, {"detail": "invalid idempotency claim"})
    task = asyncio.create_task(
        execute_and_store_camera_outcome(
            store,
            identity,
            claim.owner_token,
            context,
            image_bytes,
            content_type,
            artifact_id=artifact_id,
            vision_description=vision_description,
            use_vision=use_vision,
        )
    )
    keep_camera_task(task)
    return await asyncio.shield(task)


@app.post("/camera/upload", response_model=None)
async def upload_camera_image(
    request: Request,
    device: str = "default",
    artifact_id: str | None = None,
    vision_description: str | None = None,
    use_vision: bool = True,
) -> object:
    total_start = perf_counter()
    device_id = normalize_device_id(device)
    raw_request_id = request.headers.get("x-request-id") or "-"
    header_start = perf_counter()
    try:
        headers = parse_camera_upload_headers(request)
    except CameraUploadReadError as exc:
        context = CameraLogContext(
            device_id=device_id,
            request_id=raw_request_id,
            content_length=max(exc.expected_bytes or 0, 0),
            bytes_received=exc.received_bytes,
            sha256=request.headers.get("x-content-sha256") or "-",
            total_start=total_start,
        )
        camera_log(
            "camera.upload.failed",
            context,
            result=exc.reason,
            stage_start=header_start,
            level=logging.WARNING,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=camera_error_payload(
                exc.reason,
                exc.detail,
                received_bytes=exc.received_bytes,
                expected_bytes=exc.expected_bytes,
            ),
            headers={"Connection": "close"},
        )

    request_id_for_log = headers.request_id or "-"
    context = CameraLogContext(
        device_id=device_id,
        request_id=request_id_for_log,
        content_length=headers.content_length,
        bytes_received=0,
        sha256=headers.declared_sha256 or "-",
        total_start=total_start,
    )
    camera_log(
        "camera.upload.start",
        context,
        result="receiving",
        stage_start=total_start,
    )

    body_start = perf_counter()
    try:
        body = await read_camera_upload_body(request, headers)
    except CameraUploadReadError as exc:
        interrupted_context = CameraLogContext(
            device_id=device_id,
            request_id=request_id_for_log,
            content_length=headers.content_length,
            bytes_received=exc.received_bytes,
            sha256=headers.declared_sha256 or "-",
            total_start=total_start,
        )
        camera_log(
            "camera.upload.body_interrupted",
            interrupted_context,
            result=exc.reason,
            stage_start=body_start,
            level=logging.WARNING,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=camera_error_payload(
                exc.reason,
                exc.detail,
                received_bytes=exc.received_bytes,
                expected_bytes=exc.expected_bytes,
            ),
            headers={"Connection": "close"},
        )

    context = CameraLogContext(
        device_id=device_id,
        request_id=request_id_for_log,
        content_length=headers.content_length,
        bytes_received=body.received_bytes,
        sha256=body.sha256,
        total_start=total_start,
    )
    camera_log(
        "camera.upload.body_received",
        context,
        result="complete",
        stage_start=body_start,
    )

    validation_start = perf_counter()
    if headers.declared_sha256 is not None and body.sha256 != headers.declared_sha256:
        camera_log(
            "camera.upload.failed",
            context,
            result="sha256_mismatch",
            stage_start=validation_start,
            level=logging.WARNING,
        )
        return JSONResponse(
            status_code=422,
            content=camera_error_payload(
                "sha256_mismatch",
                "X-Content-SHA256 does not match the received JPEG body",
                received_bytes=body.received_bytes,
                expected_bytes=headers.content_length,
            ),
        )
    try:
        content_type = validate_jpeg_upload(
            body.image_bytes,
            request.headers.get("content-type"),
        )
    except CameraUploadError as exc:
        camera_log(
            "camera.upload.failed",
            context,
            result="invalid_jpeg",
            stage_start=validation_start,
            level=logging.WARNING,
            extra=f" error={exc!r}",
        )
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    camera_log(
        "camera.upload.validated",
        context,
        result="valid",
        stage_start=validation_start,
    )

    normalized_artifact_id = (artifact_id or "").strip()
    if headers.request_id is None:
        outcome = await safe_execute_camera_workflow(
            context,
            body.image_bytes,
            content_type,
            artifact_id=normalized_artifact_id,
            vision_description=vision_description,
            use_vision=use_vision,
        )
        return camera_outcome_response(outcome)

    identity = CameraRequestIdentity(
        request_id=headers.request_id,
        device_id=device_id,
        sha256=body.sha256,
        content_length=headers.content_length,
    )
    store = get_camera_idempotency_store()
    outcome = await process_idempotent_camera_upload(
        store,
        identity,
        context,
        body.image_bytes,
        content_type,
        artifact_id=normalized_artifact_id,
        vision_description=vision_description,
        use_vision=use_vision,
    )
    return camera_outcome_response(outcome)


async def execute_camera_chunk_finish(
    chunk_store: CameraChunkStore,
    session: CameraChunkSession,
    context: CameraLogContext,
    image_bytes: bytes,
) -> CameraStoredOutcome:
    async def refresh_processing_lease() -> None:
        interval = min(60.0, chunk_store.processing_lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            if not await asyncio.to_thread(
                chunk_store.refresh_processing,
                session.identity.request_id,
            ):
                return

    identity = CameraRequestIdentity(
        request_id=session.identity.request_id,
        device_id=session.identity.device_id,
        sha256=session.identity.image_sha256,
        content_length=session.identity.total,
    )
    idempotency_store = get_camera_idempotency_store()
    heartbeat = asyncio.create_task(refresh_processing_lease())
    try:
        outcome = await process_idempotent_camera_upload(
            idempotency_store,
            identity,
            context,
            image_bytes,
            "image/jpeg",
            artifact_id="",
            vision_description=None,
            use_vision=True,
        )
        claim = await asyncio.to_thread(idempotency_store.get, identity)
        if claim is None or claim.outcome is None:
            normalized_outcome = normalize_chunk_finish_outcome(outcome)
            if outcome.status_code == 409:
                await asyncio.to_thread(
                    chunk_store.complete,
                    session.identity,
                    normalized_outcome,
                )
            return normalized_outcome

        final_outcome = normalize_chunk_finish_outcome(claim.outcome)
        await asyncio.to_thread(chunk_store.complete, session.identity, final_outcome)
    finally:
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass
    camera_chunk_log(
        "camera.finish.done",
        device_id=session.identity.device_id,
        request_id=session.identity.request_id,
        offset=session.identity.total,
        chunk_size=0,
        received=session.received,
        total=session.identity.total,
        next_offset=session.received,
        stage_start=context.total_start,
        result="ready" if final_outcome.status_code == 200 else "failed",
    )
    return final_outcome


@app.post("/camera/upload/chunk", response_model=None)
async def upload_camera_chunk(
    request: Request,
    device: str | None = None,
    request_id: str | None = None,
    offset: str | None = None,
    total: str | None = None,
) -> object:
    stage_start = perf_counter()
    metadata: CameraChunkRequestMetadata | None = None
    try:
        metadata = parse_camera_chunk_metadata(
            request,
            device=device,
            request_id=request_id,
            offset=offset,
            total=total,
        )
        chunk_bytes = await read_camera_chunk_body(request, metadata.content_length)
        actual_chunk_sha256 = hashlib.sha256(chunk_bytes).hexdigest()
        if actual_chunk_sha256 != metadata.chunk_sha256:
            raise CameraChunkError(
                422,
                "chunk_sha256_mismatch",
                "X-Chunk-SHA256 does not match the received chunk",
            )
        identity = CameraChunkIdentity(
            request_id=metadata.request_id,
            device_id=metadata.device_id,
            total=metadata.total,
            image_sha256=metadata.image_sha256,
        )
        result = await asyncio.to_thread(
            get_camera_chunk_store().accept_chunk,
            identity,
            offset=metadata.offset,
            chunk_bytes=chunk_bytes,
            chunk_sha256=metadata.chunk_sha256,
        )
    except CameraChunkError as exc:
        camera_chunk_log(
            "camera.chunk.rejected",
            device_id=metadata.device_id if metadata else normalize_device_id(device),
            request_id=metadata.request_id if metadata else (request_id or "-"),
            offset=metadata.offset if metadata else -1,
            chunk_size=metadata.content_length if metadata else 0,
            received=exc.next_offset or 0,
            total=metadata.total if metadata else 0,
            next_offset=exc.next_offset or 0,
            stage_start=stage_start,
            result=exc.error_code,
            level=logging.WARNING,
        )
        return camera_chunk_error_response(exc)
    except Exception as exc:
        camera_chunk_log(
            "camera.chunk.rejected",
            device_id=metadata.device_id if metadata else normalize_device_id(device),
            request_id=metadata.request_id if metadata else (request_id or "-"),
            offset=metadata.offset if metadata else -1,
            chunk_size=metadata.content_length if metadata else 0,
            received=0,
            total=metadata.total if metadata else 0,
            next_offset=0,
            stage_start=stage_start,
            result=f"internal_error:{type(exc).__name__}",
            level=logging.ERROR,
        )
        return JSONResponse(
            status_code=500,
            content=camera_chunk_error_payload("chunk_storage_failed", "failed to store camera chunk"),
        )

    session = result.session
    event = "camera.chunk.duplicate" if result.action == "duplicate" else "camera.chunk.accepted"
    camera_chunk_log(
        event,
        device_id=metadata.device_id,
        request_id=metadata.request_id,
        offset=metadata.offset,
        chunk_size=metadata.content_length,
        received=session.received,
        total=metadata.total,
        next_offset=session.received,
        stage_start=stage_start,
        result=result.action,
    )
    if session.received == metadata.total:
        camera_chunk_log(
            "camera.upload.complete",
            device_id=metadata.device_id,
            request_id=metadata.request_id,
            offset=metadata.offset,
            chunk_size=metadata.content_length,
            received=session.received,
            total=metadata.total,
            next_offset=session.received,
            stage_start=stage_start,
            result="chunks_complete",
        )
    return {
        "accepted": True,
        "request_id": metadata.request_id,
        "offset": metadata.offset,
        "chunk_size": metadata.content_length,
        "received": session.received,
        "next_offset": session.received,
        "complete": session.received == metadata.total,
    }


@app.post("/camera/upload/finish", response_model=None)
async def finish_camera_chunk_upload(
    request: Request,
    device: str | None = None,
    request_id: str | None = None,
) -> object:
    total_start = perf_counter()
    raw_device = (device or "").strip()
    normalized_request_id = (request_id or "").strip()
    try:
        if not raw_device:
            raise CameraChunkError(400, "device_required", "device query parameter is required")
        if not CAMERA_REQUEST_ID_PATTERN.fullmatch(normalized_request_id):
            raise CameraChunkError(400, "invalid_request_id", "request_id is invalid")
        image_sha256 = (request.headers.get("x-image-sha256") or "").strip()
        if not CAMERA_SHA256_PATTERN.fullmatch(image_sha256):
            raise CameraChunkError(
                400,
                "invalid_image_sha256",
                "X-Image-SHA256 must be 64 lowercase hexadecimal characters",
            )
        await require_empty_request_body(request)
        device_id = normalize_device_id(raw_device)
        chunk_store = get_camera_chunk_store()
        session = await asyncio.to_thread(
            chunk_store.prepare_finish,
            normalized_request_id,
            device_id,
            image_sha256,
        )
    except CameraChunkError as exc:
        camera_chunk_log(
            "camera.upload.failed",
            device_id=normalize_device_id(raw_device),
            request_id=normalized_request_id or "-",
            offset=-1,
            chunk_size=0,
            received=exc.next_offset or 0,
            total=0,
            next_offset=exc.next_offset or 0,
            stage_start=total_start,
            result=exc.error_code,
            level=logging.WARNING,
        )
        return camera_chunk_error_response(exc)
    except Exception as exc:
        camera_chunk_log(
            "camera.upload.failed",
            device_id=normalize_device_id(raw_device),
            request_id=normalized_request_id or "-",
            offset=-1,
            chunk_size=0,
            received=0,
            total=0,
            next_offset=0,
            stage_start=total_start,
            result=f"finish_prepare_failed:{type(exc).__name__}",
            level=logging.ERROR,
        )
        return JSONResponse(
            status_code=500,
            content=camera_chunk_error_payload(
                "finish_prepare_failed",
                "failed to prepare camera finish",
            ),
        )

    camera_chunk_log(
        "camera.finish.start",
        device_id=session.identity.device_id,
        request_id=session.identity.request_id,
        offset=session.received,
        chunk_size=0,
        received=session.received,
        total=session.identity.total,
        next_offset=session.received,
        stage_start=total_start,
        result=session.state,
    )
    if session.outcome is not None:
        return camera_outcome_response(session.outcome)
    if session.temp_path is None:
        return camera_chunk_error_response(
            CameraChunkError(500, "temporary_file_missing", "camera temporary file path is missing")
        )

    try:
        image_bytes = await asyncio.to_thread(session.temp_path.read_bytes)
    except OSError:
        latest_session = await asyncio.to_thread(chunk_store.get, session.identity.request_id)
        if latest_session is not None and latest_session.outcome is not None:
            return camera_outcome_response(latest_session.outcome)
        idempotency_identity = CameraRequestIdentity(
            request_id=session.identity.request_id,
            device_id=session.identity.device_id,
            sha256=session.identity.image_sha256,
            content_length=session.identity.total,
        )
        existing_claim = await asyncio.to_thread(
            get_camera_idempotency_store().get,
            idempotency_identity,
        )
        if existing_claim is not None and existing_claim.outcome is not None:
            recovered = normalize_chunk_finish_outcome(existing_claim.outcome)
            await asyncio.to_thread(chunk_store.complete, session.identity, recovered)
            return camera_outcome_response(recovered)
        return camera_chunk_error_response(
            CameraChunkError(500, "temporary_file_read_failed", "failed to read camera temporary file")
        )
    if len(image_bytes) != session.identity.total:
        outcome = CameraStoredOutcome(
            409,
            camera_chunk_error_payload(
                "temporary_file_length_mismatch",
                "temporary file length does not match total",
                next_offset=session.received,
            ),
        )
        return camera_outcome_response(
            await persist_camera_chunk_outcome(
                chunk_store,
                session,
                outcome,
                stage_start=total_start,
            )
        )
    actual_image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    if actual_image_sha256 != session.identity.image_sha256:
        outcome = CameraStoredOutcome(
            422,
            camera_chunk_error_payload(
                "image_sha256_mismatch",
                "complete JPEG SHA-256 does not match X-Image-SHA256",
            ),
        )
        return camera_outcome_response(
            await persist_camera_chunk_outcome(
                chunk_store,
                session,
                outcome,
                stage_start=total_start,
            )
        )
    try:
        validate_jpeg_upload(image_bytes, "image/jpeg")
    except CameraUploadError as exc:
        outcome = CameraStoredOutcome(
            422,
            camera_chunk_error_payload("invalid_jpeg", str(exc)),
        )
        return camera_outcome_response(
            await persist_camera_chunk_outcome(
                chunk_store,
                session,
                outcome,
                stage_start=total_start,
            )
        )

    context = CameraLogContext(
        device_id=session.identity.device_id,
        request_id=session.identity.request_id,
        content_length=session.identity.total,
        bytes_received=session.received,
        sha256=session.identity.image_sha256,
        total_start=total_start,
        offset=session.received,
        chunk_size=0,
        next_offset=session.received,
    )
    task = asyncio.create_task(
        execute_camera_chunk_finish(chunk_store, session, context, image_bytes)
    )
    keep_camera_task(task)
    try:
        outcome = await asyncio.shield(task)
    except CameraChunkError as exc:
        return camera_chunk_error_response(exc)
    except Exception as exc:
        camera_chunk_log(
            "camera.upload.failed",
            device_id=session.identity.device_id,
            request_id=session.identity.request_id,
            offset=session.received,
            chunk_size=0,
            received=session.received,
            total=session.identity.total,
            next_offset=session.received,
            stage_start=total_start,
            result=f"finish_internal_error:{type(exc).__name__}",
            level=logging.ERROR,
        )
        return JSONResponse(
            status_code=500,
            content=camera_chunk_error_payload(
                "finish_internal_error",
                "camera finish processing failed",
            ),
        )
    return camera_outcome_response(outcome)


@app.post("/camera/upload/cancel", response_model=None)
async def cancel_camera_chunk_upload(
    device: str | None = None,
    request_id: str | None = None,
) -> object:
    stage_start = perf_counter()
    raw_device = (device or "").strip()
    normalized_request_id = (request_id or "").strip()
    try:
        if not raw_device:
            raise CameraChunkError(400, "device_required", "device query parameter is required")
        if not CAMERA_REQUEST_ID_PATTERN.fullmatch(normalized_request_id):
            raise CameraChunkError(400, "invalid_request_id", "request_id is invalid")
        device_id = normalize_device_id(raw_device)
        result = await asyncio.to_thread(
            get_camera_chunk_store().cancel,
            normalized_request_id,
            device_id,
        )
    except CameraChunkError as exc:
        camera_chunk_log(
            "camera.upload.failed",
            device_id=normalize_device_id(raw_device),
            request_id=normalized_request_id or "-",
            offset=-1,
            chunk_size=0,
            received=exc.next_offset or 0,
            total=0,
            next_offset=exc.next_offset or 0,
            stage_start=stage_start,
            result=exc.error_code,
            level=logging.WARNING,
        )
        return camera_chunk_error_response(exc)
    except Exception as exc:
        camera_chunk_log(
            "camera.upload.failed",
            device_id=normalize_device_id(raw_device),
            request_id=normalized_request_id or "-",
            offset=-1,
            chunk_size=0,
            received=0,
            total=0,
            next_offset=0,
            stage_start=stage_start,
            result=f"cancel_failed:{type(exc).__name__}",
            level=logging.ERROR,
        )
        return JSONResponse(
            status_code=500,
            content=camera_chunk_error_payload("cancel_failed", "failed to cancel camera upload"),
        )

    camera_chunk_log(
        "camera.upload.cancelled",
        device_id=device_id,
        request_id=normalized_request_id,
        offset=-1,
        chunk_size=0,
        received=0,
        total=0,
        next_offset=0,
        stage_start=stage_start,
        result=result,
    )
    return {"accepted": True}


def raise_ai_http_error(exc: AiProtocolError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=exc.detail)


@app.post("/ai/start")
async def ai_start(request: AiStartRequest) -> dict[str, object]:
    try:
        session = create_ai_session(request.device, request.language, request.audio_format)
        return {
            "ok": True,
            "session": session.session_id,
            "audio_format": session.audio_format,
            "chunk_size": AI_CHUNK_SIZE,
        }
    except AiProtocolError as exc:
        raise_ai_http_error(exc)


@app.post("/ai/upload")
async def ai_upload(
    request: Request,
    session: str,
    index: int,
    offset: int,
    total: int,
    device: str | None = None,
) -> dict[str, object]:
    try:
        chunk = await request.body()
        return write_ai_upload_chunk(
            session,
            chunk,
            index=index,
            offset=offset,
            total=total,
            device=device,
            content_type=request.headers.get("content-type"),
        )
    except AiProtocolError as exc:
        raise_ai_http_error(exc)


@app.post("/ai/finish")
async def ai_finish(session: str, device: str | None = None) -> dict[str, object]:
    try:
        return finish_ai_upload(session, device=device)
    except AiProtocolError as exc:
        raise_ai_http_error(exc)


@app.post("/ai/result_info")
async def ai_result_info_endpoint(
    session: str, device: str | None = None
) -> dict[str, object]:
    try:
        ai_session = get_ai_session(session)
        if device is not None and normalize_device_id(device) != ai_session.device_id:
            raise AiProtocolError(409, "device does not match session")
        return ai_result_info(ai_session)
    except AiProtocolError as exc:
        raise_ai_http_error(exc)


@app.post("/ai/result_chunk")
async def ai_result_chunk(
    session: str,
    offset: int = 0,
    length: int = Query(32768, alias="len"),
    device: str | None = None,
) -> Response:
    try:
        chunk = read_ai_result_chunk(session, offset=offset, length=length, device=device)
        return Response(content=chunk, media_type="audio/wav")
    except AiProtocolError as exc:
        raise_ai_http_error(exc)


@app.post("/ai/cancel")
async def ai_cancel(session: str, device: str | None = None) -> dict[str, object]:
    try:
        return cancel_ai_session(session, device=device)
    except AiProtocolError as exc:
        raise_ai_http_error(exc)


@app.post("/ai/stop_audio")
async def ai_stop_audio(session: str, device: str | None = None) -> dict[str, object]:
    try:
        return stop_ai_audio(session, device=device)
    except AiProtocolError as exc:
        raise_ai_http_error(exc)


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    try:
        validate_llm_config()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return StreamingResponse(
        chat_stream(user_message, normalize_device_id(request.device)),
        media_type="text/plain; charset=utf-8",
    )
