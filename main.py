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
        "stage_ms=%.1f total_ms=%.1f result=%s%s",
        event,
        context.device_id,
        context.request_id,
        context.content_length,
        context.bytes_received,
        context.sha256,
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
