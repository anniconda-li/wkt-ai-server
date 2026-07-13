import asyncio
import logging
import os
from time import perf_counter

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
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


def camera_content_length(request: Request) -> int | None:
    raw_value = request.headers.get("content-length")
    if raw_value is None:
        return None
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
    return content_length


async def read_camera_upload_body(request: Request, device_id: str) -> bytes:
    expected_bytes = camera_content_length(request)
    idle_timeout = positive_env_number("CAMERA_UPLOAD_IDLE_TIMEOUT_SECONDS", 8.0)
    total_timeout = positive_env_number("CAMERA_UPLOAD_TOTAL_TIMEOUT_SECONDS", 30.0)
    progress_bytes = max(
        1,
        int(positive_env_number("CAMERA_UPLOAD_PROGRESS_LOG_BYTES", 4096)),
    )
    start = perf_counter()
    received_bytes = 0
    next_progress = progress_bytes
    logged_first_chunk = False
    body = bytearray()
    stream = request.stream().__aiter__()

    while True:
        remaining_total = total_timeout - (perf_counter() - start)
        if remaining_total <= 0:
            raise CameraUploadReadError(
                408,
                "total_timeout",
                "camera upload timed out",
                received_bytes=received_bytes,
                expected_bytes=expected_bytes,
            )

        wait_timeout = min(idle_timeout, remaining_total)
        timeout_reason = "total_timeout" if remaining_total <= idle_timeout else "idle_timeout"
        try:
            chunk = await asyncio.wait_for(anext(stream), timeout=wait_timeout)
        except StopAsyncIteration:
            break
        except TimeoutError as exc:
            raise CameraUploadReadError(
                408,
                timeout_reason,
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

        if (
            not logged_first_chunk
            or received_bytes >= next_progress
            or (expected_bytes is not None and received_bytes == expected_bytes)
        ):
            logger.info(
                "camera.upload.receive device=%s chunk_bytes=%d received=%d expected=%s ms=%.1f",
                device_id,
                len(chunk),
                received_bytes,
                expected_bytes,
                elapsed_ms(start),
            )
            logged_first_chunk = True
            while next_progress <= received_bytes:
                next_progress += progress_bytes

    if expected_bytes is not None and received_bytes != expected_bytes:
        raise CameraUploadReadError(
            400,
            "content_length_mismatch",
            "camera upload body length does not match Content-Length",
            received_bytes=received_bytes,
            expected_bytes=expected_bytes,
        )
    return bytes(body)


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


@app.post("/camera/upload")
async def upload_camera_image(
    request: Request,
    device: str = "default",
    artifact_id: str | None = None,
    vision_description: str | None = None,
    use_vision: bool = True,
) -> dict[str, object]:
    total_start = perf_counter()
    device_id = normalize_device_id(device)
    normalized_artifact_id = (artifact_id or "").strip()
    logger.info(
        "camera.upload.start device=%s manual_artifact=%s use_vision=%s "
        "content_type=%s content_length=%s",
        device_id,
        bool(normalized_artifact_id),
        use_vision,
        request.headers.get("content-type"),
        request.headers.get("content-length"),
    )

    lookup_start = perf_counter()
    artifact = None
    if normalized_artifact_id:
        try:
            artifact = get_artifact(normalized_artifact_id)
        except ArtifactNotFoundError as exc:
            raise HTTPException(status_code=404, detail="artifact not found") from exc
    logger.info(
        "camera.upload.stage artifact_lookup_ms=%.1f artifact_id=%s",
        elapsed_ms(lookup_start),
        normalized_artifact_id or None,
    )

    read_start = perf_counter()
    try:
        image_bytes = await read_camera_upload_body(request, device_id)
    except CameraUploadReadError as exc:
        logger.warning(
            "camera.upload.failed device=%s stage=read_body reason=%s received=%d "
            "expected=%s error=%s total_ms=%.1f",
            device_id,
            exc.reason,
            exc.received_bytes,
            exc.expected_bytes,
            exc.detail,
            elapsed_ms(total_start),
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "reason": exc.reason,
                "message": exc.detail,
                "received_bytes": exc.received_bytes,
                "expected_bytes": exc.expected_bytes,
            },
            headers={"Connection": "close"},
        ) from exc
    logger.info(
        "camera.upload.stage read_body_ms=%.1f bytes=%d",
        elapsed_ms(read_start),
        len(image_bytes),
    )

    save_start = perf_counter()
    try:
        saved_image = save_camera_image(
            device_id=device_id,
            image_bytes=image_bytes,
            content_type=request.headers.get("content-type"),
        )
    except CameraUploadError as exc:
        logger.info(
            "camera.upload.failed device=%s stage=save_image error=%s total_ms=%.1f",
            device_id,
            exc,
            elapsed_ms(total_start),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info(
        "camera.upload.stage save_image_ms=%.1f image_id=%s size_bytes=%s",
        elapsed_ms(save_start),
        saved_image["image_id"],
        saved_image["size_bytes"],
    )

    recognition_start = perf_counter()
    if artifact is not None:
        recognition = build_manual_vision_result(artifact, vision_description)
        logger.info(
            "camera.upload.stage recognition_ms=%.1f mode=%s artifact_id=%s confidence=%s",
            elapsed_ms(recognition_start),
            recognition["mode"],
            recognition["artifact_id"],
            recognition["confidence"],
        )
        session_start = perf_counter()
        session = set_artifact_context(
            device_id=device_id,
            artifact_id=normalized_artifact_id,
            vision_description=str(recognition["vision_description"]),
            image_id=str(saved_image["image_id"]),
        )
        logger.info(
            "camera.upload.stage session_write_ms=%.1f latest_artifact_id=%s",
            elapsed_ms(session_start),
            session.latest_artifact_id,
        )
    elif use_vision and is_vision_configured():
        try:
            recognition = await recognize_artifact_from_image(
                image_bytes=image_bytes,
                content_type=str(saved_image["content_type"]),
            )
        except (VisionConfigError, VisionRecognitionError) as exc:
            logger.info(
                "camera.upload.failed device=%s stage=vision_recognition error=%s total_ms=%.1f",
                device_id,
                exc,
                elapsed_ms(total_start),
            )
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        logger.info(
            "camera.upload.stage recognition_ms=%.1f mode=%s predicted_artifact_id=%s "
            "accepted=%s confidence=%s",
            elapsed_ms(recognition_start),
            recognition["mode"],
            recognition.get("predicted_artifact_id"),
            recognition.get("accepted"),
            recognition.get("confidence"),
        )

        recognized_artifact_id = recognition.get("artifact_id")
        session_start = perf_counter()
        if recognized_artifact_id:
            session = set_artifact_context(
                device_id=device_id,
                artifact_id=str(recognized_artifact_id),
                vision_description=str(recognition.get("vision_description") or ""),
                image_id=str(saved_image["image_id"]),
            )
        else:
            session = set_image_context(
                device_id=device_id,
                image_id=str(saved_image["image_id"]),
                vision_description=recognition.get("vision_description"),
            )
        logger.info(
            "camera.upload.stage session_write_ms=%.1f latest_artifact_id=%s",
            elapsed_ms(session_start),
            session.latest_artifact_id,
        )
    else:
        recognition = build_unrecognized_vision_result(vision_description)
        logger.info(
            "camera.upload.stage recognition_ms=%.1f mode=%s reason=%s",
            elapsed_ms(recognition_start),
            recognition["mode"],
            "vision_disabled_or_unconfigured",
        )
        session_start = perf_counter()
        session = set_image_context(
            device_id=device_id,
            image_id=str(saved_image["image_id"]),
            vision_description=recognition["vision_description"],
        )
        logger.info(
            "camera.upload.stage session_write_ms=%.1f latest_artifact_id=%s",
            elapsed_ms(session_start),
            session.latest_artifact_id,
        )

    logger.info(
        "camera.upload.done device=%s image_id=%s latest_artifact_id=%s total_ms=%.1f",
        session.device_id,
        saved_image["image_id"],
        session.latest_artifact_id,
        elapsed_ms(total_start),
    )

    return {
        "status": "ready",
        "device_id": session.device_id,
        "image": saved_image,
        "recognition": recognition,
        "latest_artifact_id": session.latest_artifact_id,
        "latest_image_id": session.latest_image_id,
        "latest_vision_description": session.latest_vision_description,
        "upload_generation": session.upload_generation,
    }


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
