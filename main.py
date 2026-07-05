import logging
import os
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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
logger = logging.getLogger("ai_box.main")

app = FastAPI(title="Minimal AI Chat Backend")


def elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000


class ChatRequest(BaseModel):
    message: str
    device: str = "default"


class ArtifactContextRequest(BaseModel):
    artifact_id: str
    vision_description: str | None = None
    image_id: str | None = None


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
        "camera.upload.start device=%s manual_artifact=%s use_vision=%s content_type=%s",
        device_id,
        bool(normalized_artifact_id),
        use_vision,
        request.headers.get("content-type"),
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
    image_bytes = await request.body()
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
