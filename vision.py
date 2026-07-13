from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4


UPLOADS_DIR = Path(__file__).parent / "uploads"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MIN_JPEG_BYTES = 128
SUPPORTED_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "application/octet-stream",
}

logger = logging.getLogger(__name__)


class CameraUploadError(ValueError):
    pass


def normalize_content_type(content_type: str | None) -> str:
    return (content_type or "application/octet-stream").split(";", maxsplit=1)[0].strip().lower()


def safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:80] or "default"


def get_max_saved_images_per_device() -> int:
    raw_value = os.getenv("MAX_SAVED_IMAGES_PER_DEVICE", "10").strip()
    try:
        value = int(raw_value)
    except ValueError:
        return 10
    return max(value, 1)


def validate_jpeg_upload(image_bytes: bytes, content_type: str | None) -> str:
    normalized_content_type = normalize_content_type(content_type)
    if normalized_content_type not in SUPPORTED_IMAGE_CONTENT_TYPES:
        raise CameraUploadError("content-type must be image/jpeg")
    if not image_bytes:
        raise CameraUploadError("image body cannot be empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise CameraUploadError("image is too large")
    if len(image_bytes) < MIN_JPEG_BYTES:
        raise CameraUploadError("image is too small to be a complete JPEG")
    if not image_bytes.startswith(b"\xff\xd8\xff"):
        raise CameraUploadError("uploaded file is not a JPEG image")
    if not image_bytes.endswith(b"\xff\xd9"):
        raise CameraUploadError("JPEG image appears to be incomplete")
    return "image/jpeg"


def save_camera_image(
    device_id: str,
    image_bytes: bytes,
    content_type: str | None,
) -> dict[str, object]:
    normalized_content_type = validate_jpeg_upload(image_bytes, content_type)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    image_id = f"{timestamp}_{uuid4().hex[:8]}"
    device_dir = UPLOADS_DIR / safe_path_part(device_id)
    device_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{image_id}.jpg"
    path = device_dir / filename
    temporary_path = device_dir / f".{filename}.{uuid4().hex}.part"
    try:
        temporary_path.write_bytes(image_bytes)
        temporary_path.replace(path)
    except OSError:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    cleanup_old_camera_images(device_dir, get_max_saved_images_per_device())

    return {
        "image_id": image_id,
        "filename": filename,
        "path": str(path),
        "size_bytes": len(image_bytes),
        "content_type": normalized_content_type,
    }


def cleanup_old_camera_images(device_dir: Path, max_images: int) -> int:
    image_paths = sorted(
        (path for path in device_dir.glob("*.jpg") if path.is_file()),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    stale_paths = image_paths[max_images:]
    removed = 0

    for path in stale_paths:
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            logger.warning("camera.cleanup.failed path=%s error=%s", path, exc)

    if removed:
        logger.info(
            "camera.cleanup.done device_dir=%s kept=%d removed=%d",
            device_dir,
            max_images,
            removed,
        )
    return removed


def clean_optional_text(value: str | None) -> str | None:
    cleaned = (value or "").strip()
    return cleaned or None


def build_manual_vision_result(
    artifact: dict[str, Any],
    vision_description: str | None = None,
) -> dict[str, object]:
    evidence = [str(item) for item in artifact.get("recognition_features", [])[:4]]
    description = clean_optional_text(vision_description)
    if description is None:
        cues = "；".join(evidence) if evidence else "未提供额外视觉描述"
        description = f"当前图片标注为{artifact['name']}。可参考的视觉线索包括：{cues}。"

    return {
        "mode": "manual_artifact_id",
        "artifact_id": artifact["id"],
        "artifact_name": artifact["name"],
        "confidence": 1.0,
        "evidence": evidence,
        "vision_description": description,
    }


def build_unrecognized_vision_result(
    vision_description: str | None = None,
) -> dict[str, object]:
    return {
        "mode": "image_saved_only",
        "artifact_id": None,
        "artifact_name": None,
        "confidence": 0.0,
        "evidence": [],
        "vision_description": clean_optional_text(vision_description),
    }
