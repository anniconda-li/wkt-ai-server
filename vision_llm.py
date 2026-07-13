import base64
import json
import logging
import os
from time import perf_counter
from typing import Any

from openai import AsyncOpenAI

from artifacts import load_artifacts
from vision import normalize_content_type


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_VISION_PROVIDER = "dashscope"
DEFAULT_VISION_MODEL = "qwen3.6-flash-2026-04-16"
UNKNOWN_ARTIFACT_ID = "unknown"

_vision_client: AsyncOpenAI | None = None
logger = logging.getLogger(__name__)


class VisionConfigError(RuntimeError):
    pass


class VisionRecognitionError(RuntimeError):
    pass


def get_vision_provider() -> str:
    return os.getenv("VISION_PROVIDER", DEFAULT_VISION_PROVIDER).strip().lower()


def get_vision_model() -> str:
    return os.getenv("VISION_MODEL", DEFAULT_VISION_MODEL).strip()


def get_vision_base_url() -> str:
    if os.getenv("VISION_BASE_URL"):
        return os.environ["VISION_BASE_URL"].strip()
    if os.getenv("DASHSCOPE_BASE_URL"):
        return os.environ["DASHSCOPE_BASE_URL"].strip()
    if get_vision_provider() == "dashscope":
        return DASHSCOPE_BASE_URL
    return ""


def get_vision_api_key() -> str | None:
    return os.getenv("VISION_API_KEY") or os.getenv("DASHSCOPE_API_KEY")


def get_min_confidence() -> float:
    raw_value = os.getenv("VISION_MIN_CONFIDENCE", "0.60")
    try:
        value = float(raw_value)
    except ValueError:
        return 0.60
    return min(max(value, 0.0), 1.0)


def is_thinking_enabled() -> bool:
    value = os.getenv("VISION_ENABLE_THINKING", "false")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_vision_configured() -> bool:
    return bool(get_vision_provider() and get_vision_model() and get_vision_api_key())


def validate_vision_config() -> None:
    if not get_vision_api_key():
        raise VisionConfigError("VISION_API_KEY or DASHSCOPE_API_KEY is not set")
    if not get_vision_model():
        raise VisionConfigError("VISION_MODEL is not set")
    if not get_vision_base_url():
        raise VisionConfigError("VISION_BASE_URL is not set")


def get_vision_client() -> AsyncOpenAI:
    global _vision_client
    validate_vision_config()

    if _vision_client is None:
        _vision_client = AsyncOpenAI(
            api_key=get_vision_api_key(),
            base_url=get_vision_base_url(),
        )
    return _vision_client


def elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000


def image_to_data_url(image_bytes: bytes, content_type: str | None) -> str:
    normalized_content_type = normalize_content_type(content_type)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{normalized_content_type};base64,{encoded}"


def build_vision_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for artifact in load_artifacts().values():
        candidates.append(
            {
                "artifact_id": artifact["id"],
                "name": artifact["name"],
                "aliases": artifact.get("aliases", []),
                "category": artifact.get("category"),
                "material": artifact.get("material"),
                "visual_keywords": artifact.get("visual_keywords", []),
                "recognition_features": artifact.get("recognition_features", []),
            }
        )
    return candidates


def build_recognition_prompt(candidates: list[dict[str, Any]]) -> str:
    return (
        "你是博物馆文物图片识别模块。请先判断图片里是否存在可辨认的文物主体，"
        "再根据图片内容识别候选文物。"
        "图片可能来自 ESP32 摄像头，可能有模糊、眩光、屏幕反拍、畸变、遮挡或低分辨率。"
        "图片质量差本身不是返回 unknown 的理由；只要还能看出器型、材质、轮廓、纹饰、"
        "孔洞、足、耳、钮、翅形等关键特征，并且这些特征与某个候选文物匹配，"
        "就应该返回该候选 artifact_id，并根据可见特征多少给出合理 confidence。"
        "如果图片是空白、黑屏、过曝、桌面、地面、墙面、手指、人物、普通物品、"
        "展厅环境但没有可辨认文物主体，或者只能看到过于泛化的物体轮廓、"
        "无法和任何候选文物的关键特征建立匹配，才返回 unknown。"
        "返回 unknown 时 confidence 必须小于 0.30。"
        "如果能看到 2 个以上候选关键特征，即使图片模糊，也不要轻易返回 unknown。"
        "你必须只从候选 artifact_id 或 unknown 中选择。"
        "不要编造候选列表以外的新文物。不要写讲解词。"
        "只输出一个 JSON 对象，不要输出 Markdown，不要输出额外解释。"
        "JSON 字段必须是："
        "artifact_id, confidence, evidence, vision_description。"
        "confidence 使用 0 到 1 的数字。"
        "evidence 是你从图片中看到的 1 到 6 条视觉证据。"
        "vision_description 用中文客观描述图片中看见的器物和拍摄情况。"
        "\n\n候选文物列表：\n"
        f"{json.dumps(candidates, ensure_ascii=False)}"
    )


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        return "\n".join(text_parts)
    return str(content or "")


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise VisionRecognitionError("vision model did not return JSON")
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise VisionRecognitionError("vision model returned invalid JSON") from exc


def coerce_confidence(value: Any) -> float:
    if isinstance(value, int | float):
        confidence = float(value)
    elif isinstance(value, str):
        cleaned = value.strip().rstrip("%")
        try:
            confidence = float(cleaned)
            if "%" in value or confidence > 1:
                confidence = confidence / 100
        except ValueError:
            confidence = 0.0
    else:
        confidence = 0.0
    return min(max(confidence, 0.0), 1.0)


def normalize_evidence(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()][:6]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def normalize_vision_result(raw_result: dict[str, Any]) -> dict[str, object]:
    artifacts = load_artifacts()
    candidate_ids = set(artifacts)
    predicted_artifact_id = str(raw_result.get("artifact_id") or UNKNOWN_ARTIFACT_ID).strip()
    if predicted_artifact_id not in candidate_ids:
        predicted_artifact_id = UNKNOWN_ARTIFACT_ID

    confidence = coerce_confidence(raw_result.get("confidence"))
    accepted = predicted_artifact_id != UNKNOWN_ARTIFACT_ID and confidence >= get_min_confidence()
    artifact_id = predicted_artifact_id if accepted else None
    artifact = artifacts.get(artifact_id) if artifact_id else None

    return {
        "mode": "vision_llm",
        "provider": get_vision_provider(),
        "model": get_vision_model(),
        "artifact_id": artifact_id,
        "artifact_name": artifact.get("name") if artifact else None,
        "predicted_artifact_id": None if predicted_artifact_id == UNKNOWN_ARTIFACT_ID else predicted_artifact_id,
        "confidence": confidence,
        "accepted": accepted,
        "min_confidence": get_min_confidence(),
        "evidence": normalize_evidence(raw_result.get("evidence")),
        "vision_description": str(raw_result.get("vision_description") or "").strip() or None,
    }


async def recognize_artifact_from_image(
    image_bytes: bytes,
    content_type: str | None,
) -> dict[str, object]:
    total_start = perf_counter()
    client = get_vision_client()
    candidates_start = perf_counter()
    candidates = build_vision_candidates()
    logger.info(
        "vision.recognition.start provider=%s model=%s image_bytes=%d candidates=%d",
        get_vision_provider(),
        get_vision_model(),
        len(image_bytes),
        len(candidates),
    )
    logger.info(
        "vision.recognition.stage build_candidates_ms=%.1f",
        elapsed_ms(candidates_start),
    )
    api_start = perf_counter()
    try:
        response = await client.chat.completions.create(
            model=get_vision_model(),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_data_url(image_bytes, content_type)},
                        },
                        {"type": "text", "text": build_recognition_prompt(candidates)},
                    ],
                }
            ],
            temperature=0,
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": is_thinking_enabled()},
        )
    except Exception as exc:
        raise VisionRecognitionError(
            f"vision model call failed: {type(exc).__name__}: {exc}"
        ) from exc
    logger.info("vision.recognition.stage api_call_ms=%.1f", elapsed_ms(api_start))

    if not response.choices:
        raise VisionRecognitionError("vision model returned no choices")

    parse_start = perf_counter()
    text = message_content_to_text(response.choices[0].message.content)
    raw_result = extract_json_object(text)
    result = normalize_vision_result(raw_result)
    logger.info(
        "vision.recognition.stage parse_result_ms=%.1f predicted_artifact_id=%s "
        "accepted=%s confidence=%s",
        elapsed_ms(parse_start),
        result.get("predicted_artifact_id"),
        result.get("accepted"),
        result.get("confidence"),
    )
    logger.info(
        "vision.recognition.done artifact_id=%s total_ms=%.1f",
        result.get("artifact_id"),
        elapsed_ms(total_start),
    )
    return result
