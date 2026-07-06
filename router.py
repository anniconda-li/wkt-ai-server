import json
import logging
from time import perf_counter
from typing import AsyncIterator, Callable

from artifacts import ArtifactNotFoundError, find_artifacts_by_text, get_artifact, to_llm_context
from llm import stream_chat_completion
from output_format import format_for_device_display
from sessions import DeviceSession, get_session, remember_turn
from tools import get_device_status


SYSTEM_PROMPT = (
    "You are a museum guide speaking directly to a visitor through a small "
    "device. Your answer is displayed to the visitor, so do not describe how "
    "to guide, teach, present, or explain. Just give the final visitor-facing "
    "answer. Use only the provided artifact facts as factual grounding. If "
    "details are marked as pending confirmation, say briefly that the exhibit "
    "record is still being completed instead of inventing dates, provenance, "
    "or stories. If the visitor asks who you are, say that you are the AI "
    "museum guide assistant for Pingdingshan Museum, and answer directly. "
    "Do not mention special exhibitions, current exhibits, featured exhibits, "
    "or today-specific arrangements unless that information is explicitly "
    "provided in the context. Do not end with an offer that assumes a specific "
    "artifact or exhibition exists. Output plain text only: no Markdown, no "
    "headings, no bullet lists, no numbering, no ** emphasis markers, and no "
    "escaped newline symbols. Keep the tone warm, clear, and suitable for "
    "listening."
)

logger = logging.getLogger(__name__)


def should_call_tool(user_message: str) -> bool:
    text = user_message.lower()
    return "status" in text or "device" in text


def is_identity_question(user_message: str) -> bool:
    text = user_message.replace(" ", "")
    return any(pattern in text for pattern in ("你是谁", "你是啥", "你叫什么", "介绍一下你自己"))


def is_vague_intro_request(user_message: str) -> bool:
    text = user_message.replace(" ", "")
    patterns = (
        "介绍一下",
        "讲一下",
        "讲讲",
        "说说",
        "继续讲",
        "继续说",
        "那介绍",
        "那讲",
        "这是什么",
    )
    return any(pattern in text for pattern in patterns)


def direct_response_for(user_message: str, session: DeviceSession) -> str | None:
    if is_identity_question(user_message):
        return (
            "你好，我是平顶山市博物馆的 AI 讲解助手。你可以拍一张展品照片，"
            "或者直接说出文物名称，我会根据已配置的馆方资料为你做简短讲解。"
        )

    if not is_vague_intro_request(user_message):
        return None
    if find_artifacts_by_text(user_message) or session.latest_artifact_id:
        return None

    if session.latest_vision_description:
        return (
            "这张图片目前还没有匹配到具体文物。你可以重新拍清楚展品主体，"
            "或者直接告诉我文物名称，我再为你讲解。"
        )

    return "我还不知道你想让我介绍哪件展品。请先拍一张展品照片，或者直接说出文物名称。"


def resolve_artifact_context(
    user_message: str, session: DeviceSession
) -> list[dict[str, object]]:
    artifact_matches = find_artifacts_by_text(user_message)
    if artifact_matches:
        session.latest_artifact_id = str(artifact_matches[0]["id"])
        return artifact_matches

    if session.latest_artifact_id:
        try:
            return [get_artifact(session.latest_artifact_id)]
        except ArtifactNotFoundError:
            logger.warning("Latest artifact no longer exists: %s", session.latest_artifact_id)
            session.latest_artifact_id = None

    return []


def build_messages(user_message: str, session: DeviceSession) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(session.memory)

    artifact_matches = resolve_artifact_context(user_message, session)
    if artifact_matches:
        artifact_context = [to_llm_context(artifact) for artifact in artifact_matches]
        messages.append(
            {
                "role": "system",
                "content": (
                    "Matched local artifact knowledge cards. These are facts "
                    "for generating the final visitor-facing answer. Do not "
                    "say 'you can explain', do not give instructions to a "
                    "guide, and do not describe the answer strategy:\n"
                    f"{json.dumps(artifact_context, ensure_ascii=False)}"
                ),
            }
        )
    else:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Current device context: no artifact has been recognized "
                    "or selected for this device. If the visitor asks a vague "
                    "follow-up such as '介绍一下吧', '讲讲吧', '继续', or "
                    "'这是什么' without a current image/artifact context, do "
                    "not invent an exhibit, special exhibition, or object. "
                    "Ask the visitor to take a photo of an exhibit or say the "
                    "artifact name."
                ),
            }
        )

    if session.latest_vision_description and artifact_matches:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Latest visual observation for the current device. Use it "
                    "only as supporting visual context, and do not mention "
                    "internal fields or system context:\n"
                    f"{session.latest_vision_description}"
                ),
            }
        )

    if should_call_tool(user_message):
        tool_result = get_device_status()
        messages.append(
            {
                "role": "system",
                "content": (
                    "Tool result from get_device_status():\n"
                    f"{json.dumps(tool_result, ensure_ascii=False)}"
                ),
            }
        )

    messages.append({"role": "user", "content": user_message})
    return messages


def elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000


async def chat_stream(user_message: str, device_id: str) -> AsyncIterator[str]:
    total_start = perf_counter()
    session = get_session(device_id)
    chunks: list[str] = []
    direct_response = direct_response_for(user_message, session)
    logger.info(
        "chat.start device=%s input_chars=%d memory_messages=%d latest_artifact_id=%s",
        device_id,
        len(user_message),
        len(session.memory),
        session.latest_artifact_id,
    )

    if direct_response is not None:
        assistant_message = format_for_device_display(direct_response)
        remember_turn(session, user_message, assistant_message)
        logger.info(
            "chat.direct device=%s output_chars=%d total_ms=%.1f",
            device_id,
            len(assistant_message),
            elapsed_ms(total_start),
        )
        yield assistant_message
        return

    build_start = perf_counter()
    messages = build_messages(user_message, session)
    logger.info(
        "chat.stage build_messages_ms=%.1f device=%s messages=%d latest_artifact_id=%s",
        elapsed_ms(build_start),
        device_id,
        len(messages),
        session.latest_artifact_id,
    )

    try:
        stream_start = perf_counter()
        first_token_seen = False
        async for token in stream_chat_completion(messages):
            if not first_token_seen:
                first_token_seen = True
                logger.info(
                    "chat.stage first_token_ms=%.1f device=%s",
                    elapsed_ms(stream_start),
                    device_id,
                )
            chunks.append(token)
            yield token
    except Exception as exc:
        logger.exception(
            "chat.failed device=%s error=%s total_ms=%.1f",
            device_id,
            type(exc).__name__,
            elapsed_ms(total_start),
        )
        yield f"\n[LLM_ERROR] {type(exc).__name__}: {exc}\n"
        return

    remember_start = perf_counter()
    assistant_message = format_for_device_display("".join(chunks))
    remember_turn(session, user_message, assistant_message)
    logger.info(
        "chat.stage remember_turn_ms=%.1f device=%s memory_messages=%d",
        elapsed_ms(remember_start),
        device_id,
        len(session.memory),
    )
    logger.info(
        "chat.done device=%s output_chars=%d total_ms=%.1f",
        device_id,
        len(assistant_message),
        elapsed_ms(total_start),
    )


async def chat_response(
    user_message: str,
    device_id: str,
    should_continue: Callable[[], bool] | None = None,
) -> str:
    total_start = perf_counter()
    session = get_session(device_id)
    direct_response = direct_response_for(user_message, session)
    if direct_response is not None:
        assistant_message = format_for_device_display(direct_response)
        remember_turn(session, user_message, assistant_message)
        logger.info(
            "chat_once.direct device=%s output_chars=%d total_ms=%.1f",
            device_id,
            len(assistant_message),
            elapsed_ms(total_start),
        )
        return assistant_message

    messages = build_messages(user_message, session)
    chunks: list[str] = []
    logger.info(
        "chat_once.start device=%s input_chars=%d latest_artifact_id=%s",
        device_id,
        len(user_message),
        session.latest_artifact_id,
    )

    async for token in stream_chat_completion(messages):
        if should_continue is not None and not should_continue():
            logger.info("chat_once.cancelled device=%s", device_id)
            raise RuntimeError("chat cancelled")
        chunks.append(token)

    if should_continue is not None and not should_continue():
        logger.info("chat_once.cancelled device=%s", device_id)
        raise RuntimeError("chat cancelled")

    assistant_message = format_for_device_display("".join(chunks))
    remember_turn(session, user_message, assistant_message)
    logger.info(
        "chat_once.done device=%s output_chars=%d total_ms=%.1f",
        device_id,
        len(assistant_message),
        elapsed_ms(total_start),
    )
    return assistant_message
