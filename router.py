import json
import logging
from time import perf_counter
from typing import AsyncIterator

from artifacts import ArtifactNotFoundError, find_artifacts_by_text, get_artifact, to_llm_context
from llm import stream_chat_completion
from sessions import DeviceSession, get_session, remember_turn
from tools import get_device_status


SYSTEM_PROMPT = (
    "You are a museum guide speaking directly to a visitor through a small "
    "device. Your answer is displayed to the visitor, so do not describe how "
    "to guide, teach, present, or explain. Just give the final visitor-facing "
    "answer. Use only the provided artifact facts as factual grounding. If "
    "details are marked as pending confirmation, say briefly that the exhibit "
    "record is still being completed instead of inventing dates, provenance, "
    "or stories. Keep the tone warm, clear, and suitable for listening."
)

logger = logging.getLogger(__name__)


def should_call_tool(user_message: str) -> bool:
    text = user_message.lower()
    return "status" in text or "device" in text


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
    logger.info(
        "chat.start device=%s input_chars=%d memory_messages=%d latest_artifact_id=%s",
        device_id,
        len(user_message),
        len(session.memory),
        session.latest_artifact_id,
    )

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
    assistant_message = "".join(chunks)
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
