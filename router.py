import json
import logging
from typing import AsyncIterator

from artifacts import find_artifacts_by_text
from llm import stream_chat_completion
from sessions import DeviceSession, get_session, remember_turn
from tools import get_device_status


SYSTEM_PROMPT = (
    "You are a concise AI assistant. If tool context is provided, use it as "
    "fresh device telemetry and mention relevant values when helpful."
)

logger = logging.getLogger(__name__)


def should_call_tool(user_message: str) -> bool:
    text = user_message.lower()
    return "status" in text or "device" in text


def build_messages(user_message: str, session: DeviceSession) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(session.memory)

    artifact_matches = find_artifacts_by_text(user_message)
    if artifact_matches:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Local artifact knowledge cards matched by the user message:\n"
                    f"{json.dumps(artifact_matches, ensure_ascii=False)}"
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


async def chat_stream(user_message: str, device_id: str) -> AsyncIterator[str]:
    session = get_session(device_id)
    chunks: list[str] = []

    try:
        async for token in stream_chat_completion(build_messages(user_message, session)):
            chunks.append(token)
            yield token
    except Exception as exc:
        logger.exception("LLM streaming failed")
        yield f"\n[LLM_ERROR] {type(exc).__name__}: {exc}\n"
        return

    remember_turn(session, user_message, "".join(chunks))
