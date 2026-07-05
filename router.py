import json
import logging
from typing import AsyncIterator

from artifacts import find_artifacts_by_text, to_llm_context
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


def build_messages(user_message: str, session: DeviceSession) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(session.memory)

    artifact_matches = find_artifacts_by_text(user_message)
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
