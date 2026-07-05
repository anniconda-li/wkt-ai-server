import json
import logging
from typing import AsyncIterator

from llm import stream_chat_completion
from tools import get_device_status


MAX_MEMORY_ROUNDS = 10

SYSTEM_PROMPT = (
    "You are a concise AI assistant. If tool context is provided, use it as "
    "fresh device telemetry and mention relevant values when helpful."
)

memory: list[dict[str, str]] = []
logger = logging.getLogger(__name__)


def should_call_tool(user_message: str) -> bool:
    text = user_message.lower()
    return "status" in text or "device" in text


def build_messages(user_message: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(memory)

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


def remember_turn(user_message: str, assistant_message: str) -> None:
    memory.append({"role": "user", "content": user_message})
    memory.append({"role": "assistant", "content": assistant_message})
    del memory[:-MAX_MEMORY_ROUNDS * 2]


async def chat_stream(user_message: str) -> AsyncIterator[str]:
    chunks: list[str] = []

    try:
        async for token in stream_chat_completion(build_messages(user_message)):
            chunks.append(token)
            yield token
    except Exception as exc:
        logger.exception("LLM streaming failed")
        yield f"\n[LLM_ERROR] {type(exc).__name__}: {exc}\n"
        return

    remember_turn(user_message, "".join(chunks))
