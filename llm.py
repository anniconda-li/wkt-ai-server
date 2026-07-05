import os
from collections.abc import AsyncIterator

from openai import AsyncOpenAI


DEFAULT_MODEL = "gpt-4o-mini"

_client: AsyncOpenAI | None = None


def validate_llm_config() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")


def get_model() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def get_client() -> AsyncOpenAI:
    global _client
    validate_llm_config()

    if _client is None:
        kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
        if os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]

        _client = AsyncOpenAI(**kwargs)
    return _client


async def stream_chat_completion(
    messages: list[dict[str, str]],
) -> AsyncIterator[str]:
    client = get_client()

    stream = await client.chat.completions.create(
        model=get_model(),
        messages=messages,
        stream=True,
        temperature=0.7,
    )

    async for chunk in stream:
        if not chunk.choices:
            continue

        token = chunk.choices[0].delta.content
        if token:
            yield token
