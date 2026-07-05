import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable
from uuid import uuid4

from llm import validate_llm_config
from router import chat_response
from sessions import normalize_device_id
from tts import synthesize_to_device_wav


DEFAULT_REPLY_DIR = Path("outputs") / "replies"


@dataclass
class TextPipelineResult:
    device_id: str
    user_text: str
    answer_text: str
    reply_wav_path: str | None
    reply_wav_size: int
    wav_info: dict[str, object] | None
    timings_ms: dict[str, float]


def elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000


def new_reply_wav_path(device_id: str, output_dir: Path = DEFAULT_REPLY_DIR) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return output_dir / normalize_device_id(device_id) / f"{stamp}_{uuid4().hex[:8]}.wav"


async def generate_answer_text(
    device_id: str,
    user_text: str,
    should_continue: Callable[[], bool] | None = None,
) -> str:
    cleaned = user_text.strip()
    if not cleaned:
        raise ValueError("user_text cannot be empty")

    mock_answer = os.getenv("AI_MOCK_LLM_TEXT", "").strip()
    if mock_answer:
        return mock_answer

    validate_llm_config()
    return (
        await chat_response(
            cleaned,
            normalize_device_id(device_id),
            should_continue=should_continue,
        )
    ).strip()


async def run_text_pipeline(
    device_id: str,
    user_text: str,
    *,
    output_dir: Path = DEFAULT_REPLY_DIR,
    enable_tts: bool = True,
    should_continue: Callable[[], bool] | None = None,
) -> TextPipelineResult:
    total_start = perf_counter()
    normalized_device = normalize_device_id(device_id)

    llm_start = perf_counter()
    answer_text = await generate_answer_text(
        normalized_device,
        user_text,
        should_continue=should_continue,
    )
    timings = {"llm": elapsed_ms(llm_start)}

    reply_wav_path: Path | None = None
    wav_info: dict[str, object] | None = None
    reply_wav_size = 0

    if enable_tts:
        if should_continue is not None and not should_continue():
            raise RuntimeError("pipeline cancelled before TTS")
        tts_start = perf_counter()
        reply_wav_path = new_reply_wav_path(normalized_device, output_dir)
        tts_result = await asyncio.to_thread(
            synthesize_to_device_wav,
            answer_text,
            reply_wav_path,
        )
        wav_info = dict(tts_result["wav"])
        reply_wav_size = reply_wav_path.stat().st_size
        timings["tts"] = elapsed_ms(tts_start)

    timings["total"] = elapsed_ms(total_start)
    return TextPipelineResult(
        device_id=normalized_device,
        user_text=user_text,
        answer_text=answer_text,
        reply_wav_path=str(reply_wav_path) if reply_wav_path else None,
        reply_wav_size=reply_wav_size,
        wav_info=wav_info,
        timings_ms=timings,
    )
