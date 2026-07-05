import json
import os
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from wav_utils import validate_device_wav


class AsrError(RuntimeError):
    pass


class AsrConfigError(AsrError):
    pass


class AsrCancelled(AsrError):
    pass


@dataclass
class AsrResult:
    text: str
    provider: str
    model: str
    raw_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DashScopeRealtimeCollector:
    final_texts: list[str] = field(default_factory=list)
    latest_text: str = ""
    error: str | None = None
    completed: bool = False
    raw_events: list[dict[str, Any]] = field(default_factory=list)

    def push_sentence(self, sentence: dict[str, Any]) -> None:
        self.raw_events.append(sentence)
        text = str(sentence.get("text") or "").strip()
        if not text:
            return

        self.latest_text = text
        if is_final_sentence(sentence):
            if not self.final_texts or self.final_texts[-1] != text:
                self.final_texts.append(text)

    def joined_text(self) -> str:
        if self.final_texts:
            return "".join(self.final_texts).strip()
        return self.latest_text.strip()


def transcribe_device_wav(
    wav_path: Path,
    should_continue: Callable[[], bool] | None = None,
) -> AsrResult:
    mock_text = os.getenv("AI_MOCK_ASR_TEXT", "").strip()
    if mock_text:
        return AsrResult(text=mock_text, provider="mock", model="AI_MOCK_ASR_TEXT")

    provider = os.getenv("ASR_PROVIDER", "dashscope").strip().lower()
    if provider == "dashscope":
        return transcribe_dashscope_realtime(wav_path, should_continue=should_continue)

    raise AsrConfigError(f"Unsupported ASR_PROVIDER: {provider}")


def get_asr_api_key() -> str:
    api_key = (
        os.getenv("ASR_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise AsrConfigError("ASR_API_KEY or DASHSCOPE_API_KEY is required")
    return api_key


def get_asr_model() -> str:
    return os.getenv("ASR_MODEL", "paraformer-realtime-v2").strip() or "paraformer-realtime-v2"


def transcribe_dashscope_realtime(
    wav_path: Path,
    should_continue: Callable[[], bool] | None = None,
) -> AsrResult:
    try:
        import dashscope
        from dashscope.audio.asr import Recognition, RecognitionCallback
    except ModuleNotFoundError as exc:
        raise AsrConfigError(
            "dashscope package is required for ASR. Run: uv pip install -r requirements.txt"
        ) from exc

    validate_device_wav(wav_path)
    dashscope.api_key = get_asr_api_key()

    collector = DashScopeRealtimeCollector()

    class Callback(RecognitionCallback):  # type: ignore[misc, valid-type]
        def on_event(self, result: object) -> None:
            sentence = extract_sentence(result)
            if sentence:
                collector.push_sentence(sentence)

        def on_error(self, *args: object) -> None:
            collector.error = " ".join(str(arg) for arg in args)

        def on_complete(self) -> None:
            collector.completed = True

    model = get_asr_model()
    recognition_kwargs: dict[str, Any] = {
        "model": model,
        "format": "pcm",
        "sample_rate": 16000,
        "callback": Callback(),
    }
    recognition_kwargs.update(load_asr_extra_kwargs())
    recognition = Recognition(**recognition_kwargs)

    try:
        recognition.start()
        send_wav_pcm_frames(
            recognition,
            wav_path,
            should_continue=should_continue,
        )
        recognition.stop()
    except AsrCancelled:
        try:
            recognition.stop()
        except Exception:
            pass
        raise
    except Exception as exc:
        raise AsrError(f"DashScope ASR failed: {exc}") from exc

    if collector.error:
        raise AsrError(f"DashScope ASR error: {collector.error}")

    return AsrResult(
        text=collector.joined_text(),
        provider="dashscope",
        model=model,
        raw_events=collector.raw_events,
    )


def load_asr_extra_kwargs() -> dict[str, Any]:
    extra_json = os.getenv("ASR_EXTRA_KWARGS", "").strip()
    if not extra_json:
        return {}
    try:
        loaded = json.loads(extra_json)
    except json.JSONDecodeError as exc:
        raise AsrConfigError(f"ASR_EXTRA_KWARGS is invalid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise AsrConfigError("ASR_EXTRA_KWARGS must be a JSON object")
    return loaded


def send_wav_pcm_frames(
    recognition: object,
    wav_path: Path,
    should_continue: Callable[[], bool] | None = None,
) -> None:
    frame_bytes = int(os.getenv("ASR_FRAME_BYTES", "3200"))
    sleep_seconds = float(os.getenv("ASR_FRAME_SLEEP_SECONDS", "0"))

    with wave.open(str(wav_path), "rb") as wav_file:
        bytes_per_frame = wav_file.getsampwidth() * wav_file.getnchannels()
        frames_per_chunk = max(1, frame_bytes // bytes_per_frame)

        while True:
            if should_continue is not None and not should_continue():
                raise AsrCancelled("ASR cancelled")

            data = wav_file.readframes(frames_per_chunk)
            if not data:
                break

            recognition.send_audio_frame(data)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)


def extract_sentence(result: object) -> dict[str, Any]:
    if hasattr(result, "get_sentence"):
        sentence = result.get_sentence()
        if isinstance(sentence, dict):
            return sentence

    if isinstance(result, dict):
        output = result.get("output")
        if isinstance(output, dict):
            sentence = output.get("sentence")
            if isinstance(sentence, dict):
                return sentence
        sentence = result.get("sentence")
        if isinstance(sentence, dict):
            return sentence
        if "text" in result:
            return result

    return {}


def is_final_sentence(sentence: dict[str, Any]) -> bool:
    final_fields = (
        "sentence_end",
        "is_final",
        "final",
        "end",
    )
    return any(bool(sentence.get(field)) for field in final_fields)
