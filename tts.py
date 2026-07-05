import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from wav_utils import WavFormatError, convert_wav_to_device_format, validate_device_wav, write_silence_wav


DEFAULT_TTS_ENDPOINT_PATH = "/audio/speech"


class TtsError(RuntimeError):
    pass


class TtsConfigError(TtsError):
    pass


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_tts_api_key() -> str:
    api_key = (
        os.getenv("TTS_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise TtsConfigError("TTS_API_KEY or DASHSCOPE_API_KEY is required")
    return api_key


def get_tts_base_url() -> str:
    base_url = (
        os.getenv("TTS_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or ""
    ).strip()
    if not base_url:
        raise TtsConfigError("TTS_BASE_URL or DASHSCOPE_BASE_URL is required")
    return base_url.rstrip("/")


def get_tts_endpoint() -> str:
    endpoint = os.getenv("TTS_ENDPOINT", "").strip()
    if endpoint:
        return endpoint
    return f"{get_tts_base_url()}{DEFAULT_TTS_ENDPOINT_PATH}"


def synthesize_to_device_wav(text: str, output_path: Path) -> dict[str, object]:
    cleaned = text.strip()
    if not cleaned:
        raise TtsError("TTS input text cannot be empty")

    if env_bool("AI_ENABLE_MOCK_TTS", default=False):
        duration = float(os.getenv("AI_MOCK_TTS_SECONDS", "0.4"))
        write_silence_wav(output_path, duration)
        wav_info = validate_device_wav(output_path)
        return {"provider": "mock", "path": str(output_path), "wav": wav_info}

    raw_audio = request_openai_compatible_tts(cleaned)
    return save_tts_response_as_device_wav(raw_audio, output_path)


def request_openai_compatible_tts(text: str) -> bytes:
    model = os.getenv("TTS_MODEL", "qwen3-tts-flash").strip()
    voice = os.getenv("TTS_VOICE", "Cherry").strip()
    response_format = os.getenv("TTS_RESPONSE_FORMAT", "wav").strip() or "wav"
    timeout = float(os.getenv("TTS_TIMEOUT_SECONDS", "120"))
    payload: dict[str, object] = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": response_format,
    }

    extra_json = os.getenv("TTS_EXTRA_JSON", "").strip()
    if extra_json:
        try:
            payload.update(json.loads(extra_json))
        except json.JSONDecodeError as exc:
            raise TtsConfigError(f"TTS_EXTRA_JSON is invalid JSON: {exc}") from exc

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        get_tts_endpoint(),
        data=body,
        headers={
            "Authorization": f"Bearer {get_tts_api_key()}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "audio/wav, application/octet-stream, */*",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise TtsError(f"TTS HTTP {exc.code}: {detail[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise TtsError(f"TTS connection failed: {exc.reason}") from exc


def save_tts_response_as_device_wav(raw_audio: bytes, output_path: Path) -> dict[str, object]:
    if not raw_audio:
        raise TtsError("TTS returned empty audio")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        dir=str(output_path.parent),
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(raw_audio)

    try:
        try:
            wav_info = convert_wav_to_device_format(temp_path, output_path)
        except WavFormatError as exc:
            raise TtsError(
                "TTS response is not a convertible PCM WAV. "
                "Set TTS_RESPONSE_FORMAT=wav, or add an audio transcoder before ESP32 playback."
            ) from exc
    finally:
        temp_path.unlink(missing_ok=True)

    return {"provider": os.getenv("TTS_PROVIDER", "dashscope"), "path": str(output_path), "wav": wav_info}
