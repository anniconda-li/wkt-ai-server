import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from wav_utils import WavFormatError, convert_wav_to_device_format, validate_device_wav, write_silence_wav


DEFAULT_TTS_ENDPOINT_PATH = "/audio/speech"
DASHSCOPE_QWEN_TTS_PATH = "/services/aigc/multimodal-generation/generation"


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

    style = get_tts_api_style()
    if style == "dashscope_qwen":
        raw_audio = request_dashscope_qwen_tts(cleaned)
    else:
        raw_audio = request_openai_compatible_tts(cleaned)
    return save_tts_response_as_device_wav(raw_audio, output_path)


def get_tts_api_style() -> str:
    configured = os.getenv("TTS_API_STYLE", "").strip().lower()
    if configured:
        return configured
    provider = os.getenv("TTS_PROVIDER", "").strip().lower()
    model = os.getenv("TTS_MODEL", "").strip().lower()
    if provider == "dashscope" or model.startswith("qwen3-tts"):
        return "dashscope_qwen"
    return "openai_speech"


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


def get_dashscope_tts_base_url() -> str:
    base_url = (
        os.getenv("TTS_BASE_URL")
        or os.getenv("DASHSCOPE_TTS_BASE_URL")
        or ""
    ).strip()
    if not base_url:
        raise TtsConfigError(
            "DashScope Qwen-TTS requires TTS_BASE_URL, for example "
            "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1"
        )
    if "compatible-mode" in base_url:
        raise TtsConfigError(
            "TTS_BASE_URL for Qwen-TTS cannot be the compatible-mode URL. "
            "Use the Model Studio workspace URL, for example "
            "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1"
        )
    return base_url.rstrip("/")


def request_dashscope_qwen_tts(text: str) -> bytes:
    model = os.getenv("TTS_MODEL", "qwen3-tts-flash").strip()
    voice = os.getenv("TTS_VOICE", "Cherry").strip()
    language_type = os.getenv("TTS_LANGUAGE_TYPE", "Chinese").strip() or "Chinese"
    timeout = float(os.getenv("TTS_TIMEOUT_SECONDS", "120"))
    endpoint = os.getenv("TTS_ENDPOINT", "").strip()
    if not endpoint:
        endpoint = f"{get_dashscope_tts_base_url()}{DASHSCOPE_QWEN_TTS_PATH}"

    payload: dict[str, object] = {
        "model": model,
        "input": {
            "text": text,
            "voice": voice,
            "language_type": language_type,
        },
    }

    extra_json = os.getenv("TTS_EXTRA_JSON", "").strip()
    if extra_json:
        try:
            payload.update(json.loads(extra_json))
        except json.JSONDecodeError as exc:
            raise TtsConfigError(f"TTS_EXTRA_JSON is invalid JSON: {exc}") from exc

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {get_tts_api_key()}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise TtsError(f"TTS HTTP {exc.code}: {detail[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise TtsError(f"TTS connection failed: {exc.reason}") from exc

    audio_url = extract_audio_url(response_body)
    return download_audio_url(audio_url, timeout)


def extract_audio_url(response_body: bytes) -> str:
    try:
        payload = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TtsError("TTS response is not JSON and did not contain direct audio") from exc

    output = payload.get("output") or {}
    audio = output.get("audio") or {}
    audio_url = audio.get("url") or output.get("url") or payload.get("url")
    if not audio_url:
        raise TtsError(f"TTS response did not include audio url: {str(payload)[:1000]}")
    return str(audio_url)


def download_audio_url(audio_url: str, timeout: float) -> bytes:
    try:
        with urllib.request.urlopen(audio_url, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise TtsError(f"TTS audio download HTTP {exc.code}: {detail[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise TtsError(f"TTS audio download failed: {exc.reason}") from exc


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
