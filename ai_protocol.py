import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from pipeline import generate_answer_text
from sessions import normalize_device_id
from tts import synthesize_to_device_wav
from wav_utils import WavFormatError, looks_like_silence, validate_device_wav


AI_CHUNK_SIZE = 8192
AI_RESULT_CHUNK_SIZE = 32768
AI_MAX_REQUEST_WAV_BYTES = 2_100_000
AI_MAX_REPLY_WAV_BYTES = 4_000_000
AI_ROOT = Path("uploads") / "ai"
TERMINAL_STATUSES = {"audio_ready", "audio_failed", "no_speech", "cancelled", "failed"}

logger = logging.getLogger(__name__)


class AiProtocolError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


@dataclass
class AiSession:
    session_id: str
    device_id: str
    language: str = "zh"
    status: str = "created"
    tts_status: str = "idle"
    tts_error: str | None = None
    error: str | None = None
    asr_text: str = ""
    answer_text: str = ""
    total_size: int = 0
    received_chunks: dict[int, int] = field(default_factory=dict)
    request_wav_path: Path | None = None
    reply_wav_path: Path | None = None
    reply_wav_size: int = 0
    cancel_requested: bool = False
    audio_stopped: bool = False
    task: asyncio.Task[None] | None = field(default=None, repr=False)
    created_at: str = field(default_factory=lambda: now_iso())
    updated_at: str = field(default_factory=lambda: now_iso())

    @property
    def received_bytes(self) -> int:
        return sum(self.received_chunks.values())

    def touch(self) -> None:
        self.updated_at = now_iso()


ai_sessions: dict[str, AiSession] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000


def create_ai_session(device: str | None, language: str | None = "zh") -> AiSession:
    device_id = normalize_device_id(device)
    session_id = uuid4().hex
    session_dir = AI_ROOT / device_id / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    session = AiSession(
        session_id=session_id,
        device_id=device_id,
        language=(language or "zh").strip() or "zh",
        request_wav_path=session_dir / "request.wav",
        reply_wav_path=session_dir / "reply.wav",
    )
    ai_sessions[session_id] = session
    logger.info("ai.start session=%s device=%s language=%s", session_id, device_id, session.language)
    return session


def get_ai_session(session_id: str) -> AiSession:
    session = ai_sessions.get(session_id)
    if session is None:
        raise AiProtocolError(404, "session not found")
    return session


def validate_session_device(session: AiSession, device: str | None) -> None:
    if not device:
        return
    device_id = normalize_device_id(device)
    if device_id != session.device_id:
        raise AiProtocolError(409, "device does not match session")


def set_status(session: AiSession, status: str, *, error: str | None = None) -> None:
    session.status = status
    session.error = error
    session.touch()


def ai_result_info(session: AiSession) -> dict[str, object]:
    audio_ready = (
        session.status == "audio_ready"
        and session.reply_wav_path is not None
        and session.reply_wav_size > 0
        and not session.audio_stopped
    )
    return {
        "ok": True,
        "session": session.session_id,
        "device": session.device_id,
        "status": session.status,
        "asr_text": session.asr_text,
        "answer_text": session.answer_text,
        "audio_ready": audio_ready,
        "reply_wav_ready": audio_ready,
        "ready": audio_ready,
        "reply_wav_size": session.reply_wav_size if audio_ready else 0,
        "total": session.reply_wav_size if audio_ready else 0,
        "tts_status": session.tts_status,
        "tts_error": session.tts_error,
        "error": session.error,
        "received_bytes": session.received_bytes,
        "total_size": session.total_size,
        "cancel_requested": session.cancel_requested,
        "audio_stopped": session.audio_stopped,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


def write_ai_upload_chunk(
    session_id: str,
    chunk: bytes,
    *,
    index: int,
    offset: int,
    total: int,
    device: str | None = None,
) -> dict[str, object]:
    session = get_ai_session(session_id)
    validate_session_device(session, device)

    if session.cancel_requested or session.status == "cancelled":
        raise AiProtocolError(409, "session cancelled")
    if total <= 0 or total > AI_MAX_REQUEST_WAV_BYTES:
        raise AiProtocolError(400, "invalid total size")
    if offset < 0 or index < 0:
        raise AiProtocolError(400, "invalid chunk index or offset")
    if offset + len(chunk) > total:
        raise AiProtocolError(400, "chunk exceeds total size")
    if session.request_wav_path is None:
        raise AiProtocolError(500, "request path not initialized")

    session.total_size = total
    set_status(session, "uploading")
    session.request_wav_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "r+b" if session.request_wav_path.exists() else "w+b"
    with session.request_wav_path.open(mode) as output:
        output.seek(offset)
        output.write(chunk)

    session.received_chunks[offset] = len(chunk)
    session.touch()
    logger.info(
        "ai.upload session=%s device=%s index=%d offset=%d bytes=%d received=%d total=%d",
        session.session_id,
        session.device_id,
        index,
        offset,
        len(chunk),
        session.received_bytes,
        total,
    )
    return {"ok": True}


def finish_ai_upload(session_id: str, device: str | None = None) -> dict[str, object]:
    session = get_ai_session(session_id)
    validate_session_device(session, device)

    if session.cancel_requested or session.status == "cancelled":
        return ai_result_info(session)
    if session.request_wav_path is None or not session.request_wav_path.exists():
        raise AiProtocolError(400, "request WAV not uploaded")
    if session.total_size and session.received_bytes < session.total_size:
        set_status(session, "failed", error="incomplete_upload")
        return ai_result_info(session)

    set_status(session, "uploaded")
    if session.task is None or session.task.done():
        session.task = asyncio.create_task(process_ai_session(session.session_id))
    return ai_result_info(session)


def cancel_ai_session(session_id: str, device: str | None = None) -> dict[str, object]:
    session = get_ai_session(session_id)
    validate_session_device(session, device)
    session.cancel_requested = True
    session.tts_status = "cancelled"
    set_status(session, "cancelled")
    logger.info("ai.cancel session=%s device=%s", session.session_id, session.device_id)
    return ai_result_info(session)


def stop_ai_audio(session_id: str, device: str | None = None) -> dict[str, object]:
    session = get_ai_session(session_id)
    validate_session_device(session, device)
    session.audio_stopped = True
    if session.status in {"text_ready", "tts_running"}:
        session.tts_status = "stopped"
        set_status(session, "text_ready")
    else:
        session.touch()
    logger.info("ai.stop_audio session=%s device=%s", session.session_id, session.device_id)
    return ai_result_info(session)


def read_ai_result_chunk(
    session_id: str,
    *,
    offset: int,
    length: int,
    device: str | None = None,
) -> bytes:
    session = get_ai_session(session_id)
    validate_session_device(session, device)

    if session.cancel_requested or session.audio_stopped:
        raise AiProtocolError(409, "audio stopped or session cancelled")
    if session.status != "audio_ready" or session.reply_wav_path is None:
        raise AiProtocolError(409, "reply WAV is not ready")
    if length <= 0 or length > AI_RESULT_CHUNK_SIZE:
        raise AiProtocolError(400, "invalid result chunk length")
    if offset < 0 or offset >= session.reply_wav_size:
        raise AiProtocolError(416, "offset out of range")

    read_length = min(length, session.reply_wav_size - offset)
    with session.reply_wav_path.open("rb") as wav_file:
        wav_file.seek(offset)
        return wav_file.read(read_length)


async def process_ai_session(session_id: str) -> None:
    session = get_ai_session(session_id)
    total_start = perf_counter()
    logger.info("ai.process.start session=%s device=%s", session.session_id, session.device_id)
    try:
        await run_asr_stage(session)
        if session.status in TERMINAL_STATUSES:
            return

        await run_llm_stage(session)
        if session.status in TERMINAL_STATUSES or session.cancel_requested:
            return

        await run_tts_stage(session)
    except Exception as exc:
        logger.exception("ai.process.failed session=%s error=%s", session.session_id, exc)
        if not session.cancel_requested:
            session.tts_status = "failed"
            set_status(session, "failed", error=str(exc))
    finally:
        logger.info(
            "ai.process.done session=%s device=%s status=%s total_ms=%.1f",
            session.session_id,
            session.device_id,
            session.status,
            elapsed_ms(total_start),
        )


async def run_asr_stage(session: AiSession) -> None:
    if session.cancel_requested:
        set_status(session, "cancelled")
        return

    start = perf_counter()
    set_status(session, "asr_running")
    if session.request_wav_path is None:
        raise RuntimeError("request WAV path missing")

    try:
        wav_info = validate_device_wav(session.request_wav_path)
    except WavFormatError as exc:
        set_status(session, "failed", error=str(exc))
        return

    silence_threshold = float(os.getenv("AI_SILENCE_RMS_THRESHOLD", "80"))
    min_duration = float(os.getenv("AI_MIN_SPEECH_SECONDS", "0.2"))
    if wav_info["duration_seconds"] < min_duration or looks_like_silence(
        session.request_wav_path, silence_threshold
    ):
        session.answer_text = os.getenv("AI_NO_SPEECH_TEXT", "我没有听清，请再说一遍。")
        session.tts_status = "skipped"
        set_status(session, "no_speech")
        logger.info(
            "ai.asr.no_speech session=%s duration=%.2f rms_threshold=%.1f ms=%.1f",
            session.session_id,
            wav_info["duration_seconds"],
            silence_threshold,
            elapsed_ms(start),
        )
        return

    mock_text = os.getenv("AI_MOCK_ASR_TEXT", "").strip()
    if not mock_text:
        session.answer_text = os.getenv("AI_NO_SPEECH_TEXT", "我没有听清，请再说一遍。")
        session.tts_status = "skipped"
        set_status(session, "no_speech")
        logger.info("ai.asr.unconfigured session=%s ms=%.1f", session.session_id, elapsed_ms(start))
        return

    session.asr_text = mock_text
    session.touch()
    logger.info(
        "ai.asr.done session=%s chars=%d ms=%.1f",
        session.session_id,
        len(session.asr_text),
        elapsed_ms(start),
    )


async def run_llm_stage(session: AiSession) -> None:
    if session.cancel_requested:
        set_status(session, "cancelled")
        return

    start = perf_counter()
    set_status(session, "llm_running")

    answer = (
        await generate_answer_text(
            session.device_id,
            session.asr_text,
            should_continue=lambda: not session.cancel_requested,
        )
    ).strip()
    if session.cancel_requested:
        set_status(session, "cancelled")
        return
    if answer.startswith("[LLM_ERROR]"):
        raise RuntimeError(answer)

    session.answer_text = answer
    session.tts_status = "pending"
    set_status(session, "text_ready")
    logger.info(
        "ai.llm.done session=%s answer_chars=%d ms=%.1f",
        session.session_id,
        len(answer),
        elapsed_ms(start),
    )


async def run_tts_stage(session: AiSession) -> None:
    if session.cancel_requested:
        set_status(session, "cancelled")
        return
    if session.audio_stopped:
        session.tts_status = "stopped"
        set_status(session, "text_ready")
        return

    start = perf_counter()
    session.tts_status = "running"
    set_status(session, "tts_running")

    if session.reply_wav_path is None:
        raise RuntimeError("reply WAV path missing")

    try:
        await asyncio.to_thread(
            synthesize_to_device_wav,
            session.answer_text,
            session.reply_wav_path,
        )
    except Exception as exc:
        session.tts_status = "failed"
        session.tts_error = str(exc)
        set_status(session, "audio_failed")
        logger.info(
            "ai.tts.failed session=%s error=%s ms=%.1f",
            session.session_id,
            exc,
            elapsed_ms(start),
        )
        return

    if session.cancel_requested:
        set_status(session, "cancelled")
        return
    if session.audio_stopped:
        session.tts_status = "stopped"
        set_status(session, "text_ready")
        return

    reply_size = session.reply_wav_path.stat().st_size
    if reply_size > AI_MAX_REPLY_WAV_BYTES:
        raise RuntimeError("reply WAV exceeds maximum size")

    session.reply_wav_size = reply_size
    session.tts_status = "done"
    session.tts_error = None
    set_status(session, "audio_ready")
    logger.info(
        "ai.tts.done session=%s bytes=%d ms=%.1f",
        session.session_id,
        reply_size,
        elapsed_ms(start),
    )
