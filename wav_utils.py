import audioop
import math
import sys
import tempfile
import wave
from array import array
from pathlib import Path


WAV_SAMPLE_RATE = 16000
WAV_SAMPLE_WIDTH = 2
WAV_CHANNELS = 1


class WavFormatError(ValueError):
    pass


def validate_device_wav(path: Path) -> dict[str, int | float]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            compression = wav_file.getcomptype()
    except (EOFError, wave.Error) as exc:
        raise WavFormatError(f"invalid WAV file: {exc}") from exc

    if compression != "NONE":
        raise WavFormatError("WAV must use PCM encoding")
    if channels != WAV_CHANNELS:
        raise WavFormatError("WAV must be mono")
    if sample_width != WAV_SAMPLE_WIDTH:
        raise WavFormatError("WAV must be 16-bit")
    if frame_rate != WAV_SAMPLE_RATE:
        raise WavFormatError("WAV sample rate must be 16000 Hz")

    return {
        "channels": channels,
        "sample_width": sample_width,
        "sample_rate": frame_rate,
        "frames": frame_count,
        "duration_seconds": frame_count / frame_rate if frame_rate else 0.0,
    }


def wav_rms(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())

    if not frames:
        return 0.0

    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0

    square_sum = sum(sample * sample for sample in samples)
    return math.sqrt(square_sum / len(samples))


def looks_like_silence(path: Path, rms_threshold: float = 80.0) -> bool:
    return wav_rms(path) < rms_threshold


def write_silence_wav(path: Path, duration_seconds: float = 0.4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(WAV_SAMPLE_RATE * duration_seconds))
    silence = b"\x00\x00" * frame_count
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(WAV_CHANNELS)
        wav_file.setsampwidth(WAV_SAMPLE_WIDTH)
        wav_file.setframerate(WAV_SAMPLE_RATE)
        wav_file.writeframes(silence)


def convert_wav_to_device_format(source_path: Path, target_path: Path) -> dict[str, int | float]:
    try:
        with wave.open(str(source_path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            frame_rate = source.getframerate()
            compression = source.getcomptype()
            frames = source.readframes(source.getnframes())
    except (EOFError, wave.Error) as exc:
        raise WavFormatError(f"invalid WAV file: {exc}") from exc

    if compression != "NONE":
        raise WavFormatError("WAV must use PCM encoding")
    if channels == 2:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
        channels = 1
    elif channels != 1:
        raise WavFormatError("only mono or stereo WAV can be converted")

    if sample_width != WAV_SAMPLE_WIDTH:
        frames = audioop.lin2lin(frames, sample_width, WAV_SAMPLE_WIDTH)
        sample_width = WAV_SAMPLE_WIDTH

    if frame_rate != WAV_SAMPLE_RATE:
        frames, _ = audioop.ratecv(
            frames,
            sample_width,
            channels,
            frame_rate,
            WAV_SAMPLE_RATE,
            None,
        )
        frame_rate = WAV_SAMPLE_RATE

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        dir=str(target_path.parent),
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        with wave.open(str(temp_path), "wb") as target:
            target.setnchannels(WAV_CHANNELS)
            target.setsampwidth(WAV_SAMPLE_WIDTH)
            target.setframerate(WAV_SAMPLE_RATE)
            target.writeframes(frames)
        temp_path.replace(target_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return validate_device_wav(target_path)
