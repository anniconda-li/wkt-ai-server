from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import struct
import subprocess
from tempfile import NamedTemporaryFile
import wave

from opus_packets import AOP1_MAX_PACKET_BYTES, _build_ogg_page, _packet_lacing


ROP1_MAGIC = b"ROP1"
ROP1_VERSION = 1
ROP1_HEADER_LEN = 28
ROP1_CHANNELS = 1
ROP1_SAMPLE_RATE = 16_000
ROP1_FRAME_SAMPLES = 320
ROP1_FRAME_MS = 20
ROP1_TARGET_BITRATE = 20_000
ROP1_MAX_FILE_BYTES = 384 * 1024
_HEADER = struct.Struct("<4sBBHIHHIIHH")


class Rop1Error(ValueError):
    pass


@dataclass(frozen=True)
class Rop1Audio:
    frame_count: int
    pcm_samples: int
    pre_skip: int
    end_trim: int
    packets: tuple[bytes, ...]

    @property
    def duration_ms(self) -> int:
        return round(self.pcm_samples * 1000 / ROP1_SAMPLE_RATE)


@dataclass(frozen=True)
class Rop1File:
    path: Path
    total: int
    sha256: str
    audio: Rop1Audio


def build_rop1_bytes(audio: Rop1Audio) -> bytes:
    if audio.frame_count != len(audio.packets) or audio.frame_count <= 0:
        raise Rop1Error("ROP1 frame_count does not match packets")
    if audio.pcm_samples <= 0:
        raise Rop1Error("ROP1 pcm_samples must be positive")
    if min(audio.pre_skip, audio.end_trim) < 0 or max(audio.pre_skip, audio.end_trim) > 0xFFFF:
        raise Rop1Error("ROP1 trim fields are out of range")
    body = bytearray()
    for index, packet in enumerate(audio.packets):
        if not packet or len(packet) > AOP1_MAX_PACKET_BYTES:
            raise Rop1Error(f"ROP1 packet {index} has an invalid length")
        body += struct.pack("<H", len(packet))
        body += packet
    payload = _HEADER.pack(
        ROP1_MAGIC,
        ROP1_VERSION,
        ROP1_CHANNELS,
        ROP1_HEADER_LEN,
        ROP1_SAMPLE_RATE,
        ROP1_FRAME_SAMPLES,
        ROP1_FRAME_MS,
        audio.frame_count,
        audio.pcm_samples,
        audio.pre_skip,
        audio.end_trim,
    ) + bytes(body)
    if len(payload) > ROP1_MAX_FILE_BYTES:
        raise Rop1Error("ROP1 reply exceeds 384 KiB")
    return payload


def parse_rop1_bytes(data: bytes) -> Rop1Audio:
    if len(data) < ROP1_HEADER_LEN:
        raise Rop1Error("ROP1 file is shorter than its header")
    if len(data) > ROP1_MAX_FILE_BYTES:
        raise Rop1Error("ROP1 reply exceeds 384 KiB")
    magic, version, channels, header_len, sample_rate, frame_samples, frame_ms, frame_count, pcm_samples, pre_skip, end_trim = _HEADER.unpack_from(data)
    if magic != ROP1_MAGIC or version != ROP1_VERSION:
        raise Rop1Error("invalid or unsupported ROP1 header")
    if channels != ROP1_CHANNELS or header_len != ROP1_HEADER_LEN:
        raise Rop1Error("unsupported ROP1 channels or header length")
    if sample_rate != ROP1_SAMPLE_RATE or frame_samples != ROP1_FRAME_SAMPLES or frame_ms != ROP1_FRAME_MS:
        raise Rop1Error("unsupported ROP1 audio parameters")
    if frame_count <= 0 or pcm_samples <= 0:
        raise Rop1Error("invalid ROP1 frame or sample count")
    packets: list[bytes] = []
    offset = header_len
    for index in range(frame_count):
        if offset + 2 > len(data):
            raise Rop1Error(f"ROP1 packet {index} is missing its length")
        packet_len = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        if packet_len <= 0 or packet_len > AOP1_MAX_PACKET_BYTES or offset + packet_len > len(data):
            raise Rop1Error(f"ROP1 packet {index} is invalid or truncated")
        packets.append(data[offset : offset + packet_len])
        offset += packet_len
    if offset != len(data):
        raise Rop1Error("ROP1 file contains trailing bytes")
    maximum_samples = frame_count * frame_samples
    if pcm_samples > maximum_samples or maximum_samples - pcm_samples > frame_samples:
        raise Rop1Error("ROP1 pcm_samples does not match its frame count")
    return Rop1Audio(frame_count, pcm_samples, pre_skip, end_trim, tuple(packets))


def parse_rop1_file(path: Path) -> Rop1Audio:
    try:
        return parse_rop1_bytes(path.read_bytes())
    except OSError as exc:
        raise Rop1Error(f"failed to read ROP1 file: {exc}") from exc


def _parse_ogg_packets(data: bytes) -> tuple[int, int, list[bytes]]:
    offset = 0
    packets: list[bytes] = []
    partial = bytearray()
    final_granule = 0
    while offset < len(data):
        if offset + 27 > len(data) or data[offset : offset + 4] != b"OggS":
            raise Rop1Error("invalid or truncated Ogg page")
        page_segments = data[offset + 26]
        header_end = offset + 27 + page_segments
        if header_end > len(data):
            raise Rop1Error("truncated Ogg segment table")
        lacing = data[offset + 27 : header_end]
        body_len = sum(lacing)
        body_end = header_end + body_len
        if body_end > len(data):
            raise Rop1Error("truncated Ogg page body")
        granule = struct.unpack_from("<Q", data, offset + 6)[0]
        if granule != 0xFFFFFFFFFFFFFFFF:
            final_granule = granule
        cursor = header_end
        for segment_len in lacing:
            partial += data[cursor : cursor + segment_len]
            cursor += segment_len
            if segment_len < 255:
                packets.append(bytes(partial))
                partial.clear()
        offset = body_end
    if partial or len(packets) < 3 or not packets[0].startswith(b"OpusHead"):
        raise Rop1Error("Ogg file does not contain complete Opus packets")
    if len(packets[0]) < 19:
        raise Rop1Error("OpusHead is truncated")
    pre_skip_48k = struct.unpack_from("<H", packets[0], 10)[0]
    return pre_skip_48k, final_granule, packets[2:]


def _device_wav_samples(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2 or wav_file.getframerate() != 16000:
                raise Rop1Error("TTS WAV must be 16 kHz mono 16-bit PCM")
            return wav_file.getnframes()
    except (OSError, wave.Error) as exc:
        raise Rop1Error(f"invalid TTS WAV: {exc}") from exc


def encode_wav_to_rop1(source: Path, destination: Path) -> Rop1File:
    source_samples = _device_wav_samples(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".ogg", dir=destination.parent, delete=False) as temporary:
        ogg_path = Path(temporary.name)
    command = [
        os.getenv("FFMPEG_BIN", "ffmpeg"), "-threads", "1", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-filter_threads", "1", "-threads", "1", "-ac", "1", "-ar", "16000",
        "-c:a", "libopus", "-application", "voip", "-b:a", str(ROP1_TARGET_BITRATE),
        "-vbr", "on", "-frame_duration", "20", "-f", "ogg", str(ogg_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        if result.returncode != 0:
            raise Rop1Error(result.stderr.strip() or f"FFmpeg exited with {result.returncode}")
        pre_skip_48k, final_granule, packets = _parse_ogg_packets(ogg_path.read_bytes())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Rop1Error(f"failed to encode Opus reply: {exc}") from exc
    finally:
        ogg_path.unlink(missing_ok=True)

    if not packets or final_granule <= pre_skip_48k:
        raise Rop1Error("encoded Ogg has invalid timing metadata")
    pcm_samples = round((final_granule - pre_skip_48k) / 3)
    if abs(pcm_samples - source_samples) > ROP1_FRAME_SAMPLES:
        raise Rop1Error("encoded Opus duration differs from source WAV by more than 20 ms")
    pcm_samples = source_samples
    pre_skip = round(pre_skip_48k / 3)
    end_trim_48k = len(packets) * 960 + pre_skip_48k - final_granule
    end_trim = max(0, round(end_trim_48k / 3))
    audio = Rop1Audio(len(packets), pcm_samples, pre_skip, end_trim, tuple(packets))
    data = build_rop1_bytes(audio)
    temporary_path = destination.with_name(f".{destination.name}.part")
    temporary_path.write_bytes(data)
    temporary_path.replace(destination)
    return Rop1File(destination, len(data), hashlib.sha256(data).hexdigest(), audio)


def mux_rop1_to_ogg(source: Path, destination: Path) -> Rop1Audio:
    audio = parse_rop1_file(source)
    serial = int.from_bytes(os.urandom(4), "little")
    pre_skip_48k = audio.pre_skip * 3
    vendor = b"wkt-ai-server"
    opus_head = struct.pack("<8sBBHIhB", b"OpusHead", 1, 1, pre_skip_48k, 16000, 0, 0)
    opus_tags = b"OpusTags" + struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", 0)
    pages = [
        _build_ogg_page([opus_head], serial=serial, sequence=0, granule_position=0, header_type=0x02),
        _build_ogg_page([opus_tags], serial=serial, sequence=1, granule_position=0, header_type=0),
    ]
    sequence = 2
    index = 0
    while index < audio.frame_count:
        page_packets: list[bytes] = []
        segments = 0
        while index < audio.frame_count and len(page_packets) < 50:
            packet = audio.packets[index]
            packet_segments = len(_packet_lacing(packet))
            if page_packets and segments + packet_segments > 255:
                break
            page_packets.append(packet)
            segments += packet_segments
            index += 1
        last = index == audio.frame_count
        granule = pre_skip_48k + (audio.pcm_samples * 3 if last else index * 960)
        pages.append(_build_ogg_page(page_packets, serial=serial, sequence=sequence, granule_position=granule, header_type=0x04 if last else 0))
        sequence += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"".join(pages))
    return audio
