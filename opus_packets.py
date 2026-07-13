import os
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path


AOP1_MAGIC = b"AOP1"
AOP1_VERSION = 1
AOP1_HEADER_LEN = 24
AOP1_CHANNELS = 1
AOP1_SAMPLE_RATE = 16_000
AOP1_FRAME_SAMPLES = 320
AOP1_FRAME_DURATION_MS = 20
AOP1_MAX_PACKET_BYTES = 1_275
AOP1_MAX_FRAMES = 3_000
AOP1_MAX_PCM_SAMPLES = 960_000
AOP1_MAX_FILE_BYTES = 256 * 1024


class OpusPacketsError(ValueError):
    pass


@dataclass(frozen=True)
class OpusPackets:
    frame_count: int
    pcm_samples: int
    packets: tuple[bytes, ...]

    @property
    def duration_seconds(self) -> float:
        return self.pcm_samples / AOP1_SAMPLE_RATE


def parse_aop1_bytes(data: bytes) -> OpusPackets:
    if len(data) < AOP1_HEADER_LEN:
        raise OpusPacketsError("AOP1 file is shorter than its header")
    if len(data) > AOP1_MAX_FILE_BYTES:
        raise OpusPacketsError("AOP1 file exceeds 256 KiB")

    magic, version, channels, header_len, sample_rate, frame_samples, frame_ms, frame_count, pcm_samples = (
        struct.unpack_from("<4sBBHIHHII", data, 0)
    )
    if magic != AOP1_MAGIC:
        raise OpusPacketsError("invalid AOP1 magic")
    if version != AOP1_VERSION:
        raise OpusPacketsError(f"unsupported AOP1 version: {version}")
    if channels != AOP1_CHANNELS:
        raise OpusPacketsError(f"unsupported AOP1 channel count: {channels}")
    if header_len != AOP1_HEADER_LEN:
        raise OpusPacketsError(f"invalid AOP1 header length: {header_len}")
    if sample_rate != AOP1_SAMPLE_RATE:
        raise OpusPacketsError(f"unsupported AOP1 sample rate: {sample_rate}")
    if frame_samples != AOP1_FRAME_SAMPLES or frame_ms != AOP1_FRAME_DURATION_MS:
        raise OpusPacketsError(
            f"unsupported AOP1 frame: {frame_samples} samples/{frame_ms} ms"
        )
    if frame_count <= 0 or frame_count > AOP1_MAX_FRAMES:
        raise OpusPacketsError(f"invalid AOP1 frame count: {frame_count}")
    if pcm_samples <= 0 or pcm_samples > AOP1_MAX_PCM_SAMPLES:
        raise OpusPacketsError(f"invalid AOP1 PCM sample count: {pcm_samples}")

    minimum_samples = (frame_count - 1) * AOP1_FRAME_SAMPLES + 1
    maximum_samples = frame_count * AOP1_FRAME_SAMPLES
    if not minimum_samples <= pcm_samples <= maximum_samples:
        raise OpusPacketsError(
            f"AOP1 PCM sample count {pcm_samples} does not match {frame_count} frames"
        )

    packets: list[bytes] = []
    offset = header_len
    for frame_index in range(frame_count):
        if offset + 2 > len(data):
            raise OpusPacketsError(f"AOP1 frame {frame_index} is missing its length")
        packet_len = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        if packet_len <= 0 or packet_len > AOP1_MAX_PACKET_BYTES:
            raise OpusPacketsError(
                f"AOP1 frame {frame_index} has invalid packet length: {packet_len}"
            )
        if offset + packet_len > len(data):
            raise OpusPacketsError(f"AOP1 frame {frame_index} is truncated")
        packets.append(data[offset : offset + packet_len])
        offset += packet_len

    if offset != len(data):
        raise OpusPacketsError(f"AOP1 file has {len(data) - offset} trailing bytes")
    return OpusPackets(
        frame_count=frame_count,
        pcm_samples=pcm_samples,
        packets=tuple(packets),
    )


def parse_aop1_file(path: Path) -> OpusPackets:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise OpusPacketsError(f"failed to read AOP1 file: {exc}") from exc
    return parse_aop1_bytes(data)


def _ogg_crc(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if crc & 0x80000000 else (crc << 1) & 0xFFFFFFFF
    return crc


def _packet_lacing(packet: bytes) -> bytes:
    full_segments, remainder = divmod(len(packet), 255)
    values = [255] * full_segments
    values.append(remainder)
    return bytes(values)


def _build_ogg_page(
    packets: list[bytes],
    *,
    serial: int,
    sequence: int,
    granule_position: int,
    header_type: int,
) -> bytes:
    lacing = b"".join(_packet_lacing(packet) for packet in packets)
    if len(lacing) > 255:
        raise OpusPacketsError("too many Ogg lacing segments in one page")
    body = b"".join(packets)
    header = bytearray(
        struct.pack(
            "<4sBBQIIIB",
            b"OggS",
            0,
            header_type,
            granule_position,
            serial,
            sequence,
            0,
            len(lacing),
        )
    )
    page = header + lacing + body
    struct.pack_into("<I", page, 22, _ogg_crc(page))
    return bytes(page)


def mux_aop1_to_ogg(source: Path, destination: Path) -> OpusPackets:
    opus = parse_aop1_file(source)
    serial = int.from_bytes(os.urandom(4), "little")
    vendor = b"wkt-ai-server"
    opus_head = struct.pack(
        "<8sBBHIhB",
        b"OpusHead",
        1,
        AOP1_CHANNELS,
        0,
        AOP1_SAMPLE_RATE,
        0,
        0,
    )
    opus_tags = b"OpusTags" + struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", 0)

    pages = [
        _build_ogg_page(
            [opus_head], serial=serial, sequence=0, granule_position=0, header_type=0x02
        ),
        _build_ogg_page(
            [opus_tags], serial=serial, sequence=1, granule_position=0, header_type=0
        ),
    ]

    sequence = 2
    packet_index = 0
    while packet_index < opus.frame_count:
        page_packets: list[bytes] = []
        segment_count = 0
        while packet_index < opus.frame_count and len(page_packets) < 50:
            packet = opus.packets[packet_index]
            packet_segments = len(_packet_lacing(packet))
            if page_packets and segment_count + packet_segments > 255:
                break
            page_packets.append(packet)
            segment_count += packet_segments
            packet_index += 1

        is_last = packet_index == opus.frame_count
        granule_samples = opus.pcm_samples if is_last else packet_index * AOP1_FRAME_SAMPLES
        pages.append(
            _build_ogg_page(
                page_packets,
                serial=serial,
                sequence=sequence,
                granule_position=granule_samples * 3,
                header_type=0x04 if is_last else 0,
            )
        )
        sequence += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"".join(pages))
    return opus


def decode_ogg_to_device_wav(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        os.getenv("FFMPEG_BIN", "ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OpusPacketsError(f"failed to run FFmpeg: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise OpusPacketsError(f"FFmpeg failed to decode Ogg/Opus: {detail}")
