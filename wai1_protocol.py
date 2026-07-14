from __future__ import annotations

from dataclasses import dataclass
import struct
import zlib


WAI1_MAGIC = b"WAI1"
WAI1_VERSION = 1
WAI1_HEADER_LEN = 32
WAI1_MAX_PAYLOAD = 4096

PACKET_VOICE_UPLOAD = 1
PACKET_CAMERA_UPLOAD = 2
PACKET_REPLY_OPUS = 3
VALID_PACKET_TYPES = {
    PACKET_VOICE_UPLOAD,
    PACKET_CAMERA_UPLOAD,
    PACKET_REPLY_OPUS,
}

_HEADER = struct.Struct("<4sBBHIIIIHHI")


class Wai1Error(ValueError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass(frozen=True)
class Wai1Packet:
    packet_type: int
    stream_id: int
    sequence: int
    offset: int
    total: int
    payload: bytes
    flags: int = 0

    @property
    def next_offset(self) -> int:
        return self.offset + len(self.payload)


def encode_wai1_packet(packet: Wai1Packet) -> bytes:
    payload = bytes(packet.payload)
    if packet.packet_type not in VALID_PACKET_TYPES:
        raise Wai1Error("invalid_message", "unsupported WAI1 packet type")
    if len(payload) > WAI1_MAX_PAYLOAD:
        raise Wai1Error("payload_too_large", "WAI1 payload exceeds 4096 bytes")
    if min(packet.stream_id, packet.sequence, packet.offset, packet.total) < 0:
        raise Wai1Error("invalid_message", "WAI1 integer fields cannot be negative")
    if packet.offset + len(payload) > packet.total:
        raise Wai1Error("total_mismatch", "WAI1 payload exceeds declared total")
    header = _HEADER.pack(
        WAI1_MAGIC,
        WAI1_VERSION,
        packet.packet_type,
        packet.flags,
        packet.stream_id,
        packet.sequence,
        packet.offset,
        packet.total,
        len(payload),
        0,
        zlib.crc32(payload) & 0xFFFFFFFF,
    )
    return header + payload


def decode_wai1_packet(message: bytes) -> Wai1Packet:
    if len(message) < WAI1_HEADER_LEN:
        raise Wai1Error("invalid_message", "WAI1 message is shorter than its header")
    (
        magic,
        version,
        packet_type,
        flags,
        stream_id,
        sequence,
        offset,
        total,
        payload_len,
        reserved,
        expected_crc,
    ) = _HEADER.unpack_from(message)
    if magic != WAI1_MAGIC:
        raise Wai1Error("invalid_message", "invalid WAI1 magic")
    if version != WAI1_VERSION:
        raise Wai1Error("unsupported_protocol", f"unsupported WAI1 version: {version}")
    if packet_type not in VALID_PACKET_TYPES:
        raise Wai1Error("invalid_message", "unsupported WAI1 packet type")
    if reserved != 0:
        raise Wai1Error("invalid_message", "WAI1 reserved field must be zero")
    if payload_len > WAI1_MAX_PAYLOAD:
        raise Wai1Error("payload_too_large", "WAI1 payload exceeds 4096 bytes")
    if len(message) != WAI1_HEADER_LEN + payload_len:
        raise Wai1Error("invalid_message", "WAI1 message length does not match payload_len")
    if offset + payload_len > total:
        raise Wai1Error("total_mismatch", "WAI1 payload exceeds declared total")
    payload = message[WAI1_HEADER_LEN:]
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise Wai1Error("crc_mismatch", "WAI1 payload CRC32 does not match")
    return Wai1Packet(
        packet_type=packet_type,
        flags=flags,
        stream_id=stream_id,
        sequence=sequence,
        offset=offset,
        total=total,
        payload=payload,
    )
