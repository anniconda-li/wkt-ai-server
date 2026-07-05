from dataclasses import dataclass, field
from datetime import datetime, timezone


MAX_MEMORY_ROUNDS = 10
DEFAULT_DEVICE_ID = "default"


@dataclass
class DeviceSession:
    device_id: str
    memory: list[dict[str, str]] = field(default_factory=list)
    latest_image_id: str | None = None
    latest_artifact_id: str | None = None
    latest_vision_description: str | None = None
    last_answer: str | None = None
    upload_generation: int = 0
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


sessions: dict[str, DeviceSession] = {}


def normalize_device_id(device_id: str | None) -> str:
    cleaned = (device_id or DEFAULT_DEVICE_ID).strip()
    return cleaned or DEFAULT_DEVICE_ID


def get_session(device_id: str | None) -> DeviceSession:
    normalized = normalize_device_id(device_id)
    if normalized not in sessions:
        sessions[normalized] = DeviceSession(device_id=normalized)
    return sessions[normalized]


def touch_session(session: DeviceSession) -> None:
    session.updated_at = datetime.now(timezone.utc).isoformat()


def remember_turn(session: DeviceSession, user_message: str, assistant_message: str) -> None:
    session.memory.append({"role": "user", "content": user_message})
    session.memory.append({"role": "assistant", "content": assistant_message})
    del session.memory[:-MAX_MEMORY_ROUNDS * 2]
    session.last_answer = assistant_message
    touch_session(session)


def clear_session(device_id: str | None) -> None:
    session = get_session(device_id)
    session.memory.clear()
    session.last_answer = None
    touch_session(session)


def session_snapshot(device_id: str | None) -> dict[str, object]:
    session = get_session(device_id)
    return {
        "device_id": session.device_id,
        "memory": list(session.memory),
        "latest_image_id": session.latest_image_id,
        "latest_artifact_id": session.latest_artifact_id,
        "latest_vision_description": session.latest_vision_description,
        "last_answer": session.last_answer,
        "upload_generation": session.upload_generation,
        "updated_at": session.updated_at,
    }


def list_session_summaries() -> list[dict[str, object]]:
    return [
        {
            "device_id": session.device_id,
            "memory_messages": len(session.memory),
            "latest_image_id": session.latest_image_id,
            "latest_artifact_id": session.latest_artifact_id,
            "upload_generation": session.upload_generation,
            "updated_at": session.updated_at,
        }
        for session in sessions.values()
    ]
