from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any
from uuid import uuid4

from opus_packets import AOP1_MAX_FILE_BYTES
from vision import MAX_IMAGE_BYTES, safe_path_part
from wai1_protocol import WAI1_MAX_PAYLOAD


DEFAULT_TTL_SECONDS = 20 * 60
DEFAULT_MAX_SESSIONS = 200
DEFAULT_MAX_SESSIONS_PER_DEVICE = 4
DEFAULT_MAX_TEMP_BYTES = 64 * 1024 * 1024
DEFAULT_CLEANUP_INTERVAL_SECONDS = 60
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class AiWsStoreError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        next_offset: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.next_offset = next_offset
        self.retryable = retryable


@dataclass(frozen=True)
class AiWsSession:
    session_id: str
    device_id: str
    request_id: str
    kind: str
    language: str
    content_type: str
    total: int
    sha256: str
    received: int
    state: str
    status: str
    temp_path: Path | None
    result: dict[str, Any] | None
    error: str | None
    reply_path: Path | None
    reply_total: int
    reply_sha256: str | None
    reply_duration_ms: int

    @property
    def stream_id(self) -> int:
        return 1 if self.kind == "voice" else 2


def _positive_number(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)).strip())
    except ValueError:
        return default
    return value if value > 0 else default


def ai_ws_db_path() -> Path:
    configured = os.getenv("AI_WS_DB_PATH", "").strip()
    return Path(configured) if configured else Path(__file__).parent / "uploads" / "ai_ws.sqlite3"


def ai_ws_temp_dir() -> Path:
    configured = os.getenv("AI_WS_TEMP_DIR", "").strip()
    return Path(configured) if configured else Path(__file__).parent / "uploads" / "ai_ws"


def ai_ws_session_ttl_seconds() -> float:
    return max(600.0, _positive_number("AI_WS_SESSION_TTL_SECONDS", DEFAULT_TTL_SECONDS))


def ai_ws_cleanup_interval_seconds() -> float:
    return _positive_number("AI_WS_CLEANUP_INTERVAL_SECONDS", DEFAULT_CLEANUP_INTERVAL_SECONDS)


class AiWsSessionStore:
    def __init__(
        self,
        db_path: Path,
        temp_dir: Path,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_sessions_per_device: int = DEFAULT_MAX_SESSIONS_PER_DEVICE,
        max_temp_bytes: int = DEFAULT_MAX_TEMP_BYTES,
        clock: Any = time.time,
    ) -> None:
        self.db_path = db_path
        self.temp_dir = temp_dir
        self.ttl_seconds = max(600.0, ttl_seconds)
        self.max_sessions = max(1, max_sessions)
        self.max_sessions_per_device = max(1, max_sessions_per_device)
        self.max_temp_bytes = max(WAI1_MAX_PAYLOAD, max_temp_bytes)
        self.clock = clock
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_ws_sessions (
                    session_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK (kind IN ('voice', 'camera')),
                    language TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    total INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    received INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    status TEXT NOT NULL,
                    temp_path TEXT,
                    result_json TEXT,
                    error TEXT,
                    reply_path TEXT,
                    reply_total INTEGER NOT NULL DEFAULT 0,
                    reply_sha256 TEXT,
                    reply_duration_ms INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    UNIQUE(device_id, request_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_ws_parts (
                    session_id TEXT NOT NULL,
                    offset INTEGER NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    chunk_sha256 TEXT NOT NULL,
                    PRIMARY KEY (session_id, offset),
                    FOREIGN KEY (session_id) REFERENCES ai_ws_sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_ai_ws_expiry ON ai_ws_sessions(expires_at)")

    @staticmethod
    def _from_row(row: sqlite3.Row) -> AiWsSession:
        result = json.loads(row["result_json"]) if row["result_json"] else None
        return AiWsSession(
            session_id=str(row["session_id"]), device_id=str(row["device_id"]),
            request_id=str(row["request_id"]), kind=str(row["kind"]),
            language=str(row["language"]), content_type=str(row["content_type"]),
            total=int(row["total"]), sha256=str(row["sha256"]), received=int(row["received"]),
            state=str(row["state"]), status=str(row["status"]),
            temp_path=Path(str(row["temp_path"])) if row["temp_path"] else None,
            result=result, error=row["error"],
            reply_path=Path(str(row["reply_path"])) if row["reply_path"] else None,
            reply_total=int(row["reply_total"]), reply_sha256=row["reply_sha256"],
            reply_duration_ms=int(row["reply_duration_ms"]),
        )

    @staticmethod
    def _validate_start(kind: str, request_id: str, total: int, sha256: str) -> None:
        if kind not in {"voice", "camera"}:
            raise AiWsStoreError("invalid_message", "invalid upload kind")
        if not REQUEST_ID_RE.fullmatch(request_id):
            raise AiWsStoreError("invalid_message", "request_id has an invalid format")
        if not SHA256_RE.fullmatch(sha256):
            raise AiWsStoreError("invalid_message", "sha256 must be 64 lowercase hexadecimal characters")
        maximum = AOP1_MAX_FILE_BYTES if kind == "voice" else MAX_IMAGE_BYTES
        if total <= 0 or total > maximum:
            raise AiWsStoreError("total_mismatch", f"{kind} total exceeds its size limit")

    def _new_temp_path(self, device: str, session_id: str, kind: str) -> Path:
        suffix = "aop1" if kind == "voice" else "jpg"
        directory = self.temp_dir / safe_path_part(device) / session_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"request.{suffix}.part"

    def start(
        self, *, device_id: str, request_id: str, kind: str, total: int,
        sha256: str, language: str = "zh", content_type: str = "",
    ) -> AiWsSession:
        self._validate_start(kind, request_id, total, sha256)
        now = float(self.clock())
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM ai_ws_sessions WHERE device_id=? AND request_id=?",
                (device_id, request_id),
            ).fetchone()
            if row is not None:
                if (
                    row["kind"] != kind
                    or int(row["total"]) != total
                    or row["sha256"] != sha256
                    or row["language"] != language
                    or row["content_type"] != content_type
                ):
                    connection.rollback()
                    raise AiWsStoreError("total_mismatch", "request_id conflicts with an existing session", next_offset=int(row["received"]))
                connection.execute(
                    "UPDATE ai_ws_sessions SET updated_at=?, expires_at=? WHERE session_id=?",
                    (now, now + self.ttl_seconds, row["session_id"]),
                )
                connection.commit()
                return self.get(str(row["session_id"]), device_id=device_id)  # type: ignore[return-value]

            counts = connection.execute(
                "SELECT COUNT(*) AS total, SUM(CASE WHEN device_id=? AND state IN ('uploading','processing') THEN 1 ELSE 0 END) AS per_device, COALESCE(SUM(CASE WHEN temp_path IS NOT NULL THEN total ELSE 0 END) + SUM(reply_total),0) AS reserved FROM ai_ws_sessions",
                (device_id,),
            ).fetchone()
            if int(counts["total"]) >= self.max_sessions or int(counts["per_device"] or 0) >= self.max_sessions_per_device:
                connection.rollback()
                raise AiWsStoreError("server_busy", "AI WebSocket session limit reached", retryable=True)
            if int(counts["reserved"]) + total > self.max_temp_bytes:
                connection.rollback()
                raise AiWsStoreError("server_busy", "AI WebSocket temporary space limit reached", retryable=True)
            session_id = uuid4().hex
            path = self._new_temp_path(device_id, session_id, kind)
            path.touch(exist_ok=False)
            try:
                connection.execute(
                    """INSERT INTO ai_ws_sessions (
                        session_id,device_id,request_id,kind,language,content_type,total,sha256,
                        received,state,status,temp_path,created_at,updated_at,expires_at
                    ) VALUES (?,?,?,?,?,?,?,?,0,'uploading','uploading',?,?,?,?)""",
                    (session_id, device_id, request_id, kind, language, content_type, total, sha256,
                     str(path), now, now, now + self.ttl_seconds),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                path.unlink(missing_ok=True)
                raise
        return self.get(session_id, device_id=device_id)  # type: ignore[return-value]

    def get(self, session_id: str, *, device_id: str | None = None) -> AiWsSession | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM ai_ws_sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            return None
        if device_id is not None and row["device_id"] != device_id:
            return None
        return self._from_row(row)

    def append(self, session_id: str, *, device_id: str, kind: str, offset: int, total: int, payload: bytes) -> tuple[AiWsSession, str]:
        if not payload:
            raise AiWsStoreError("invalid_message", "empty binary payload")
        if len(payload) > WAI1_MAX_PAYLOAD:
            raise AiWsStoreError("payload_too_large", "binary payload exceeds 4096 bytes")
        chunk_sha = hashlib.sha256(payload).hexdigest()
        now = float(self.clock())
        appended_path: Path | None = None
        prior = 0
        with closing(self._connect()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute("SELECT * FROM ai_ws_sessions WHERE session_id=?", (session_id,)).fetchone()
                if row is None or row["device_id"] != device_id:
                    raise AiWsStoreError("session_not_found", "session does not exist")
                received = int(row["received"])
                if row["kind"] != kind:
                    raise AiWsStoreError("invalid_stream", "packet type does not match the session")
                if int(row["total"]) != total:
                    raise AiWsStoreError("total_mismatch", "binary total does not match voice_start/camera_start", next_offset=received)
                if row["state"] == "cancelled":
                    raise AiWsStoreError("session_cancelled", "session is cancelled")
                if offset < received:
                    part = connection.execute(
                        "SELECT * FROM ai_ws_parts WHERE session_id=? AND offset=?", (session_id, offset)
                    ).fetchone()
                    if part is None or int(part["chunk_size"]) != len(payload) or part["chunk_sha256"] != chunk_sha:
                        connection.execute(
                            "UPDATE ai_ws_sessions SET state='failed',status='failed',error=?,updated_at=?,expires_at=? WHERE session_id=?",
                            ("repeated chunk conflicts with stored bytes", now, now + self.ttl_seconds, session_id),
                        )
                        connection.commit()
                        raise AiWsStoreError("offset_mismatch", "repeated chunk conflicts with stored bytes", next_offset=received)
                    connection.commit()
                    return self.get(session_id, device_id=device_id), "duplicate"  # type: ignore[return-value]
                if offset > received:
                    raise AiWsStoreError("offset_mismatch", "chunk is ahead of server next_offset", next_offset=received, retryable=True)
                if row["state"] != "uploading":
                    raise AiWsStoreError("offset_mismatch", "upload has already finished", next_offset=received)
                if offset + len(payload) > total:
                    raise AiWsStoreError("total_mismatch", "chunk exceeds declared total", next_offset=received)
                path = Path(str(row["temp_path"]))
                prior = received
                if not path.exists() or path.stat().st_size < received:
                    raise AiWsStoreError("internal_error", "temporary upload file is missing")
                if path.stat().st_size > received:
                    with path.open("r+b") as handle:
                        handle.truncate(received)
                appended_path = path
                with path.open("ab") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                next_offset = received + len(payload)
                connection.execute("INSERT INTO ai_ws_parts VALUES (?,?,?,?)", (session_id, offset, len(payload), chunk_sha))
                connection.execute(
                    "UPDATE ai_ws_sessions SET received=?,updated_at=?,expires_at=? WHERE session_id=?",
                    (next_offset, now, now + self.ttl_seconds, session_id),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                if appended_path is not None:
                    try:
                        with appended_path.open("r+b") as handle:
                            handle.truncate(prior)
                    except OSError:
                        pass
                raise
        return self.get(session_id, device_id=device_id), "accepted"  # type: ignore[return-value]

    def prepare_finish(self, session_id: str, *, device_id: str, sha256: str) -> tuple[AiWsSession, str]:
        now = float(self.clock())
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM ai_ws_sessions WHERE session_id=?", (session_id,)).fetchone()
            if row is None or row["device_id"] != device_id:
                connection.rollback(); raise AiWsStoreError("session_not_found", "session does not exist")
            if sha256 != row["sha256"]:
                connection.rollback(); raise AiWsStoreError("sha256_mismatch", "finish SHA-256 differs from start")
            if row["state"] == "cancelled":
                connection.rollback(); raise AiWsStoreError("session_cancelled", "session is cancelled")
            if int(row["received"]) != int(row["total"]):
                connection.rollback(); raise AiWsStoreError("offset_mismatch", "upload is incomplete", next_offset=int(row["received"]), retryable=True)
            if row["state"] in {"processing", "completed", "failed"}:
                connection.commit(); return self._from_row(row), "existing"
            path = Path(str(row["temp_path"]))
            try:
                data = path.read_bytes()
            except OSError as exc:
                connection.rollback(); raise AiWsStoreError("internal_error", f"failed to read upload: {exc}") from exc
            if len(data) != int(row["total"]):
                connection.rollback(); raise AiWsStoreError("total_mismatch", "temporary file length differs from total")
            if hashlib.sha256(data).hexdigest() != row["sha256"]:
                connection.rollback(); raise AiWsStoreError("sha256_mismatch", "complete upload SHA-256 does not match")
            connection.execute(
                "UPDATE ai_ws_sessions SET state='processing',status='uploaded',updated_at=?,expires_at=? WHERE session_id=?",
                (now, now + self.ttl_seconds, session_id),
            )
            connection.commit()
        return self.get(session_id, device_id=device_id), "claimed"  # type: ignore[return-value]

    def update_status(self, session_id: str, status: str, *, result: dict[str, Any] | None = None, error: str | None = None, state: str | None = None) -> AiWsSession:
        now = float(self.clock())
        result_json = json.dumps(result, ensure_ascii=False, separators=(",", ":")) if result is not None else None
        with closing(self._connect()) as connection:
            connection.execute(
                """UPDATE ai_ws_sessions SET status=?,result_json=COALESCE(?,result_json),error=?,
                state=COALESCE(?,state),updated_at=?,expires_at=? WHERE session_id=?""",
                (status, result_json, error, state, now, now + self.ttl_seconds, session_id),
            )
        session = self.get(session_id)
        if session is None:
            raise AiWsStoreError("session_not_found", "session does not exist")
        return session

    def set_reply(self, session_id: str, *, path: Path, total: int, sha256: str, duration_ms: int) -> AiWsSession:
        now = float(self.clock())
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT COALESCE(SUM(CASE WHEN temp_path IS NOT NULL THEN total ELSE 0 END) + SUM(reply_total),0) AS stored FROM ai_ws_sessions"
            ).fetchone()
            if int(row["stored"]) + total > self.max_temp_bytes:
                connection.rollback()
                raise AiWsStoreError("server_busy", "AI WebSocket temporary space limit reached", retryable=True)
            connection.execute(
                """UPDATE ai_ws_sessions SET reply_path=?,reply_total=?,reply_sha256=?,reply_duration_ms=?,
                status='audio_ready',state='completed',updated_at=?,expires_at=? WHERE session_id=?""",
                (str(path), total, sha256, duration_ms, now, now + self.ttl_seconds, session_id),
            )
            connection.commit()
        session = self.get(session_id)
        if session is None:
            raise AiWsStoreError("session_not_found", "session does not exist")
        return session

    def cancel(self, session_id: str, *, device_id: str) -> AiWsSession:
        session = self.get(session_id, device_id=device_id)
        if session is None:
            raise AiWsStoreError("session_not_found", "session does not exist")
        if session.status != "cancelled":
            session = self.update_status(session_id, "cancelled", state="cancelled")
        return session

    def cleanup(self) -> int:
        now = float(self.clock())
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("SELECT * FROM ai_ws_sessions WHERE expires_at<=?", (now,)).fetchall()
            connection.executemany("DELETE FROM ai_ws_sessions WHERE session_id=?", [(row["session_id"],) for row in rows])
            referenced = {str(row["temp_path"]) for row in connection.execute("SELECT temp_path FROM ai_ws_sessions WHERE temp_path IS NOT NULL")}
            referenced.update(str(row["reply_path"]) for row in connection.execute("SELECT reply_path FROM ai_ws_sessions WHERE reply_path IS NOT NULL"))
            connection.commit()
        for row in rows:
            for column in ("temp_path", "reply_path"):
                if row[column]:
                    Path(str(row[column])).unlink(missing_ok=True)
        cutoff = now - self.ttl_seconds
        for path in self.temp_dir.rglob("*"):
            try:
                if path.is_file() and str(path) not in referenced and path.stat().st_mtime <= cutoff:
                    path.unlink()
            except OSError:
                pass
        return len(rows)


_stores: dict[tuple[str, str, float, int, int, int], AiWsSessionStore] = {}
_stores_lock = threading.Lock()


def get_ai_ws_store() -> AiWsSessionStore:
    db_path = ai_ws_db_path().resolve()
    temp_dir = ai_ws_temp_dir().resolve()
    ttl = ai_ws_session_ttl_seconds()
    max_sessions = int(_positive_number("AI_WS_MAX_SESSIONS", DEFAULT_MAX_SESSIONS))
    per_device = int(_positive_number("AI_WS_MAX_SESSIONS_PER_DEVICE", DEFAULT_MAX_SESSIONS_PER_DEVICE))
    temp_bytes = int(_positive_number("AI_WS_MAX_TEMP_BYTES", DEFAULT_MAX_TEMP_BYTES))
    key = (str(db_path), str(temp_dir), ttl, max_sessions, per_device, temp_bytes)
    with _stores_lock:
        store = _stores.get(key)
        if store is None:
            store = AiWsSessionStore(db_path, temp_dir, ttl_seconds=ttl, max_sessions=max_sessions, max_sessions_per_device=per_device, max_temp_bytes=temp_bytes)
            _stores[key] = store
        return store
