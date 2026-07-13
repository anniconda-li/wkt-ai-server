from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
from uuid import uuid4

from camera_idempotency import CameraStoredOutcome, camera_idempotency_db_path
from vision import MAX_IMAGE_BYTES, safe_path_part


DEFAULT_SESSION_TTL_SECONDS = 10 * 60
DEFAULT_COMPLETED_TTL_SECONDS = 20 * 60
DEFAULT_MAX_SESSIONS = 100
DEFAULT_MAX_TEMP_BYTES = 64 * 1024 * 1024
DEFAULT_CLEANUP_INTERVAL_SECONDS = 60
MAX_CHUNK_BYTES = 4096

logger = logging.getLogger("wkt_ai_server.main")


class CameraChunkError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        *,
        next_offset: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.next_offset = next_offset


@dataclass(frozen=True)
class CameraChunkIdentity:
    request_id: str
    device_id: str
    total: int
    image_sha256: str


@dataclass(frozen=True)
class CameraChunkSession:
    identity: CameraChunkIdentity
    state: str
    received: int
    temp_path: Path | None
    outcome: CameraStoredOutcome | None = None


@dataclass(frozen=True)
class CameraChunkAcceptResult:
    action: str
    session: CameraChunkSession
    offset: int
    chunk_size: int


def _positive_number(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default


def camera_chunk_session_ttl_seconds() -> float:
    return _positive_number("CAMERA_CHUNK_SESSION_TTL_SECONDS", DEFAULT_SESSION_TTL_SECONDS)


def camera_chunk_completed_ttl_seconds() -> float:
    return _positive_number(
        "CAMERA_CHUNK_COMPLETED_TTL_SECONDS",
        DEFAULT_COMPLETED_TTL_SECONDS,
    )


def camera_chunk_max_sessions() -> int:
    return max(1, int(_positive_number("CAMERA_CHUNK_MAX_SESSIONS", DEFAULT_MAX_SESSIONS)))


def camera_chunk_max_temp_bytes() -> int:
    return max(
        MAX_CHUNK_BYTES,
        int(_positive_number("CAMERA_CHUNK_MAX_TEMP_BYTES", DEFAULT_MAX_TEMP_BYTES)),
    )


def camera_chunk_cleanup_interval_seconds() -> float:
    return _positive_number(
        "CAMERA_CHUNK_CLEANUP_INTERVAL_SECONDS",
        DEFAULT_CLEANUP_INTERVAL_SECONDS,
    )


def camera_chunk_temp_dir() -> Path:
    configured = os.getenv("CAMERA_CHUNK_TEMP_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).parent / "uploads" / "camera_chunks"


class CameraChunkStore:
    def __init__(
        self,
        db_path: Path,
        temp_dir: Path,
        *,
        session_ttl_seconds: float,
        completed_ttl_seconds: float,
        max_sessions: int,
        max_temp_bytes: int,
        cleanup_interval_seconds: float,
        clock: Any = time.time,
    ) -> None:
        self.db_path = db_path
        self.temp_dir = temp_dir
        self.session_ttl_seconds = session_ttl_seconds
        self.completed_ttl_seconds = completed_ttl_seconds
        self.processing_lease_seconds = max(300.0, completed_ttl_seconds)
        self.max_sessions = max_sessions
        self.max_temp_bytes = max_temp_bytes
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.clock = clock
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_lock = threading.Lock()
        self._last_cleanup = 0.0
        self._initialize()
        self.cleanup()

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
                CREATE TABLE IF NOT EXISTS camera_chunk_uploads (
                    request_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    total INTEGER NOT NULL,
                    image_sha256 TEXT NOT NULL,
                    received INTEGER NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('uploading', 'processing', 'completed')),
                    temp_path TEXT,
                    http_status INTEGER,
                    response_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS camera_chunk_parts (
                    request_id TEXT NOT NULL,
                    offset INTEGER NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    chunk_sha256 TEXT NOT NULL,
                    PRIMARY KEY (request_id, offset),
                    FOREIGN KEY (request_id) REFERENCES camera_chunk_uploads(request_id)
                        ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_camera_chunk_uploads_expires "
                "ON camera_chunk_uploads(expires_at)"
            )

    @staticmethod
    def _identity_from_row(row: sqlite3.Row) -> CameraChunkIdentity:
        return CameraChunkIdentity(
            request_id=str(row["request_id"]),
            device_id=str(row["device_id"]),
            total=int(row["total"]),
            image_sha256=str(row["image_sha256"]),
        )

    @staticmethod
    def _outcome_from_row(row: sqlite3.Row) -> CameraStoredOutcome | None:
        if row["state"] != "completed":
            return None
        if row["http_status"] is None or row["response_json"] is None:
            raise CameraChunkError(500, "stored_result_invalid", "completed upload has no result")
        payload = json.loads(str(row["response_json"]))
        if not isinstance(payload, dict):
            raise CameraChunkError(500, "stored_result_invalid", "stored result is not an object")
        return CameraStoredOutcome(int(row["http_status"]), payload)

    @classmethod
    def _session_from_row(cls, row: sqlite3.Row) -> CameraChunkSession:
        raw_path = row["temp_path"]
        return CameraChunkSession(
            identity=cls._identity_from_row(row),
            state=str(row["state"]),
            received=int(row["received"]),
            temp_path=Path(str(raw_path)) if raw_path else None,
            outcome=cls._outcome_from_row(row),
        )

    @staticmethod
    def _identity_matches(row: sqlite3.Row, identity: CameraChunkIdentity) -> bool:
        return (
            row["device_id"] == identity.device_id
            and int(row["total"]) == identity.total
            and row["image_sha256"] == identity.image_sha256
        )

    def _new_temp_path(self, identity: CameraChunkIdentity) -> Path:
        device_dir = self.temp_dir / safe_path_part(identity.device_id)
        device_dir.mkdir(parents=True, exist_ok=True)
        return device_dir / f"{uuid4().hex}.jpeg.part"

    @staticmethod
    def _unlink(path: Path | None) -> None:
        if path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("camera.chunk.cleanup_file_failed path=%s error=%r", path, exc)

    def _expired_rows_locked(self, connection: sqlite3.Connection, now: float) -> list[sqlite3.Row]:
        rows = connection.execute(
            """
            SELECT * FROM camera_chunk_uploads
            WHERE expires_at <= ?
            """,
            (now,),
        ).fetchall()
        if rows:
            connection.executemany(
                "DELETE FROM camera_chunk_uploads WHERE request_id = ?",
                [(row["request_id"],) for row in rows],
            )
        return rows

    def _log_expired(self, row: sqlite3.Row) -> None:
        logger.info(
            "camera.upload.expired device=%s request_id=%s offset=-1 chunk_size=0 "
            "received=%d total=%d next_offset=%d stage_ms=0.0 result=expired",
            row["device_id"],
            row["request_id"],
            row["received"],
            row["total"],
            row["received"],
        )

    def maybe_cleanup(self) -> int:
        now = float(self.clock())
        with self._cleanup_lock:
            if now - self._last_cleanup < self.cleanup_interval_seconds:
                return 0
            self._last_cleanup = now
        return self.cleanup()

    def cleanup(self) -> int:
        now = float(self.clock())
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = self._expired_rows_locked(connection, now)
            referenced = {
                str(row["temp_path"])
                for row in connection.execute(
                    "SELECT temp_path FROM camera_chunk_uploads WHERE temp_path IS NOT NULL"
                ).fetchall()
            }
            connection.commit()

        for row in rows:
            self._unlink(Path(str(row["temp_path"])) if row["temp_path"] else None)
            self._log_expired(row)

        orphan_cutoff = now - self.session_ttl_seconds
        for path in self.temp_dir.rglob("*.part"):
            try:
                if str(path) not in referenced and path.stat().st_mtime <= orphan_cutoff:
                    path.unlink()
            except OSError:
                continue
        return len(rows)

    def _ensure_capacity_locked(self, connection: sqlite3.Connection, identity: CameraChunkIdentity) -> None:
        row = connection.execute(
            """
            SELECT COUNT(*) AS sessions, COALESCE(SUM(total), 0) AS reserved
            FROM camera_chunk_uploads
            WHERE state IN ('uploading', 'processing')
            """
        ).fetchone()
        if int(row["sessions"]) >= self.max_sessions:
            raise CameraChunkError(503, "session_capacity_full", "camera upload session limit reached")
        if int(row["reserved"]) + identity.total > self.max_temp_bytes:
            raise CameraChunkError(413, "temporary_space_limit", "camera temporary space limit reached")

    @staticmethod
    def _reconcile_file(path: Path, received: int) -> None:
        if not path.exists():
            if received != 0:
                raise CameraChunkError(500, "temporary_file_missing", "camera temporary file is missing")
            return
        actual_size = path.stat().st_size
        if actual_size < received:
            raise CameraChunkError(500, "temporary_file_incomplete", "camera temporary file is shorter than database state")
        if actual_size > received:
            with path.open("r+b") as handle:
                handle.truncate(received)
                handle.flush()
                os.fsync(handle.fileno())

    def accept_chunk(
        self,
        identity: CameraChunkIdentity,
        *,
        offset: int,
        chunk_bytes: bytes,
        chunk_sha256: str,
    ) -> CameraChunkAcceptResult:
        chunk_size = len(chunk_bytes)
        if identity.total <= 0 or identity.total > MAX_IMAGE_BYTES:
            raise CameraChunkError(413, "total_too_large", "total exceeds the JPEG size limit")
        if chunk_size <= 0:
            raise CameraChunkError(400, "empty_chunk", "chunk body cannot be empty")
        if chunk_size > MAX_CHUNK_BYTES:
            raise CameraChunkError(413, "chunk_too_large", "chunk exceeds the 4096 byte limit")
        if offset < 0:
            raise CameraChunkError(400, "invalid_offset", "offset cannot be negative")
        if offset + chunk_size > identity.total:
            raise CameraChunkError(413, "chunk_exceeds_total", "chunk would exceed total")
        if offset + chunk_size < identity.total and chunk_size != MAX_CHUNK_BYTES:
            raise CameraChunkError(
                400,
                "non_final_chunk_size",
                "every non-final chunk must contain exactly 4096 bytes",
            )
        if hashlib.sha256(chunk_bytes).hexdigest() != chunk_sha256:
            raise CameraChunkError(
                422,
                "chunk_sha256_mismatch",
                "chunk SHA-256 does not match chunk bytes",
            )
        self.maybe_cleanup()
        now = float(self.clock())
        connection = self._connect()
        appended_path: Path | None = None
        previous_received = 0
        expired_rows: list[sqlite3.Row] = []
        transaction_committed = False
        try:
            connection.execute("BEGIN IMMEDIATE")
            expired_rows = self._expired_rows_locked(connection, now)
            row = connection.execute(
                "SELECT * FROM camera_chunk_uploads WHERE request_id = ?",
                (identity.request_id,),
            ).fetchone()
            if row is None:
                if offset != 0:
                    raise CameraChunkError(404, "upload_not_found", "camera upload session does not exist")
                self._ensure_capacity_locked(connection, identity)
                temp_path = self._new_temp_path(identity)
                connection.execute(
                    """
                    INSERT INTO camera_chunk_uploads (
                        request_id, device_id, total, image_sha256, received, state,
                        temp_path, created_at, updated_at, expires_at
                    ) VALUES (?, ?, ?, ?, 0, 'uploading', ?, ?, ?, ?)
                    """,
                    (
                        identity.request_id,
                        identity.device_id,
                        identity.total,
                        identity.image_sha256,
                        str(temp_path),
                        now,
                        now,
                        now + self.session_ttl_seconds,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM camera_chunk_uploads WHERE request_id = ?",
                    (identity.request_id,),
                ).fetchone()
            elif not self._identity_matches(row, identity):
                raise CameraChunkError(
                    409,
                    "upload_identity_conflict",
                    "request_id belongs to another device, total, or image SHA-256",
                    next_offset=int(row["received"]),
                )

            received = int(row["received"])
            if offset < received:
                part = connection.execute(
                    """
                    SELECT chunk_size, chunk_sha256 FROM camera_chunk_parts
                    WHERE request_id = ? AND offset = ?
                    """,
                    (identity.request_id, offset),
                ).fetchone()
                if (
                    part is None
                    or int(part["chunk_size"]) != len(chunk_bytes)
                    or part["chunk_sha256"] != chunk_sha256
                ):
                    raise CameraChunkError(
                        409,
                        "chunk_conflict",
                        "repeated chunk does not match the previously accepted chunk",
                        next_offset=received,
                    )
                connection.commit()
                transaction_committed = True
                current = self.get(identity.request_id)
                if current is None:
                    raise CameraChunkError(500, "upload_state_lost", "camera upload state was lost")
                return CameraChunkAcceptResult("duplicate", current, offset, len(chunk_bytes))

            if offset > received:
                raise CameraChunkError(
                    409,
                    "unexpected_offset",
                    "chunk offset is ahead of the server next_offset",
                    next_offset=received,
                )
            if row["state"] != "uploading":
                raise CameraChunkError(
                    409,
                    "upload_already_finished",
                    "camera upload is already being processed or completed",
                    next_offset=received,
                )

            path = Path(str(row["temp_path"]))
            previous_received = received
            self._reconcile_file(path, received)
            path.parent.mkdir(parents=True, exist_ok=True)
            appended_path = path
            with path.open("ab") as handle:
                handle.write(chunk_bytes)
                handle.flush()
                os.fsync(handle.fileno())

            next_offset = received + len(chunk_bytes)
            connection.execute(
                """
                INSERT INTO camera_chunk_parts (request_id, offset, chunk_size, chunk_sha256)
                VALUES (?, ?, ?, ?)
                """,
                (identity.request_id, offset, len(chunk_bytes), chunk_sha256),
            )
            connection.execute(
                """
                UPDATE camera_chunk_uploads
                SET received = ?, updated_at = ?, expires_at = ?
                WHERE request_id = ?
                """,
                (
                    next_offset,
                    now,
                    now + self.session_ttl_seconds,
                    identity.request_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM camera_chunk_uploads WHERE request_id = ?",
                (identity.request_id,),
            ).fetchone()
            connection.commit()
            transaction_committed = True
        except Exception:
            if appended_path is not None:
                try:
                    with appended_path.open("r+b") as handle:
                        handle.truncate(previous_received)
                        handle.flush()
                        os.fsync(handle.fileno())
                except OSError:
                    pass
            connection.rollback()
            raise
        finally:
            connection.close()
            if transaction_committed:
                for expired in expired_rows:
                    self._unlink(Path(str(expired["temp_path"])) if expired["temp_path"] else None)
                    self._log_expired(expired)

        return CameraChunkAcceptResult(
            "accepted",
            self._session_from_row(updated),
            offset,
            len(chunk_bytes),
        )

    def get(self, request_id: str) -> CameraChunkSession | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM camera_chunk_uploads WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return self._session_from_row(row) if row is not None else None

    def prepare_finish(
        self,
        request_id: str,
        device_id: str,
        image_sha256: str,
    ) -> CameraChunkSession:
        self.maybe_cleanup()
        now = float(self.clock())
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM camera_chunk_uploads WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise CameraChunkError(404, "upload_not_found", "camera upload session does not exist")
            if row["device_id"] != device_id or row["image_sha256"] != image_sha256:
                connection.rollback()
                raise CameraChunkError(
                    409,
                    "upload_identity_conflict",
                    "device or image SHA-256 does not match the upload session",
                    next_offset=int(row["received"]),
                )
            if int(row["received"]) != int(row["total"]):
                connection.rollback()
                raise CameraChunkError(
                    409,
                    "upload_incomplete",
                    "camera upload has not received all bytes",
                    next_offset=int(row["received"]),
                )
            if row["state"] == "uploading":
                connection.execute(
                    """
                    UPDATE camera_chunk_uploads
                    SET state = 'processing', updated_at = ?, expires_at = ?
                    WHERE request_id = ? AND state = 'uploading'
                    """,
                    (now, now + self.processing_lease_seconds, request_id),
                )
                row = connection.execute(
                    "SELECT * FROM camera_chunk_uploads WHERE request_id = ?",
                    (request_id,),
                ).fetchone()
            connection.commit()
        return self._session_from_row(row)

    def refresh_processing(self, request_id: str) -> bool:
        now = float(self.clock())
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE camera_chunk_uploads
                SET updated_at = ?, expires_at = ?
                WHERE request_id = ? AND state = 'processing'
                """,
                (now, now + self.processing_lease_seconds, request_id),
            )
            return cursor.rowcount == 1

    def complete(
        self,
        identity: CameraChunkIdentity,
        outcome: CameraStoredOutcome,
    ) -> CameraChunkSession:
        now = float(self.clock())
        response_json = json.dumps(outcome.payload, ensure_ascii=False, separators=(",", ":"))
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM camera_chunk_uploads WHERE request_id = ?",
                (identity.request_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                raise CameraChunkError(404, "upload_not_found", "camera upload session does not exist")
            if not self._identity_matches(row, identity):
                connection.rollback()
                raise CameraChunkError(409, "upload_identity_conflict", "camera upload identity changed")
            if row["state"] == "completed":
                connection.commit()
                return self._session_from_row(row)
            connection.execute(
                """
                UPDATE camera_chunk_uploads
                SET state = 'completed', http_status = ?, response_json = ?,
                    updated_at = ?, expires_at = ?
                WHERE request_id = ?
                """,
                (
                    outcome.status_code,
                    response_json,
                    now,
                    now + self.completed_ttl_seconds,
                    identity.request_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM camera_chunk_uploads WHERE request_id = ?",
                (identity.request_id,),
            ).fetchone()
            connection.commit()
        self._unlink(Path(str(row["temp_path"])) if row["temp_path"] else None)
        return self._session_from_row(updated)

    def cancel(self, request_id: str, device_id: str) -> str:
        self.maybe_cleanup()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM camera_chunk_uploads WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None:
                connection.commit()
                return "not_found"
            if row["device_id"] != device_id:
                connection.rollback()
                raise CameraChunkError(
                    409,
                    "upload_identity_conflict",
                    "request_id belongs to another device",
                    next_offset=int(row["received"]),
                )
            if row["state"] == "processing":
                connection.rollback()
                raise CameraChunkError(
                    409,
                    "recognition_in_progress",
                    "recognition has started and cannot be safely cancelled",
                    next_offset=int(row["received"]),
                )
            if row["state"] == "completed":
                connection.rollback()
                raise CameraChunkError(
                    409,
                    "upload_already_completed",
                    "camera upload has already completed",
                    next_offset=int(row["received"]),
                )
            connection.execute(
                "DELETE FROM camera_chunk_uploads WHERE request_id = ?",
                (request_id,),
            )
            connection.commit()
        self._unlink(Path(str(row["temp_path"])) if row["temp_path"] else None)
        return "cancelled"

    def count(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM camera_chunk_uploads").fetchone()
            return int(row["count"])


_stores: dict[tuple[str, str, float, float, int, int, float], CameraChunkStore] = {}
_stores_lock = threading.Lock()


def get_camera_chunk_store() -> CameraChunkStore:
    db_path = camera_idempotency_db_path().resolve()
    temp_dir = camera_chunk_temp_dir().resolve()
    session_ttl = camera_chunk_session_ttl_seconds()
    completed_ttl = camera_chunk_completed_ttl_seconds()
    max_sessions = camera_chunk_max_sessions()
    max_temp_bytes = camera_chunk_max_temp_bytes()
    cleanup_interval = camera_chunk_cleanup_interval_seconds()
    key = (
        str(db_path),
        str(temp_dir),
        session_ttl,
        completed_ttl,
        max_sessions,
        max_temp_bytes,
        cleanup_interval,
    )
    with _stores_lock:
        store = _stores.get(key)
        if store is None:
            store = CameraChunkStore(
                db_path,
                temp_dir,
                session_ttl_seconds=session_ttl,
                completed_ttl_seconds=completed_ttl,
                max_sessions=max_sessions,
                max_temp_bytes=max_temp_bytes,
                cleanup_interval_seconds=cleanup_interval,
            )
            _stores[key] = store
        return store
