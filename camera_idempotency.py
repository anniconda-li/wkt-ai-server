from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
from uuid import uuid4


DEFAULT_TTL_SECONDS = 20 * 60
DEFAULT_MAX_RECORDS = 1000
DEFAULT_WAIT_TIMEOUT_SECONDS = 180
DEFAULT_POLL_INTERVAL_SECONDS = 0.1


class CameraIdempotencyError(RuntimeError):
    pass


class CameraIdempotencyCapacityError(CameraIdempotencyError):
    pass


@dataclass(frozen=True)
class CameraRequestIdentity:
    request_id: str
    device_id: str
    sha256: str
    content_length: int


@dataclass(frozen=True)
class CameraStoredOutcome:
    status_code: int
    payload: dict[str, Any]


@dataclass(frozen=True)
class CameraIdempotencyClaim:
    action: str
    owner_token: str | None = None
    outcome: CameraStoredOutcome | None = None
    existing_device_id: str | None = None
    existing_sha256: str | None = None
    existing_content_length: int | None = None


def _positive_number(name: str, default: float) -> float:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError:
        return default
    return value if value > 0 else default


def camera_idempotency_ttl_seconds() -> float:
    return _positive_number("CAMERA_IDEMPOTENCY_TTL_SECONDS", DEFAULT_TTL_SECONDS)


def camera_idempotency_max_records() -> int:
    return max(
        1,
        int(_positive_number("CAMERA_IDEMPOTENCY_MAX_RECORDS", DEFAULT_MAX_RECORDS)),
    )


def camera_idempotency_wait_timeout_seconds() -> float:
    return _positive_number(
        "CAMERA_IDEMPOTENCY_WAIT_TIMEOUT_SECONDS",
        DEFAULT_WAIT_TIMEOUT_SECONDS,
    )


def camera_idempotency_poll_interval_seconds() -> float:
    return _positive_number(
        "CAMERA_IDEMPOTENCY_POLL_INTERVAL_SECONDS",
        DEFAULT_POLL_INTERVAL_SECONDS,
    )


def camera_idempotency_db_path() -> Path:
    configured = os.getenv("CAMERA_IDEMPOTENCY_DB_PATH", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).parent / "uploads" / "camera_idempotency.sqlite3"


class CameraIdempotencyStore:
    def __init__(
        self,
        path: Path,
        *,
        ttl_seconds: float,
        max_records: int,
        clock: Any = time.time,
    ) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.max_records = max_records
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS camera_idempotency (
                    request_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    content_length INTEGER NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('processing', 'completed')),
                    owner_token TEXT NOT NULL,
                    http_status INTEGER,
                    response_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_camera_idempotency_expires "
                "ON camera_idempotency(expires_at)"
            )

    @staticmethod
    def _row_matches(row: sqlite3.Row, identity: CameraRequestIdentity) -> bool:
        return (
            row["device_id"] == identity.device_id
            and row["sha256"] == identity.sha256
            and row["content_length"] == identity.content_length
        )

    @staticmethod
    def _outcome_from_row(row: sqlite3.Row) -> CameraStoredOutcome | None:
        if row["state"] != "completed":
            return None
        if row["http_status"] is None or row["response_json"] is None:
            raise CameraIdempotencyError("completed idempotency record has no outcome")
        payload = json.loads(row["response_json"])
        if not isinstance(payload, dict):
            raise CameraIdempotencyError("stored idempotency outcome must be a JSON object")
        return CameraStoredOutcome(status_code=int(row["http_status"]), payload=payload)

    def _delete_expired(self, connection: sqlite3.Connection, now: float) -> int:
        cursor = connection.execute(
            "DELETE FROM camera_idempotency WHERE expires_at <= ?",
            (now,),
        )
        return max(cursor.rowcount, 0)

    def _make_capacity(self, connection: sqlite3.Connection) -> None:
        row = connection.execute("SELECT COUNT(*) AS count FROM camera_idempotency").fetchone()
        count = int(row["count"])
        if count < self.max_records:
            return
        remove_count = count - self.max_records + 1
        connection.execute(
            """
            DELETE FROM camera_idempotency
            WHERE request_id IN (
                SELECT request_id
                FROM camera_idempotency
                WHERE state = 'completed'
                ORDER BY updated_at ASC
                LIMIT ?
            )
            """,
            (remove_count,),
        )
        row = connection.execute("SELECT COUNT(*) AS count FROM camera_idempotency").fetchone()
        if int(row["count"]) >= self.max_records:
            raise CameraIdempotencyCapacityError("camera idempotency capacity is full")

    def claim(self, identity: CameraRequestIdentity) -> CameraIdempotencyClaim:
        now = float(self.clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._delete_expired(connection, now)
            row = connection.execute(
                "SELECT * FROM camera_idempotency WHERE request_id = ?",
                (identity.request_id,),
            ).fetchone()
            if row is not None:
                if not self._row_matches(row, identity):
                    connection.commit()
                    return CameraIdempotencyClaim(
                        action="conflict",
                        existing_device_id=str(row["device_id"]),
                        existing_sha256=str(row["sha256"]),
                        existing_content_length=int(row["content_length"]),
                    )
                outcome = self._outcome_from_row(row)
                connection.commit()
                return CameraIdempotencyClaim(
                    action="hit" if outcome is not None else "wait",
                    outcome=outcome,
                )

            self._make_capacity(connection)
            owner_token = uuid4().hex
            connection.execute(
                """
                INSERT INTO camera_idempotency (
                    request_id, device_id, sha256, content_length, state,
                    owner_token, created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, 'processing', ?, ?, ?, ?)
                """,
                (
                    identity.request_id,
                    identity.device_id,
                    identity.sha256,
                    identity.content_length,
                    owner_token,
                    now,
                    now,
                    now + self.ttl_seconds,
                ),
            )
            connection.commit()
            return CameraIdempotencyClaim(action="owner", owner_token=owner_token)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def complete(
        self,
        identity: CameraRequestIdentity,
        owner_token: str,
        outcome: CameraStoredOutcome,
    ) -> bool:
        now = float(self.clock())
        response_json = json.dumps(outcome.payload, ensure_ascii=False, separators=(",", ":"))
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE camera_idempotency
                SET state = 'completed', http_status = ?, response_json = ?,
                    updated_at = ?, expires_at = ?
                WHERE request_id = ? AND owner_token = ? AND state = 'processing'
                """,
                (
                    outcome.status_code,
                    response_json,
                    now,
                    now + self.ttl_seconds,
                    identity.request_id,
                    owner_token,
                ),
            )
            return cursor.rowcount == 1

    def get(self, identity: CameraRequestIdentity) -> CameraIdempotencyClaim | None:
        now = float(self.clock())
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._delete_expired(connection, now)
            row = connection.execute(
                "SELECT * FROM camera_idempotency WHERE request_id = ?",
                (identity.request_id,),
            ).fetchone()
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        if row is None:
            return None
        if not self._row_matches(row, identity):
            return CameraIdempotencyClaim(
                action="conflict",
                existing_device_id=str(row["device_id"]),
                existing_sha256=str(row["sha256"]),
                existing_content_length=int(row["content_length"]),
            )
        outcome = self._outcome_from_row(row)
        return CameraIdempotencyClaim(
            action="hit" if outcome is not None else "wait",
            outcome=outcome,
        )

    def cleanup(self) -> int:
        now = float(self.clock())
        with closing(self._connect()) as connection:
            return self._delete_expired(connection, now)

    def count(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM camera_idempotency").fetchone()
            return int(row["count"])


_stores: dict[tuple[str, float, int], CameraIdempotencyStore] = {}
_stores_lock = threading.Lock()


def get_camera_idempotency_store() -> CameraIdempotencyStore:
    path = camera_idempotency_db_path().resolve()
    ttl_seconds = camera_idempotency_ttl_seconds()
    max_records = camera_idempotency_max_records()
    key = (str(path), ttl_seconds, max_records)
    with _stores_lock:
        store = _stores.get(key)
        if store is None:
            store = CameraIdempotencyStore(
                path,
                ttl_seconds=ttl_seconds,
                max_records=max_records,
            )
            _stores[key] = store
        return store
