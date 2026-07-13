import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from starlette.requests import Request

import main
import sessions
import vision
from camera_chunk_upload import (
    CameraChunkError,
    CameraChunkIdentity,
    CameraChunkStore,
    get_camera_chunk_store,
)
from main import app, read_camera_chunk_body


SAMPLE_JPEG = Path("samples/camera/yingguo_jade_eagle_esp32.jpg").read_bytes()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def recognized_result() -> dict[str, object]:
    return {
        "mode": "vision_llm",
        "artifact_id": "yingguo_jade_eagle",
        "artifact_name": "应国玉鹰",
        "predicted_artifact_id": "yingguo_jade_eagle",
        "confidence": 0.91,
        "accepted": True,
        "min_confidence": 0.60,
        "evidence": ["鹰形"],
        "vision_description": "识别为应国玉鹰。",
    }


class CameraChunkApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.env_patch = patch.dict(
            os.environ,
            {
                "CAMERA_IDEMPOTENCY_DB_PATH": str(self.root / "camera.sqlite3"),
                "CAMERA_IDEMPOTENCY_TTL_SECONDS": "1200",
                "CAMERA_IDEMPOTENCY_MAX_RECORDS": "100",
                "CAMERA_IDEMPOTENCY_WAIT_TIMEOUT_SECONDS": "5",
                "CAMERA_IDEMPOTENCY_POLL_INTERVAL_SECONDS": "0.01",
                "CAMERA_CHUNK_TEMP_DIR": str(self.root / "chunks"),
                "CAMERA_CHUNK_SESSION_TTL_SECONDS": "600",
                "CAMERA_CHUNK_COMPLETED_TTL_SECONDS": "1200",
                "CAMERA_CHUNK_MAX_SESSIONS": "20",
                "CAMERA_CHUNK_MAX_TEMP_BYTES": str(16 * 1024 * 1024),
                "CAMERA_CHUNK_CLEANUP_INTERVAL_SECONDS": "60",
            },
        )
        self.env_patch.start()
        self.upload_patch = patch.object(vision, "UPLOADS_DIR", self.root / "uploads")
        self.upload_patch.start()
        sessions.sessions.clear()

    def tearDown(self) -> None:
        self.upload_patch.stop()
        self.env_patch.stop()
        self.temp_dir.cleanup()

    def post_chunk(
        self,
        data: bytes,
        *,
        request_id: str,
        offset: int,
        total: int,
        device: str = "chunk-device",
        image_sha256: str | None = None,
        chunk_sha256: str | None = None,
        forwarded_for: str | None = None,
        client_ip: str | None = None,
    ):
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(data)),
            "X-Image-SHA256": image_sha256 or digest(SAMPLE_JPEG),
            "X-Chunk-SHA256": chunk_sha256 or digest(data),
        }
        if forwarded_for:
            headers["X-Forwarded-For"] = forwarded_for
        return TestClient(app, client=(client_ip or "testclient", 50000)).post(
            "/camera/upload/chunk",
            params={
                "device": device,
                "request_id": request_id,
                "offset": offset,
                "total": total,
            },
            content=data,
            headers=headers,
        )

    def finish(
        self,
        request_id: str,
        *,
        device: str = "chunk-device",
        image_sha256: str | None = None,
    ):
        return TestClient(app).post(
            "/camera/upload/finish",
            params={"device": device, "request_id": request_id},
            headers={"X-Image-SHA256": image_sha256 or digest(SAMPLE_JPEG)},
        )

    def upload_all(
        self,
        request_id: str,
        data: bytes = SAMPLE_JPEG,
        *,
        image_sha256: str | None = None,
        forwarded_ips: list[str] | None = None,
    ) -> list:
        responses = []
        image_hash = image_sha256 or digest(data)
        for index, offset in enumerate(range(0, len(data), 4096)):
            chunk = data[offset : offset + 4096]
            responses.append(
                self.post_chunk(
                    chunk,
                    request_id=request_id,
                    offset=offset,
                    total=len(data),
                    image_sha256=image_hash,
                    forwarded_for=(forwarded_ips or [None])[index % len(forwarded_ips or [None])],
                    client_ip=(forwarded_ips or [None])[index % len(forwarded_ips or [None])],
                )
            )
        return responses

    def test_four_kib_chunks_and_short_final_chunk_finish_successfully(self) -> None:
        responses = self.upload_all("chunk-success")
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual([response.json()["chunk_size"] for response in responses[:-1]], [4096] * 3)
        self.assertLess(responses[-1].json()["chunk_size"], 4096)
        self.assertTrue(responses[-1].json()["complete"])

        with patch.object(main, "is_vision_configured", return_value=False):
            finished = self.finish("chunk-success")
        self.assertEqual(finished.status_code, 200, finished.text)
        self.assertEqual(finished.json()["status"], "ready")
        self.assertEqual(finished.json()["device_id"], "chunk-device")
        self.assertIn("recognition", finished.json())

    def test_repeated_chunk_does_not_append_twice(self) -> None:
        chunk = SAMPLE_JPEG[:4096]
        first = self.post_chunk(chunk, request_id="chunk-duplicate", offset=0, total=len(SAMPLE_JPEG))
        session = get_camera_chunk_store().get("chunk-duplicate")
        self.assertIsNotNone(session)
        path = session.temp_path
        self.assertEqual(path.stat().st_size, 4096)

        repeated = self.post_chunk(chunk, request_id="chunk-duplicate", offset=0, total=len(SAMPLE_JPEG))
        self.assertEqual(first.status_code, 200)
        self.assertEqual(repeated.status_code, 200)
        self.assertEqual(repeated.json()["next_offset"], 4096)
        self.assertEqual(path.stat().st_size, 4096)

    def test_offset_jump_returns_server_next_offset(self) -> None:
        self.post_chunk(
            SAMPLE_JPEG[:4096],
            request_id="chunk-offset",
            offset=0,
            total=len(SAMPLE_JPEG),
        )
        response = self.post_chunk(
            SAMPLE_JPEG[8192:12288],
            request_id="chunk-offset",
            offset=8192,
            total=len(SAMPLE_JPEG),
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error_code"], "unexpected_offset")
        self.assertEqual(response.json()["next_offset"], 4096)

    def test_request_id_identity_conflicts_return_409(self) -> None:
        request_id = "chunk-identity"
        first_chunk = SAMPLE_JPEG[:4096]
        self.post_chunk(first_chunk, request_id=request_id, offset=0, total=len(SAMPLE_JPEG))

        different_device = self.post_chunk(
            SAMPLE_JPEG[4096:8192],
            request_id=request_id,
            offset=4096,
            total=len(SAMPLE_JPEG),
            device="other-device",
        )
        different_total = self.post_chunk(
            SAMPLE_JPEG[4096:8192],
            request_id=request_id,
            offset=4096,
            total=len(SAMPLE_JPEG) + 1,
        )
        different_sha = self.post_chunk(
            SAMPLE_JPEG[4096:8192],
            request_id=request_id,
            offset=4096,
            total=len(SAMPLE_JPEG),
            image_sha256="0" * 64,
        )
        self.assertEqual(different_device.status_code, 409)
        self.assertEqual(different_total.status_code, 409)
        self.assertEqual(different_sha.status_code, 409)

    def test_wrong_chunk_sha_returns_422(self) -> None:
        response = self.post_chunk(
            SAMPLE_JPEG[:4096],
            request_id="chunk-bad-sha",
            offset=0,
            total=len(SAMPLE_JPEG),
            chunk_sha256="0" * 64,
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error_code"], "chunk_sha256_mismatch")
        self.assertIsNone(get_camera_chunk_store().get("chunk-bad-sha"))

    def test_chunk_cannot_exceed_total(self) -> None:
        response = self.post_chunk(
            SAMPLE_JPEG[:4096],
            request_id="chunk-over-total",
            offset=0,
            total=3000,
            image_sha256=digest(SAMPLE_JPEG[:3000]),
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["error_code"], "chunk_exceeds_total")

    def test_finish_before_all_chunks_returns_409(self) -> None:
        self.post_chunk(
            SAMPLE_JPEG[:4096],
            request_id="chunk-incomplete",
            offset=0,
            total=len(SAMPLE_JPEG),
        )
        response = self.finish("chunk-incomplete")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error_code"], "upload_incomplete")
        self.assertEqual(response.json()["next_offset"], 4096)

    def test_full_sha_mismatch_does_not_save_or_recognize(self) -> None:
        bad_full_hash = "0" * 64
        self.upload_all("chunk-full-sha", image_sha256=bad_full_hash)
        recognize = AsyncMock(return_value=recognized_result())
        with (
            patch.object(main, "is_vision_configured", return_value=True),
            patch.object(main, "recognize_artifact_from_image", recognize),
            patch.object(main, "save_camera_image", wraps=main.save_camera_image) as save,
        ):
            response = self.finish("chunk-full-sha", image_sha256=bad_full_hash)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error_code"], "image_sha256_mismatch")
        save.assert_not_called()
        recognize.assert_not_awaited()

    def test_invalid_jpeg_does_not_recognize(self) -> None:
        invalid = b"x" * 5000
        self.upload_all("chunk-invalid-jpeg", invalid)
        recognize = AsyncMock(return_value=recognized_result())
        with (
            patch.object(main, "is_vision_configured", return_value=True),
            patch.object(main, "recognize_artifact_from_image", recognize),
        ):
            response = self.finish(
                "chunk-invalid-jpeg",
                image_sha256=digest(invalid),
            )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error_code"], "invalid_jpeg")
        recognize.assert_not_awaited()

    def test_repeated_finish_recognizes_once_and_returns_same_result(self) -> None:
        self.upload_all("chunk-finish-retry")
        recognize = AsyncMock(return_value=recognized_result())
        with (
            patch.object(main, "is_vision_configured", return_value=True),
            patch.object(main, "recognize_artifact_from_image", recognize),
        ):
            first = self.finish("chunk-finish-retry")
            second = self.finish("chunk-finish-retry")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(recognize.await_count, 1)

    def test_concurrent_finish_recognizes_once(self) -> None:
        self.upload_all("chunk-finish-concurrent")
        calls = 0
        lock = threading.Lock()

        async def delayed_recognition(*args, **kwargs):
            nonlocal calls
            with lock:
                calls += 1
            await asyncio.sleep(0.15)
            return recognized_result()

        with (
            patch.object(main, "is_vision_configured", return_value=True),
            patch.object(main, "recognize_artifact_from_image", side_effect=delayed_recognition),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            responses = list(pool.map(lambda _: self.finish("chunk-finish-concurrent"), range(2)))
        self.assertEqual([response.status_code for response in responses], [200, 200])
        self.assertEqual(responses[0].json(), responses[1].json())
        self.assertEqual(calls, 1)

    def test_cancel_removes_temporary_upload_and_is_idempotent(self) -> None:
        self.post_chunk(
            SAMPLE_JPEG[:4096],
            request_id="chunk-cancel",
            offset=0,
            total=len(SAMPLE_JPEG),
        )
        store = get_camera_chunk_store()
        path = store.get("chunk-cancel").temp_path
        first = TestClient(app).post(
            "/camera/upload/cancel",
            params={"device": "chunk-device", "request_id": "chunk-cancel"},
        )
        second = TestClient(app).post(
            "/camera/upload/cancel",
            params={"device": "chunk-device", "request_id": "chunk-cancel"},
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertFalse(path.exists())
        self.assertIsNone(store.get("chunk-cancel"))

    def test_cancel_cannot_delete_another_devices_upload(self) -> None:
        self.post_chunk(
            SAMPLE_JPEG[:4096],
            request_id="chunk-cancel-device",
            offset=0,
            total=len(SAMPLE_JPEG),
        )
        response = TestClient(app).post(
            "/camera/upload/cancel",
            params={"device": "other-device", "request_id": "chunk-cancel-device"},
        )
        self.assertEqual(response.status_code, 409)
        self.assertIsNotNone(get_camera_chunk_store().get("chunk-cancel-device"))

    def test_cancel_reports_when_recognition_is_in_progress(self) -> None:
        self.upload_all("chunk-cancel-processing")
        started = threading.Event()
        release = threading.Event()

        async def delayed_recognition(*args, **kwargs):
            started.set()
            await asyncio.to_thread(release.wait)
            return recognized_result()

        with (
            patch.object(main, "is_vision_configured", return_value=True),
            patch.object(main, "recognize_artifact_from_image", side_effect=delayed_recognition),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            future = pool.submit(self.finish, "chunk-cancel-processing")
            self.assertTrue(started.wait(timeout=2))
            cancelled = TestClient(app).post(
                "/camera/upload/cancel",
                params={
                    "device": "chunk-device",
                    "request_id": "chunk-cancel-processing",
                },
            )
            release.set()
            finished = future.result(timeout=3)
        self.assertEqual(cancelled.status_code, 409)
        self.assertEqual(cancelled.json()["error_code"], "recognition_in_progress")
        self.assertEqual(finished.status_code, 200)

    def test_different_public_ips_share_request_id(self) -> None:
        responses = self.upload_all(
            "chunk-changing-ip",
            forwarded_ips=["219.157.76.138", "218.28.63.102", "219.157.79.110"],
        )
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertTrue(responses[-1].json()["complete"])

    def test_chunk_log_contains_progress_fields(self) -> None:
        with self.assertLogs("wkt_ai_server.main", level="INFO") as logs:
            response = self.post_chunk(
                SAMPLE_JPEG[:4096],
                request_id="chunk-log-fields",
                offset=0,
                total=len(SAMPLE_JPEG),
            )
        self.assertEqual(response.status_code, 200)
        output = "\n".join(logs.output)
        self.assertIn("camera.chunk.accepted", output)
        for field in (
            "device=",
            "request_id=",
            "offset=",
            "chunk_size=",
            "received=",
            "total=",
            "next_offset=",
            "stage_ms=",
            "result=",
        ):
            self.assertIn(field, output)

    def test_legacy_single_post_still_succeeds(self) -> None:
        with patch.object(main, "is_vision_configured", return_value=False):
            response = TestClient(app).post(
                "/camera/upload",
                params={"device": "legacy-camera"},
                content=SAMPLE_JPEG,
                headers={"Content-Type": "image/jpeg", "Content-Length": str(len(SAMPLE_JPEG))},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")


class CameraChunkBodyTests(unittest.TestCase):
    def test_content_length_mismatch_fails(self) -> None:
        messages = iter([{"type": "http.request", "body": b"abc", "more_body": False}])

        async def receive():
            return next(messages)

        request = Request(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "POST",
                "scheme": "http",
                "path": "/camera/upload/chunk",
                "raw_path": b"/camera/upload/chunk",
                "query_string": b"",
                "headers": [],
                "client": ("127.0.0.1", 1),
                "server": ("127.0.0.1", 8000),
            },
            receive,
        )
        with self.assertRaises(CameraChunkError) as raised:
            asyncio.run(read_camera_chunk_body(request, 4))
        self.assertEqual(raised.exception.error_code, "content_length_mismatch")


class CameraChunkCleanupTests(unittest.TestCase):
    def test_independent_store_instances_share_chunk_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            kwargs = {
                "session_ttl_seconds": 60,
                "completed_ttl_seconds": 120,
                "max_sessions": 10,
                "max_temp_bytes": 1024 * 1024,
                "cleanup_interval_seconds": 60,
            }
            first_store = CameraChunkStore(root / "shared.sqlite3", root / "chunks", **kwargs)
            second_store = CameraChunkStore(root / "shared.sqlite3", root / "chunks", **kwargs)
            identity = CameraChunkIdentity("shared", "device", 5000, digest(b"a" * 5000))
            first_store.accept_chunk(
                identity,
                offset=0,
                chunk_bytes=b"a" * 4096,
                chunk_sha256=digest(b"a" * 4096),
            )
            duplicate = second_store.accept_chunk(
                identity,
                offset=0,
                chunk_bytes=b"a" * 4096,
                chunk_sha256=digest(b"a" * 4096),
            )
            self.assertEqual(duplicate.action, "duplicate")
            self.assertEqual(duplicate.session.received, 4096)

    def test_expired_session_and_file_are_cleaned(self) -> None:
        now = [1000.0]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = CameraChunkStore(
                root / "camera.sqlite3",
                root / "chunks",
                session_ttl_seconds=10,
                completed_ttl_seconds=20,
                max_sessions=10,
                max_temp_bytes=1024 * 1024,
                cleanup_interval_seconds=1,
                clock=lambda: now[0],
            )
            identity = CameraChunkIdentity("expired", "device", 3, digest(b"abc"))
            accepted = store.accept_chunk(
                identity,
                offset=0,
                chunk_bytes=b"abc",
                chunk_sha256=digest(b"abc"),
            )
            path = accepted.session.temp_path
            self.assertTrue(path.exists())
            now[0] += 11
            self.assertEqual(store.cleanup(), 1)
            self.assertFalse(path.exists())
            self.assertIsNone(store.get("expired"))

    def test_capacity_and_reserved_space_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_limited = CameraChunkStore(
                root / "sessions.sqlite3",
                root / "session-chunks",
                session_ttl_seconds=60,
                completed_ttl_seconds=120,
                max_sessions=1,
                max_temp_bytes=1024 * 1024,
                cleanup_interval_seconds=60,
            )
            first = CameraChunkIdentity("first", "device", 4000, digest(b"a" * 4000))
            second = CameraChunkIdentity("second", "device", 4000, digest(b"b" * 4000))
            session_limited.accept_chunk(
                first,
                offset=0,
                chunk_bytes=b"a" * 4000,
                chunk_sha256=digest(b"a" * 4000),
            )
            with self.assertRaises(CameraChunkError) as raised:
                session_limited.accept_chunk(
                    second,
                    offset=0,
                    chunk_bytes=b"b" * 4000,
                    chunk_sha256=digest(b"b" * 4000),
                )
            self.assertEqual(raised.exception.error_code, "session_capacity_full")

            space_limited = CameraChunkStore(
                root / "space.sqlite3",
                root / "space-chunks",
                session_ttl_seconds=60,
                completed_ttl_seconds=120,
                max_sessions=10,
                max_temp_bytes=5000,
                cleanup_interval_seconds=60,
            )
            space_limited.accept_chunk(
                first,
                offset=0,
                chunk_bytes=b"a" * 4000,
                chunk_sha256=digest(b"a" * 4000),
            )
            too_large = CameraChunkIdentity("third", "device", 2000, digest(b"c" * 2000))
            with self.assertRaises(CameraChunkError) as raised:
                space_limited.accept_chunk(
                    too_large,
                    offset=0,
                    chunk_bytes=b"c" * 2000,
                    chunk_sha256=digest(b"c" * 2000),
                )
            self.assertEqual(raised.exception.error_code, "temporary_space_limit")


if __name__ == "__main__":
    unittest.main()
