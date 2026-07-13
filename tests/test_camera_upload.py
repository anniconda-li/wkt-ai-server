import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from starlette.requests import Request

import main
import sessions
import vision
from camera_idempotency import (
    CameraIdempotencyStore,
    CameraRequestIdentity,
    CameraStoredOutcome,
    get_camera_idempotency_store,
)
from main import (
    CameraLogContext,
    CameraUploadHeaders,
    CameraUploadReadError,
    app,
    parse_camera_upload_headers,
    process_idempotent_camera_upload,
    read_camera_upload_body,
)
from vision import MAX_IMAGE_BYTES


SAMPLE_ONE = Path("samples/camera/yingguo_jade_eagle_esp32.jpg").read_bytes()
SAMPLE_TWO = Path("samples/camera/shuyao_chuilin_sheng_ding_esp32.jpg").read_bytes()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_headers(data: bytes, request_id: str) -> dict[str, str]:
    return {
        "Content-Type": "image/jpeg",
        "Content-Length": str(len(data)),
        "X-Request-ID": request_id,
        "X-Content-SHA256": sha256(data),
    }


def vision_result() -> dict[str, object]:
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


def build_request(
    receive: Callable[[], Awaitable[dict[str, object]]],
    *,
    content_length: int | None,
    request_id: str | None = None,
    content_sha256: str | None = None,
) -> Request:
    headers: list[tuple[bytes, bytes]] = [(b"content-type", b"image/jpeg")]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode("ascii")))
    if request_id is not None:
        headers.append((b"x-request-id", request_id.encode("ascii")))
    if content_sha256 is not None:
        headers.append((b"x-content-sha256", content_sha256.encode("ascii")))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/camera/upload",
        "raw_path": b"/camera/upload",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 18080),
    }
    return Request(scope, receive)


class CameraUploadBodyTests(unittest.TestCase):
    def test_reads_streamed_body_and_checks_content_length(self) -> None:
        messages = iter(
            [
                {"type": "http.request", "body": b"abcd", "more_body": True},
                {"type": "http.request", "body": b"ef", "more_body": False},
            ]
        )

        async def receive() -> dict[str, object]:
            return next(messages)

        request = build_request(receive, content_length=6)
        headers = parse_camera_upload_headers(request)
        body = asyncio.run(read_camera_upload_body(request, headers))
        self.assertEqual(body.image_bytes, b"abcdef")
        self.assertEqual(body.received_bytes, 6)
        self.assertEqual(body.sha256, sha256(b"abcdef"))

    def test_rejects_missing_and_oversized_content_length_before_reading(self) -> None:
        async def receive() -> dict[str, object]:
            raise AssertionError("invalid headers must be rejected before reading")

        missing = build_request(receive, content_length=None)
        with self.assertRaises(CameraUploadReadError) as raised:
            parse_camera_upload_headers(missing)
        self.assertEqual(raised.exception.status_code, 411)

        oversized = build_request(receive, content_length=MAX_IMAGE_BYTES + 1)
        with self.assertRaises(CameraUploadReadError) as raised:
            parse_camera_upload_headers(oversized)
        self.assertEqual(raised.exception.status_code, 413)

    def test_strict_headers_must_be_sent_together(self) -> None:
        async def receive() -> dict[str, object]:
            raise AssertionError("invalid headers must be rejected before reading")

        request_id_only = build_request(receive, content_length=10, request_id="paired-1")
        with self.assertRaises(CameraUploadReadError) as raised:
            parse_camera_upload_headers(request_id_only)
        self.assertEqual(raised.exception.reason, "content_sha256_required")

        sha_only = build_request(receive, content_length=10, content_sha256="a" * 64)
        with self.assertRaises(CameraUploadReadError) as raised:
            parse_camera_upload_headers(sha_only)
        self.assertEqual(raised.exception.reason, "request_id_required")

    def test_content_length_mismatch_fails(self) -> None:
        sent = False

        async def receive() -> dict[str, object]:
            nonlocal sent
            if sent:
                raise AssertionError("receive called after completed request")
            sent = True
            return {"type": "http.request", "body": b"abc", "more_body": False}

        request = build_request(receive, content_length=4)
        headers = parse_camera_upload_headers(request)
        with self.assertRaises(CameraUploadReadError) as raised:
            asyncio.run(read_camera_upload_body(request, headers))
        self.assertEqual(raised.exception.reason, "content_length_mismatch")
        self.assertEqual(raised.exception.received_bytes, 3)

    def test_idle_timeout_reports_partial_progress(self) -> None:
        messages = iter([{"type": "http.request", "body": b"abcd", "more_body": True}])

        async def receive() -> dict[str, object]:
            try:
                return next(messages)
            except StopIteration:
                await asyncio.sleep(1)
                raise AssertionError("sleep should be cancelled by idle timeout")

        request = build_request(receive, content_length=10)
        headers = parse_camera_upload_headers(request)
        with patch.dict(os.environ, {"CAMERA_UPLOAD_IDLE_TIMEOUT_SECONDS": "0.01"}):
            with self.assertRaises(CameraUploadReadError) as raised:
                asyncio.run(read_camera_upload_body(request, headers))
        self.assertEqual(raised.exception.status_code, 408)
        self.assertEqual(raised.exception.reason, "idle_timeout")
        self.assertEqual(raised.exception.received_bytes, 4)


class CameraUploadApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.env_patch = patch.dict(
            os.environ,
            {
                "CAMERA_IDEMPOTENCY_DB_PATH": str(self.root / "idempotency.sqlite3"),
                "CAMERA_IDEMPOTENCY_TTL_SECONDS": "1200",
                "CAMERA_IDEMPOTENCY_MAX_RECORDS": "100",
                "CAMERA_IDEMPOTENCY_WAIT_TIMEOUT_SECONDS": "5",
                "CAMERA_IDEMPOTENCY_POLL_INTERVAL_SECONDS": "0.01",
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

    def post(
        self,
        data: bytes,
        *,
        device: str = "camera-api-test",
        request_id: str | None = None,
        content_sha256: str | None = None,
        use_vision: bool = True,
    ):
        headers = {
            "Content-Type": "image/jpeg",
            "Content-Length": str(len(data)),
        }
        if request_id is not None:
            headers["X-Request-ID"] = request_id
        if content_sha256 is not None:
            headers["X-Content-SHA256"] = content_sha256
        return TestClient(app).post(
            "/camera/upload",
            params={"device": device, "use_vision": str(use_vision).lower()},
            content=data,
            headers=headers,
        )

    def test_legacy_upload_without_new_headers_still_succeeds(self) -> None:
        response = self.post(SAMPLE_ONE, use_vision=False)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "ready")
        self.assertEqual(response.json()["device_id"], "camera-api-test")

    def test_request_id_and_correct_sha256_succeed(self) -> None:
        recognize = AsyncMock(return_value=vision_result())
        with (
            patch.object(main, "is_vision_configured", return_value=True),
            patch.object(main, "recognize_artifact_from_image", recognize),
        ):
            response = self.post(
                SAMPLE_ONE,
                request_id="camera-api-test-1",
                content_sha256=sha256(SAMPLE_ONE),
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "ready")
        recognize.assert_awaited_once()

    def test_sha256_mismatch_returns_422_without_save_or_recognition(self) -> None:
        recognize = AsyncMock(return_value=vision_result())
        with (
            patch.object(main, "is_vision_configured", return_value=True),
            patch.object(main, "recognize_artifact_from_image", recognize),
            patch.object(main, "save_camera_image", wraps=main.save_camera_image) as save,
        ):
            response = self.post(
                SAMPLE_ONE,
                request_id="camera-api-test-sha-mismatch",
                content_sha256="0" * 64,
            )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["reason"], "sha256_mismatch")
        save.assert_not_called()
        recognize.assert_not_awaited()

    def test_invalid_jpeg_does_not_start_recognition(self) -> None:
        invalid = b"not-a-jpeg-body"
        recognize = AsyncMock(return_value=vision_result())
        with (
            patch.object(main, "is_vision_configured", return_value=True),
            patch.object(main, "recognize_artifact_from_image", recognize),
            patch.object(main, "save_camera_image", wraps=main.save_camera_image) as save,
        ):
            response = self.post(
                invalid,
                request_id="camera-api-test-invalid",
                content_sha256=sha256(invalid),
            )
        self.assertEqual(response.status_code, 400)
        save.assert_not_called()
        recognize.assert_not_awaited()

    def test_same_request_retry_saves_and_recognizes_once(self) -> None:
        recognize = AsyncMock(return_value=vision_result())
        request_id = "camera-api-test-retry"
        with (
            patch.object(main, "is_vision_configured", return_value=True),
            patch.object(main, "recognize_artifact_from_image", recognize),
            patch.object(main, "save_camera_image", wraps=main.save_camera_image) as save,
        ):
            first = self.post(SAMPLE_ONE, request_id=request_id, content_sha256=sha256(SAMPLE_ONE))
            second = self.post(SAMPLE_ONE, request_id=request_id, content_sha256=sha256(SAMPLE_ONE))
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(save.call_count, 1)
        self.assertEqual(recognize.await_count, 1)

    def test_two_concurrent_identical_requests_recognize_once(self) -> None:
        calls = 0
        lock = threading.Lock()

        async def delayed_recognition(*args, **kwargs):
            nonlocal calls
            with lock:
                calls += 1
            await asyncio.sleep(0.15)
            return vision_result()

        request_id = "camera-api-test-concurrent"

        def send():
            return self.post(
                SAMPLE_ONE,
                request_id=request_id,
                content_sha256=sha256(SAMPLE_ONE),
            )

        with (
            patch.object(main, "is_vision_configured", return_value=True),
            patch.object(main, "recognize_artifact_from_image", side_effect=delayed_recognition),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            responses = list(pool.map(lambda _: send(), range(2)))
        self.assertEqual([response.status_code for response in responses], [200, 200])
        self.assertEqual(responses[0].json(), responses[1].json())
        self.assertEqual(calls, 1)

    def test_same_request_id_with_different_sha256_returns_409(self) -> None:
        recognize = AsyncMock(return_value=vision_result())
        request_id = "camera-api-test-conflict"
        with (
            patch.object(main, "is_vision_configured", return_value=True),
            patch.object(main, "recognize_artifact_from_image", recognize),
        ):
            first = self.post(SAMPLE_ONE, request_id=request_id, content_sha256=sha256(SAMPLE_ONE))
            second = self.post(SAMPLE_TWO, request_id=request_id, content_sha256=sha256(SAMPLE_TWO))
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 409)
        self.assertEqual(recognize.await_count, 1)

    def test_same_request_id_with_different_device_returns_409(self) -> None:
        request_id = "camera-api-test-device-conflict"
        first = self.post(
            SAMPLE_ONE,
            device="camera-api-test-a",
            request_id=request_id,
            content_sha256=sha256(SAMPLE_ONE),
            use_vision=False,
        )
        second = self.post(
            SAMPLE_ONE,
            device="camera-api-test-b",
            request_id=request_id,
            content_sha256=sha256(SAMPLE_ONE),
            use_vision=False,
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 409)

    def test_incomplete_disconnect_does_not_save_recognize_or_create_record(self) -> None:
        full_body = b"0123456789"
        messages = iter(
            [
                {"type": "http.request", "body": full_body[:4], "more_body": True},
                {"type": "http.disconnect"},
            ]
        )

        async def receive() -> dict[str, object]:
            return next(messages)

        request = build_request(
            receive,
            content_length=len(full_body),
            request_id="camera-api-test-disconnect",
            content_sha256=sha256(full_body),
        )
        recognize = AsyncMock(return_value=vision_result())
        with (
            patch.object(main, "recognize_artifact_from_image", recognize),
            patch.object(main, "save_camera_image", wraps=main.save_camera_image) as save,
            self.assertLogs("wkt_ai_server.main", level="WARNING") as logs,
        ):
            response = asyncio.run(main.upload_camera_image(request, device="camera-api-test"))

        self.assertEqual(response.status_code, 400)
        self.assertIn("camera.upload.body_interrupted", "\n".join(logs.output))
        save.assert_not_called()
        recognize.assert_not_awaited()
        self.assertEqual(get_camera_idempotency_store().count(), 0)

    def test_logs_separate_body_receive_and_recognition_stages(self) -> None:
        with self.assertLogs("wkt_ai_server.main", level="INFO") as logs:
            response = self.post(SAMPLE_ONE, use_vision=False)
        self.assertEqual(response.status_code, 200)
        output = "\n".join(logs.output)
        for event in (
            "camera.upload.body_received",
            "camera.upload.validated",
            "camera.recognition.start",
            "camera.recognition.done",
        ):
            self.assertIn(event, output)
        for field in (
            "device=",
            "request_id=",
            "content_length=",
            "bytes_received=",
            "sha256=",
            "stage_ms=",
            "total_ms=",
            "result=",
        ):
            self.assertIn(field, output)


class CameraIdempotencyTests(unittest.TestCase):
    def test_independent_store_instances_share_processing_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "shared.sqlite3"
            first_store = CameraIdempotencyStore(path, ttl_seconds=60, max_records=10)
            second_store = CameraIdempotencyStore(path, ttl_seconds=60, max_records=10)
            identity = CameraRequestIdentity("shared-request", "device", "a" * 64, 100)

            first = first_store.claim(identity)
            second = second_store.claim(identity)

            self.assertEqual(first.action, "owner")
            self.assertEqual(second.action, "wait")

    def test_expired_records_are_cleaned(self) -> None:
        now = [1000.0]
        with tempfile.TemporaryDirectory() as temp_dir:
            store = CameraIdempotencyStore(
                Path(temp_dir) / "ttl.sqlite3",
                ttl_seconds=10,
                max_records=10,
                clock=lambda: now[0],
            )
            identity = CameraRequestIdentity("ttl-request", "device", "b" * 64, 100)
            claim = store.claim(identity)
            self.assertEqual(claim.action, "owner")
            self.assertIsNotNone(claim.owner_token)
            store.complete(
                identity,
                str(claim.owner_token),
                CameraStoredOutcome(200, {"status": "ready"}),
            )
            self.assertEqual(store.count(), 1)

            now[0] += 11
            self.assertEqual(store.cleanup(), 1)
            self.assertEqual(store.count(), 0)

    def test_completed_body_processing_survives_waiter_cancellation(self) -> None:
        async def scenario(store: CameraIdempotencyStore):
            started = asyncio.Event()
            release = asyncio.Event()
            calls = 0

            async def delayed_workflow(*args, **kwargs):
                nonlocal calls
                calls += 1
                started.set()
                await release.wait()
                return CameraStoredOutcome(200, {"status": "ready", "marker": 1})

            identity = CameraRequestIdentity("disconnect-after-body", "device", "c" * 64, 10)
            context = CameraLogContext(
                device_id="device",
                request_id=identity.request_id,
                content_length=10,
                bytes_received=10,
                sha256=identity.sha256,
                total_start=main.perf_counter(),
            )
            with patch.object(main, "safe_execute_camera_workflow", side_effect=delayed_workflow):
                waiter = asyncio.create_task(
                    process_idempotent_camera_upload(
                        store,
                        identity,
                        context,
                        b"0123456789",
                        "image/jpeg",
                        artifact_id="",
                        vision_description=None,
                        use_vision=True,
                    )
                )
                await started.wait()
                waiter.cancel()
                with suppress(asyncio.CancelledError):
                    await waiter

                release.set()
                for _ in range(100):
                    record = await asyncio.to_thread(store.get, identity)
                    if record is not None and record.outcome is not None:
                        break
                    await asyncio.sleep(0.01)
                retry = await process_idempotent_camera_upload(
                    store,
                    identity,
                    context,
                    b"0123456789",
                    "image/jpeg",
                    artifact_id="",
                    vision_description=None,
                    use_vision=True,
                )
            return retry, calls

        with tempfile.TemporaryDirectory() as temp_dir:
            store = CameraIdempotencyStore(
                Path(temp_dir) / "disconnect.sqlite3",
                ttl_seconds=60,
                max_records=10,
            )
            retry, calls = asyncio.run(scenario(store))
        self.assertEqual(retry.payload, {"status": "ready", "marker": 1})
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()
