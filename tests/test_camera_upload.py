import asyncio
import os
import unittest
from collections.abc import Awaitable, Callable
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.requests import Request

from main import CameraUploadReadError, app, read_camera_upload_body
from vision import MAX_IMAGE_BYTES


def build_request(
    receive: Callable[[], Awaitable[dict[str, object]]],
    *,
    content_length: int | None = None,
) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode("ascii")))
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
        body = asyncio.run(read_camera_upload_body(request, "camera-test"))
        self.assertEqual(body, b"abcdef")

    def test_rejects_content_length_mismatch(self) -> None:
        sent = False

        async def receive() -> dict[str, object]:
            nonlocal sent
            if sent:
                raise AssertionError("receive called after completed request")
            sent = True
            return {"type": "http.request", "body": b"abc", "more_body": False}

        request = build_request(receive, content_length=4)
        with self.assertRaises(CameraUploadReadError) as raised:
            asyncio.run(read_camera_upload_body(request, "camera-test"))

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.reason, "content_length_mismatch")
        self.assertEqual(raised.exception.received_bytes, 3)
        self.assertEqual(raised.exception.expected_bytes, 4)

    def test_rejects_oversized_declared_body_before_reading(self) -> None:
        async def receive() -> dict[str, object]:
            raise AssertionError("oversized body should be rejected before reading")

        request = build_request(receive, content_length=MAX_IMAGE_BYTES + 1)
        with self.assertRaises(CameraUploadReadError) as raised:
            asyncio.run(read_camera_upload_body(request, "camera-test"))

        self.assertEqual(raised.exception.status_code, 413)
        self.assertEqual(raised.exception.reason, "content_length_too_large")

    def test_api_returns_structured_error_and_closes_partial_connection(self) -> None:
        response = TestClient(app).post(
            "/camera/upload",
            params={"device": "camera-test", "use_vision": "false"},
            content=b"small",
            headers={
                "Content-Type": "image/jpeg",
                "Content-Length": str(MAX_IMAGE_BYTES + 1),
            },
        )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.headers.get("connection"), "close")
        detail = response.json()["detail"]
        self.assertEqual(detail["reason"], "content_length_too_large")
        self.assertEqual(detail["received_bytes"], 0)
        self.assertEqual(detail["expected_bytes"], MAX_IMAGE_BYTES + 1)

    def test_idle_timeout_reports_partial_progress(self) -> None:
        messages = iter(
            [{"type": "http.request", "body": b"abcd", "more_body": True}]
        )

        async def receive() -> dict[str, object]:
            try:
                return next(messages)
            except StopIteration:
                await asyncio.sleep(1)
                raise AssertionError("sleep should be cancelled by timeout")

        request = build_request(receive, content_length=10)
        with patch.dict(
            os.environ,
            {
                "CAMERA_UPLOAD_IDLE_TIMEOUT_SECONDS": "0.01",
                "CAMERA_UPLOAD_TOTAL_TIMEOUT_SECONDS": "0.2",
            },
        ):
            with self.assertRaises(CameraUploadReadError) as raised:
                asyncio.run(read_camera_upload_body(request, "camera-test"))

        self.assertEqual(raised.exception.status_code, 408)
        self.assertEqual(raised.exception.reason, "idle_timeout")
        self.assertEqual(raised.exception.received_bytes, 4)
        self.assertEqual(raised.exception.expected_bytes, 10)

    def test_client_disconnect_reports_partial_progress(self) -> None:
        messages = iter(
            [
                {"type": "http.request", "body": b"abcd", "more_body": True},
                {"type": "http.disconnect"},
            ]
        )

        async def receive() -> dict[str, object]:
            return next(messages)

        request = build_request(receive, content_length=10)
        with self.assertRaises(CameraUploadReadError) as raised:
            asyncio.run(read_camera_upload_body(request, "camera-test"))

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.reason, "client_disconnected")
        self.assertEqual(raised.exception.received_bytes, 4)
        self.assertEqual(raised.exception.expected_bytes, 10)


if __name__ == "__main__":
    unittest.main()
