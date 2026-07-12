import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

os.environ["AI_MOCK_ASR_TEXT"] = "这是什么"
os.environ["AI_MOCK_LLM_TEXT"] = "这是离线测试回答。"
os.environ["AI_ENABLE_MOCK_TTS"] = "true"
os.environ["AI_MOCK_TTS_SECONDS"] = "0.1"
os.environ["AI_SILENCE_RMS_THRESHOLD"] = "0"
os.environ["AI_MIN_SPEECH_SECONDS"] = "0"

import ai_protocol
import main
import vision
from wav_utils import validate_device_wav, write_silence_wav


class ServiceIdentitySmokeTests(unittest.TestCase):
    def test_fastapi_identity_and_existing_routes(self) -> None:
        self.assertEqual(main.app.title, "wkt-ai-server")
        routes = {route.path for route in main.app.routes}
        expected = {
            "/health",
            "/chat",
            "/sessions",
            "/sessions/{device_id}",
            "/sessions/{device_id}/clear",
            "/sessions/{device_id}/artifact-context",
            "/artifacts",
            "/artifacts/{artifact_id}",
            "/camera/upload",
            "/ai/start",
            "/ai/upload",
            "/ai/finish",
            "/ai/result_info",
            "/ai/result_chunk",
            "/ai/cancel",
            "/ai/stop_audio",
        }
        self.assertTrue(expected.issubset(routes))

    def test_health_and_camera_upload_interfaces(self) -> None:
        sample = Path("samples/camera/yingguo_jade_eagle_esp32.jpg").read_bytes()
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(vision, "UPLOADS_DIR", Path(temp_dir)):
                client = TestClient(main.app)
                health = client.get("/health")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json(), {"status": "ok"})

                response = client.post(
                    "/camera/upload",
                    params={
                        "device": "smoke-camera",
                        "artifact_id": "yingguo_jade_eagle",
                        "use_vision": "false",
                    },
                    content=sample,
                    headers={"Content-Type": "image/jpeg"},
                )
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()
                self.assertEqual(payload["status"], "ready")
                self.assertEqual(payload["latest_artifact_id"], "yingguo_jade_eagle")


class AiAudioPipelineSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        ai_protocol.ai_sessions.clear()

    def test_mock_asr_llm_tts_and_wav_chunk_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(ai_protocol, "AI_ROOT", Path(temp_dir) / "uploads" / "ai"):
                source = Path(temp_dir) / "request.wav"
                write_silence_wav(source, duration_seconds=0.25)
                request_bytes = source.read_bytes()

                session = ai_protocol.create_ai_session("smoke-audio", "zh")
                result = ai_protocol.write_ai_upload_chunk(
                    session.session_id,
                    request_bytes,
                    index=0,
                    offset=0,
                    total=len(request_bytes),
                    device="smoke-audio",
                )
                self.assertEqual(result, {"ok": True})

                asyncio.run(ai_protocol.process_ai_session(session.session_id))

                self.assertEqual(session.status, "audio_ready")
                self.assertEqual(session.asr_text, "这是什么")
                self.assertEqual(session.answer_text, "这是离线测试回答。")
                self.assertEqual(session.tts_status, "done")
                self.assertGreater(session.reply_wav_size, 44)
                self.assertEqual(
                    ai_protocol.read_ai_result_chunk(
                        session.session_id,
                        offset=0,
                        length=min(256, session.reply_wav_size),
                        device="smoke-audio",
                    )[:4],
                    b"RIFF",
                )
                self.assertEqual(validate_device_wav(session.reply_wav_path)["sample_rate"], 16000)


if __name__ == "__main__":
    unittest.main()
