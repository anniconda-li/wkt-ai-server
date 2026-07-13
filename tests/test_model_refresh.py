import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import asr
import tts
import vision_llm
from wav_utils import write_silence_wav


class ModelDefaultsTests(unittest.TestCase):
    def test_refreshed_model_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(asr.get_asr_model(), "qwen3-asr-flash-2026-02-10")
            self.assertEqual(asr.get_asr_fallback_model(), "paraformer-realtime-v2")
            self.assertEqual(vision_llm.get_vision_model(), "qwen3.6-flash-2026-04-16")
            self.assertFalse(vision_llm.is_thinking_enabled())
            self.assertEqual(tts.DEFAULT_TTS_MODEL, "qwen3-tts-flash-2025-11-27")

    def test_tts_request_defaults_to_fixed_snapshot_cherry_and_wav(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"wav-bytes"
        with (
            patch("tts.urllib.request.urlopen", return_value=response) as urlopen,
            patch.dict(
                os.environ,
                {
                    "TTS_API_KEY": "test-key",
                    "TTS_BASE_URL": "https://tts.example.test",
                },
                clear=True,
            ),
        ):
            self.assertEqual(tts.request_openai_compatible_tts("测试"), b"wav-bytes")

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "qwen3-tts-flash-2025-11-27")
        self.assertEqual(payload["voice"], "Cherry")
        self.assertEqual(payload["response_format"], "wav")


class QwenAsrTests(unittest.TestCase):
    def test_qwen_asr_sends_complete_wav_as_data_uri(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="这是应国玉鹰。"))],
            model_dump=lambda mode: {"id": "test-response", "mode": mode},
        )
        create = MagicMock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "request.wav"
            write_silence_wav(wav_path, duration_seconds=0.1)
            with (
                patch("asr.OpenAI", return_value=client) as openai_client,
                patch.dict(
                    os.environ,
                    {
                        "DASHSCOPE_API_KEY": "test-key",
                        "ASR_MODEL": "qwen3-asr-flash-2026-02-10",
                        "ASR_LANGUAGE": "zh",
                    },
                    clear=True,
                ),
            ):
                result = asr.transcribe_qwen_flash(wav_path)

        self.assertEqual(result.text, "这是应国玉鹰。")
        self.assertEqual(result.model, "qwen3-asr-flash-2026-02-10")
        self.assertEqual(
            openai_client.call_args.kwargs["base_url"],
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        request = create.call_args.kwargs
        audio_data = request["messages"][0]["content"][0]["input_audio"]["data"]
        self.assertTrue(audio_data.startswith("data:audio/wav;base64,UklGR"))
        self.assertEqual(request["extra_body"]["asr_options"]["language"], "zh")
        self.assertFalse(request["extra_body"]["asr_options"]["enable_itn"])

    def test_qwen_asr_accepts_ogg_data_uri(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="这是什么文物？"))],
            model_dump=lambda mode: {"id": "test-ogg-response", "mode": mode},
        )
        create = MagicMock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        with tempfile.TemporaryDirectory() as temp_dir:
            ogg_path = Path(temp_dir) / "request.ogg"
            ogg_path.write_bytes(b"OggS-test-data")
            with (
                patch("asr.OpenAI", return_value=client),
                patch.dict(
                    os.environ,
                    {
                        "DASHSCOPE_API_KEY": "test-key",
                        "ASR_MODEL": "qwen3-asr-flash-2026-02-10",
                    },
                    clear=True,
                ),
            ):
                result = asr.transcribe_qwen_flash(ogg_path)

        self.assertEqual(result.text, "这是什么文物？")
        request = create.call_args.kwargs
        audio_data = request["messages"][0]["content"][0]["input_audio"]["data"]
        self.assertTrue(audio_data.startswith("data:audio/ogg;base64,"))

    def test_qwen_failure_falls_back_to_paraformer(self) -> None:
        fallback_result = asr.AsrResult(
            text="回退识别成功",
            provider="dashscope",
            model="paraformer-realtime-v2",
        )
        with (
            patch.dict(
                os.environ,
                {
                    "ASR_PROVIDER": "dashscope",
                    "ASR_MODEL": "qwen3-asr-flash-2026-02-10",
                    "ASR_FALLBACK_MODEL": "paraformer-realtime-v2",
                },
                clear=True,
            ),
            patch("asr.transcribe_qwen_flash", side_effect=asr.AsrError("primary failed")),
            patch("asr.transcribe_dashscope_realtime", return_value=fallback_result) as fallback,
        ):
            result = asr.transcribe_device_wav(Path("unused.wav"))

        self.assertEqual(result.text, "回退识别成功")
        self.assertEqual(fallback.call_args.kwargs["model"], "paraformer-realtime-v2")


class VisionRequestTests(unittest.TestCase):
    def test_vision_disables_thinking_and_requests_json(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"artifact_id":"yingguo_jade_eagle","confidence":0.92,'
                            '"evidence":["鹰形轮廓"],"vision_description":"玉鹰"}'
                        )
                    )
                )
            ]
        )
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

        with (
            patch("vision_llm.get_vision_client", return_value=client),
            patch.dict(
                os.environ,
                {
                    "VISION_MODEL": "qwen3.6-flash-2026-04-16",
                    "VISION_ENABLE_THINKING": "false",
                },
                clear=True,
            ),
        ):
            result = asyncio.run(
                vision_llm.recognize_artifact_from_image(b"jpeg-bytes", "image/jpeg")
            )

        self.assertEqual(result["artifact_id"], "yingguo_jade_eagle")
        request = create.await_args.kwargs
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["extra_body"], {"enable_thinking": False})


if __name__ == "__main__":
    unittest.main()
