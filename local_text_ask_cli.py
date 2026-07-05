import argparse
import asyncio
import json
import sys
from pathlib import Path

from artifacts import ArtifactNotFoundError, get_artifact
from pipeline import run_text_pipeline
from sessions import set_artifact_context

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False


def configure_stdio() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def print_json(data: dict[str, object]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


async def async_main() -> None:
    configure_stdio()
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run local text -> LLM -> TTS -> ESP32 WAV pipeline."
    )
    parser.add_argument("message", nargs="*", help="User text. If omitted, prompt in terminal.")
    parser.add_argument("--device", default="default", help="Device/session id.")
    parser.add_argument("--artifact-id", help="Optional artifact context id for this test.")
    parser.add_argument("--vision-description", help="Optional latest visual description.")
    parser.add_argument(
        "--output-dir",
        default=str(Path("outputs") / "replies"),
        help="Directory for reply WAV files.",
    )
    parser.add_argument("--no-tts", action="store_true", help="Only generate answer_text.")
    args = parser.parse_args()

    message = " ".join(args.message).strip()
    if not message:
        message = input("You> ").strip()
    if not message:
        raise SystemExit("message cannot be empty")

    if args.artifact_id:
        try:
            get_artifact(args.artifact_id)
        except ArtifactNotFoundError as exc:
            raise SystemExit(f"artifact not found: {args.artifact_id}") from exc
        set_artifact_context(
            device_id=args.device,
            artifact_id=args.artifact_id,
            vision_description=args.vision_description,
            image_id="local-text-test",
        )

    result = await run_text_pipeline(
        device_id=args.device,
        user_text=message,
        output_dir=Path(args.output_dir),
        enable_tts=not args.no_tts,
    )

    print("\nAnswer:")
    print(result.answer_text)
    print("\nResult:")
    print_json(
        {
            "device_id": result.device_id,
            "reply_wav_path": result.reply_wav_path,
            "reply_wav_size": result.reply_wav_size,
            "wav_info": result.wav_info,
            "timings_ms": result.timings_ms,
        }
    )


if __name__ == "__main__":
    asyncio.run(async_main())
