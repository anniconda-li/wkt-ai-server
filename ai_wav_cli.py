import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_UPLOAD_CHUNK_SIZE = 8192
DEFAULT_RESULT_CHUNK_SIZE = 32768
TERMINAL_STATUSES = {"audio_ready", "audio_failed", "no_speech", "cancelled", "failed"}


def configure_stdio() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def post_json(url: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def post_bytes(url: str, payload: bytes, content_type: str) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def build_url(base_url: str, path: str, query: dict[str, object]) -> str:
    encoded = urllib.parse.urlencode(query)
    return f"{base_url.rstrip('/')}{path}?{encoded}"


def start_session(base_url: str, device: str, language: str) -> tuple[str, int]:
    response = post_json(
        f"{base_url.rstrip('/')}/ai/start",
        {"device": device, "language": language},
    )
    session = str(response["session"])
    chunk_size = int(response.get("chunk_size") or DEFAULT_UPLOAD_CHUNK_SIZE)
    print(f"session: {session}")
    print(f"upload chunk_size: {chunk_size}")
    return session, chunk_size


def upload_wav(base_url: str, session: str, device: str, wav_path: Path, chunk_size: int) -> None:
    total = wav_path.stat().st_size
    with wav_path.open("rb") as wav_file:
        index = 0
        offset = 0
        while True:
            chunk = wav_file.read(chunk_size)
            if not chunk:
                break
            url = build_url(
                base_url,
                "/ai/upload",
                {
                    "session": session,
                    "device": device,
                    "index": index,
                    "offset": offset,
                    "total": total,
                },
            )
            post_bytes(url, chunk, "application/octet-stream")
            offset += len(chunk)
            index += 1

    print(f"uploaded: {total} bytes")


def poll_result(
    base_url: str,
    session: str,
    device: str,
    interval: float,
    timeout: float,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    last_status = ""
    shown_text = False
    while time.monotonic() < deadline:
        info = post_json(
            build_url(base_url, "/ai/result_info", {"session": session, "device": device}),
            {},
        )
        status = str(info.get("status") or "")
        if status != last_status:
            print(f"status: {status}")
            last_status = status

        answer_text = str(info.get("answer_text") or "")
        if answer_text and not shown_text:
            print("\nAnswer:")
            print(answer_text)
            print()
            shown_text = True

        if status in TERMINAL_STATUSES:
            return info

        time.sleep(interval)

    raise TimeoutError(f"result_info timeout after {timeout} seconds")


def download_reply_wav(
    base_url: str,
    session: str,
    device: str,
    total: int,
    output_path: Path,
    chunk_size: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        offset = 0
        while offset < total:
            length = min(chunk_size, total - offset)
            url = build_url(
                base_url,
                "/ai/result_chunk",
                {
                    "session": session,
                    "device": device,
                    "offset": offset,
                    "len": length,
                },
            )
            request = urllib.request.Request(
                url,
                data=b"{}",
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                chunk = response.read()
            if not chunk:
                raise RuntimeError("result_chunk returned empty bytes before EOF")
            output.write(chunk)
            offset += len(chunk)

    return output_path


def print_json(data: dict[str, object]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def main() -> None:
    configure_stdio()
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Simulate ESP32 /ai WAV upload, polling, and reply WAV download."
    )
    parser.add_argument("wav_path", help="Request WAV path, must be 16k/16-bit/mono PCM.")
    parser.add_argument("--device", default="walkie-01", help="Device id.")
    parser.add_argument("--language", default="zh", help="Language code.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Backend base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument("--poll-interval", type=float, default=1.0, help="result_info poll interval seconds.")
    parser.add_argument("--timeout", type=float, default=300.0, help="result_info timeout seconds.")
    parser.add_argument("--no-download", action="store_true", help="Do not download reply WAV.")
    parser.add_argument(
        "--output-dir",
        default=str(Path("outputs") / "device_protocol"),
        help="Reply WAV output directory.",
    )
    args = parser.parse_args()

    wav_path = Path(args.wav_path)
    if not wav_path.exists():
        raise SystemExit(f"WAV not found: {wav_path}")

    try:
        session, chunk_size = start_session(args.base_url, args.device, args.language)
        upload_wav(args.base_url, session, args.device, wav_path, chunk_size)
        finish_info = post_json(
            build_url(args.base_url, "/ai/finish", {"session": session, "device": args.device}),
            {},
        )
        print("finish:")
        print_json(finish_info)

        info = poll_result(args.base_url, session, args.device, args.poll_interval, args.timeout)
        print("final:")
        print_json(info)

        reply_size = int(info.get("reply_wav_size") or 0)
        if not args.no_download and reply_size > 0:
            output_path = Path(args.output_dir) / args.device / f"{session}_reply.wav"
            download_reply_wav(
                args.base_url,
                session,
                args.device,
                reply_size,
                output_path,
                DEFAULT_RESULT_CHUNK_SIZE,
            )
            print(f"reply_wav_path: {output_path}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"connect failed: {exc.reason}") from exc


if __name__ == "__main__":
    main()
