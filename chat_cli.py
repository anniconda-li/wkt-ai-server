import argparse
import codecs
import json
import sys
import urllib.error
import urllib.request


DEFAULT_URL = "http://127.0.0.1:8000/chat"


def configure_stdio() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def stream_chat(message: str, url: str) -> None:
    body = json.dumps({"message": message}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    decoder = codecs.getincrementaldecoder("utf-8")("replace")

    with urllib.request.urlopen(request, timeout=120) as response:
        while True:
            chunk = response.read(64)
            if not chunk:
                break

            text = decoder.decode(chunk)
            if text:
                print(text, end="", flush=True)

        rest = decoder.decode(b"", final=True)
        if rest:
            print(rest, end="", flush=True)

    print()


def run_repl(url: str) -> None:
    print("Terminal chat client. Type /exit to quit.")
    while True:
        try:
            message = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not message:
            continue
        if message in {"/exit", "/quit"}:
            return

        print("AI> ", end="", flush=True)
        try:
            stream_chat(message, url)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"\n[HTTP_ERROR] {exc.code}: {detail}")
        except urllib.error.URLError as exc:
            print(f"\n[CONNECT_ERROR] {exc.reason}")
        except TimeoutError:
            print("\n[TIMEOUT] Backend did not respond in time.")


def main() -> None:
    configure_stdio()

    parser = argparse.ArgumentParser(description="Simple terminal chat client.")
    parser.add_argument("message", nargs="*", help="Send one message and exit.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Chat endpoint. Default: {DEFAULT_URL}")
    args = parser.parse_args()

    if args.message:
        stream_chat(" ".join(args.message), args.url)
        return

    run_repl(args.url)


if __name__ == "__main__":
    main()
