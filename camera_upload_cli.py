import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def build_url(base_url: str, device: str, artifact_id: str | None, vision_description: str | None) -> str:
    query = {"device": device}
    if artifact_id:
        query["artifact_id"] = artifact_id
    if vision_description:
        query["vision_description"] = vision_description
    return f"{base_url.rstrip('/')}/camera/upload?{urlencode(query)}"


def upload_image(
    image_path: Path,
    base_url: str,
    device: str,
    artifact_id: str | None,
    vision_description: str | None,
) -> dict[str, object]:
    image_bytes = image_path.read_bytes()
    request = Request(
        build_url(base_url, device, artifact_id, vision_description),
        data=image_bytes,
        headers={"Content-Type": "image/jpeg"},
        method="POST",
    )

    with urlopen(request, timeout=60) as response:
        response_text = response.read().decode("utf-8")

    return json.loads(response_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a JPEG image to /camera/upload.")
    parser.add_argument("image", type=Path, help="Path to a JPEG image.")
    parser.add_argument("--device", default="walkie-01", help="Device id.")
    parser.add_argument("--artifact-id", help="Manual artifact id for simulated recognition.")
    parser.add_argument("--vision-description", help="Optional visual description.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Backend base URL.")
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"image file not found: {args.image}")

    result = upload_image(
        image_path=args.image,
        base_url=args.url,
        device=args.device,
        artifact_id=args.artifact_id,
        vision_description=args.vision_description,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
