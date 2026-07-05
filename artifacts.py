import json
from functools import lru_cache
from pathlib import Path
from typing import Any


ARTIFACTS_DIR = Path(__file__).parent / "data" / "artifacts"


class ArtifactNotFoundError(KeyError):
    pass


@lru_cache(maxsize=1)
def load_artifacts() -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}

    for path in sorted(ARTIFACTS_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            artifact = json.load(file)

        artifact_id = artifact.get("id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError(f"Artifact file {path} is missing a valid id")
        if artifact_id in artifacts:
            raise ValueError(f"Duplicate artifact id: {artifact_id}")

        artifacts[artifact_id] = artifact

    return artifacts


def list_artifacts() -> list[dict[str, Any]]:
    return [
        {
            "id": artifact["id"],
            "name": artifact["name"],
            "category": artifact.get("category"),
            "period": artifact.get("period"),
            "data_status": artifact.get("data_status"),
        }
        for artifact in load_artifacts().values()
    ]


def get_artifact(artifact_id: str) -> dict[str, Any]:
    try:
        return load_artifacts()[artifact_id]
    except KeyError as exc:
        raise ArtifactNotFoundError(artifact_id) from exc


def find_artifacts_by_text(text: str) -> list[dict[str, Any]]:
    normalized = text.lower()
    matches: list[dict[str, Any]] = []

    for artifact in load_artifacts().values():
        terms = [artifact["name"], *artifact.get("aliases", [])]
        if any(term and term.lower() in normalized for term in terms):
            matches.append(artifact)

    return matches
