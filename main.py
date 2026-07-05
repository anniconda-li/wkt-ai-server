from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from artifacts import ArtifactNotFoundError, get_artifact, list_artifacts
from llm import validate_llm_config
from router import chat_stream
from sessions import clear_session, list_session_summaries, normalize_device_id, session_snapshot


app = FastAPI(title="Minimal AI Chat Backend")


class ChatRequest(BaseModel):
    message: str
    device: str = "default"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sessions")
async def sessions() -> list[dict[str, object]]:
    return list_session_summaries()


@app.get("/artifacts")
async def artifacts() -> list[dict[str, object]]:
    return list_artifacts()


@app.get("/artifacts/{artifact_id}")
async def artifact_detail(artifact_id: str) -> dict[str, object]:
    try:
        return get_artifact(artifact_id)
    except ArtifactNotFoundError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc


@app.get("/sessions/{device_id}")
async def get_device_session(device_id: str) -> dict[str, object]:
    return session_snapshot(device_id)


@app.post("/sessions/{device_id}/clear")
async def clear_device_session(device_id: str) -> dict[str, str]:
    clear_session(device_id)
    return {"status": "cleared", "device_id": normalize_device_id(device_id)}


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    try:
        validate_llm_config()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return StreamingResponse(
        chat_stream(user_message, normalize_device_id(request.device)),
        media_type="text/plain; charset=utf-8",
    )
