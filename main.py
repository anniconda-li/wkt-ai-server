from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm import validate_llm_config
from router import chat_stream


app = FastAPI(title="Minimal AI Chat Backend")


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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
        chat_stream(user_message),
        media_type="text/plain; charset=utf-8",
    )
