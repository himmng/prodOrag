from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..config.store import load_config
from ..llm.client import OpenAICompatibleClient


router = APIRouter()


@router.post("/stream")
async def chat_stream(payload: dict) -> StreamingResponse:
    messages = payload.get("messages", [])
    config = load_config()
    client = OpenAICompatibleClient(config)

    def token_stream():
        for token in client.chat_completion_stream(messages):
            yield token

    return StreamingResponse(token_stream(), media_type="text/plain")
