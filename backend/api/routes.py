from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from backend.core.config import get_config, AppConfig
from backend.core.orchestrator import get_orchestrator
from backend.core.document_store import save_document_and_index

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, config: AppConfig = Depends(get_config)):
    orchestrator = get_orchestrator(config)
    result = await orchestrator.chat(payload.message, [m.dict() for m in payload.history])
    return ChatResponse(answer=result["answer"], sources=result["sources"])


@router.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    config: AppConfig = Depends(get_config),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    await save_document_and_index(file, config)
    return {"status": "ok"}


@router.get("/config")
async def get_config_endpoint(config: AppConfig = Depends(get_config)):
    return config


class ConfigUpdate(BaseModel):
    llm: Optional[Dict[str, Any]] = None
    embeddings: Optional[Dict[str, Any]] = None
    storage: Optional[Dict[str, Any]] = None
    rag: Optional[Dict[str, Any]] = None


@router.post("/config")
async def update_config_endpoint(update: ConfigUpdate):
    from backend.core.config import load_config, save_config

    config = load_config()
    data = config.dict()

    for section in ["llm", "embeddings", "storage", "rag"]:
        section_update = getattr(update, section)
        if section_update:
            data[section].update(section_update)

    new_config = AppConfig(**data)
    save_config(new_config)
    return new_config


@router.get("/health")
async def health():
    return {"status": "ok"}
