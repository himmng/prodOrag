from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db import SessionLocal
from app.models.domain.sessions import ChatMessage, ChatSession
from app.models.domain.documents import Document
from app.models.schemas.chat import (
    ChatMessageCreate,
    ChatMessageRead,
    ChatSessionCreate,
    ChatSessionRead,
    ChatSessionWithMessages,
)

router = APIRouter(prefix="/chat", tags=["chat"])


async def get_session() -> AsyncSession:
    async with SessionLocal() as session:
        yield session


@router.post("/sessions", response_model=ChatSessionRead)
async def create_session(
    payload: ChatSessionCreate,
    session: AsyncSession = Depends(get_session),
) -> ChatSessionRead:
    title = payload.title or "New chat"
    chat_session = ChatSession(
        workspace_id=str(payload.workspace_id) if payload.workspace_id else "default",
        config_profile_id=str(payload.config_profile_id) if payload.config_profile_id else None,
        title=title,
    )
    session.add(chat_session)
    await session.commit()
    await session.refresh(chat_session)
    return ChatSessionRead.model_validate(chat_session)


@router.get("/sessions/{session_id}", response_model=ChatSessionWithMessages)
async def get_session_with_messages(
    session_id: str,
    session: AsyncSession = Depends(get_session),
) -> ChatSessionWithMessages:
    db_session = await session.get(ChatSession, session_id)
    if db_session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    result = await session.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at)
    )
    messages = result.scalars().all()
    return ChatSessionWithMessages(
        id=db_session.id,
        workspace_id=db_session.workspace_id,
        config_profile_id=db_session.config_profile_id,
        title=db_session.title,
        created_at=db_session.created_at,
        updated_at=db_session.updated_at,
        messages=[ChatMessageRead.model_validate(m) for m in messages],
    )


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageRead)
async def post_message(
    session_id: str,
    payload: ChatMessageCreate,
    session: AsyncSession = Depends(get_session),
) -> ChatMessageRead:
    db_session = await session.get(ChatSession, session_id)
    if db_session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    message = ChatMessage(
        session_id=session_id,
        role="user",
        content=payload.content,
        token_count=None,
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return ChatMessageRead.model_validate(message)


@router.get("/sessions/{session_id}/export-summary")
async def export_session_summary(
    session_id: str,
    session: AsyncSession = Depends(get_session),
) -> JSONResponse:
    db_session = await session.get(ChatSession, session_id)
    if db_session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")
    result = await session.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at)
    )
    messages = result.scalars().all()
    summary_text = "\n".join(f"[{m.role}] {m.content}" for m in messages)
    payload = {
        "session_id": session_id,
        "title": db_session.title,
        "summary": summary_text,
    }
    return JSONResponse(content=payload)


@router.get("/sessions/{session_id}/export-irag")
async def export_session_irag(
    session_id: str,
    session: AsyncSession = Depends(get_session),
) -> JSONResponse:
    result = await session.execute(select(Document).where(Document.session_id == session_id))
    docs = result.scalars().all()
    payload = {
        "session_id": session_id,
        "documents": [
            {
                "id": d.id,
                "filename": d.filename,
                "mime_type": d.mime_type,
                "size_bytes": d.size_bytes,
                "status": d.status,
                "num_chunks": d.num_chunks,
            }
            for d in docs
        ],
    }
    return JSONResponse(content=payload)
