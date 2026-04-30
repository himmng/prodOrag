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
    ChatQueryRequest,
    ChatQueryResponse,
    RetrievedChunk,
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


@router.get("/sessions", response_model=List[ChatSessionRead])
async def list_sessions(
    session: AsyncSession = Depends(get_session),
) -> List[ChatSessionRead]:
    result = await session.execute(select(ChatSession).order_by(ChatSession.created_at.desc()))
    sessions = result.scalars().all()
    return [ChatSessionRead.model_validate(s) for s in sessions]


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


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    Delete a session and all its associated data:
    - Chat messages
    - Documents and their files
    - Vector embeddings in Qdrant
    """
    db_session = await session.get(ChatSession, session_id)
    if db_session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Delete all documents and their embeddings
    result = await session.execute(
        select(Document).where(Document.session_id == session_id)
    )
    documents = result.scalars().all()
    
    for doc in documents:
        # Delete file from disk
        from pathlib import Path
        import shutil
        from app.config.settings import settings
        
        storage_root = Path(settings.data.storage_root).resolve()
        doc_dir = storage_root / "sessions" / str(session_id) / str(doc.id)
        if doc_dir.exists():
            shutil.rmtree(doc_dir, ignore_errors=True)
        
        # Delete embeddings from Qdrant
        from app.services.vector import QdrantVectorStore
        vector_store = QdrantVectorStore()
        await vector_store.delete_by_document(str(session_id), str(doc.id))
        
        await session.delete(doc)
    
    # Delete all messages in the session
    result = await session.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id)
    )
    messages = result.scalars().all()
    for msg in messages:
        await session.delete(msg)
    
    # Delete the session itself
    await session.delete(db_session)
    await session.commit()
    
    # Drop the session's Qdrant collection
    from app.services.vector import QdrantVectorStore
    vector_store = QdrantVectorStore()
    await vector_store.drop_session(str(session_id))
    
    return {"status": "deleted", "session_id": session_id}


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


@router.post("/query", response_model=ChatQueryResponse)
async def query_chat(
    payload: ChatQueryRequest,
    session: AsyncSession = Depends(get_session),
) -> ChatQueryResponse:
    """
    Chat with RAG using a per-session vector store.

    - Stores user message
    - Retrieves recent conversation history
    - Embeds the new user message
    - Searches only the current session's vector collection
    - Calls the local LLM (Ollama/OpenAI-compatible) with context
    - Stores assistant reply
    """
    from app.services.vector import QdrantVectorStore

    session_id_str = str(payload.session_id)

    # Ensure session exists
    db_session = await session.get(ChatSession, session_id_str)
    if db_session is None:
        raise HTTPException(status_code=404, detail="Chat session not found")

    # Store user message
    user_message = ChatMessage(
        session_id=session_id_str,
        role="user",
        content=payload.message,
    )
    session.add(user_message)
    await session.commit()
    await session.refresh(user_message)

    # Get conversation history (last 10 messages)
    result = await session.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id_str)
        .order_by(ChatMessage.created_at.desc())
        .limit(10)
    )
    recent_messages = result.scalars().all()[::-1]  # reverse to chronological

    # Embed user message
    from app.services.llm import get_embedding_client
    embedding_client = get_embedding_client()
    embeddings = await embedding_client.embed([payload.message])
    query_embedding = embeddings[0]

    # Search vector store for this specific session
    vector_store = QdrantVectorStore()
    # Ensure the per-session collection exists (no-op if empty)
    await vector_store.ensure_collection(session_id_str)
    search_results = await vector_store.search(session_id_str, query_embedding, top_k=5)

    # Build context from retrieved chunks
    context = "\n\n".join([hit["content"] for hit in search_results]) if search_results else ""

    # Build prompt
    system_prompt = "You are a helpful assistant."
    if context:
        system_prompt += (
            "\nUse the following context to answer the user's question. "
            "If the context is not relevant, fall back to your general knowledge.\n\n"
            f"{context}"
        )

    system_message = {
        "role": "system",
        "content": system_prompt,
    }
    conversation = [system_message] + [
        {"role": msg.role, "content": msg.content} for msg in recent_messages
    ]

    # Call LLM using factory function
    from app.services.llm import get_chat_client
    chat_client = get_chat_client()
    assistant_response = await chat_client.chat(conversation)

    # Store assistant message
    assistant_message = ChatMessage(
        session_id=session_id_str,
        role="assistant",
        content=assistant_response,
    )
    session.add(assistant_message)
    await session.commit()
    await session.refresh(assistant_message)

    # Format response
    retrieved_chunks = [
        RetrievedChunk(content=hit["content"], metadata=hit["metadata"], score=hit["score"])
        for hit in search_results
    ]

    return ChatQueryResponse(
        assistant_message=ChatMessageRead.model_validate(assistant_message),
        retrieved_chunks=retrieved_chunks,
    )
