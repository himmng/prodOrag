from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db import SessionLocal
from app.models.domain.documents import Document
from app.models.schemas.documents import DocumentRead
from app.models.schemas.documents import DocumentCreate, DocumentRead

router = APIRouter(prefix="/documents", tags=["documents"])


async def get_session() -> AsyncSession:
    async with SessionLocal() as session:
        yield session


@router.get("", response_model=List[DocumentRead])
async def list_documents(session: AsyncSession = Depends(get_session)) -> List[DocumentRead]:
    result = await session.execute(select(Document))
    docs = result.scalars().all()
    return [DocumentRead.model_validate(d) for d in docs]


@router.get("/by-session/{session_id}", response_model=List[DocumentRead])
async def list_documents_by_session(
    session_id: str, session: AsyncSession = Depends(get_session)
) -> List[DocumentRead]:
    result = await session.execute(select(Document).where(Document.session_id == session_id))
    docs = result.scalars().all()
    return [DocumentRead.model_validate(d) for d in docs]


@router.post("", response_model=DocumentRead)
async def create_document(
    payload: DocumentCreate,
    session: AsyncSession = Depends(get_session),
) -> DocumentRead:
    doc = Document(
        workspace_id=str(payload.workspace_id),
        session_id=str(payload.session_id) if payload.session_id else None,
        filename=payload.filename,
        mime_type=payload.mime_type,
        size_bytes=payload.size_bytes,
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)
    return DocumentRead.model_validate(doc)


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(
    document_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> DocumentRead:
    doc = await session.get(Document, str(document_id))
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentRead.model_validate(doc)


@router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> None:
    doc = await session.get(Document, str(document_id))
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    await session.delete(doc)
    await session.commit()


@router.post("/upload/{session_id}", response_model=List[DocumentRead])
async def upload_documents_for_session(
    session_id: str,
    files: List[UploadFile] = File(...),
    session: AsyncSession = Depends(get_session),
) -> List[DocumentRead]:
    uploaded_docs: List[DocumentRead] = []
    for file in files:
        contents = await file.read()
        size_bytes = len(contents)
        doc = Document(
            workspace_id="default",
            session_id=session_id,
            filename=file.filename,
            mime_type=file.content_type or "application/octet-stream",
            size_bytes=size_bytes,
        )
        session.add(doc)
        await session.flush()
        uploaded_docs.append(DocumentRead.model_validate(doc))
    await session.commit()
    return uploaded_docs
