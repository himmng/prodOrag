from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.infrastructure.db import SessionLocal
from app.models.domain.documents import Document, IngestionRun
from app.models.schemas.documents import DocumentCreate, DocumentRead, IngestionRunRead
from app.services.ingestion import IngestionService
from app.services.vector import QdrantVectorStore

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
    document_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    Delete a document, its stored files, and its vector embeddings.
    """
    doc = await session.get(Document, str(document_id))
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete files from ./storage/sessions/{session_id}/{document_id}
    from pathlib import Path
    import shutil

    storage_root = Path(settings.data.storage_root).resolve()
    if doc.session_id:
        doc_dir = storage_root / "sessions" / str(doc.session_id) / str(doc.id)
        if doc_dir.exists():
            shutil.rmtree(doc_dir, ignore_errors=True)

        # Delete embeddings for this doc within the session collection
        vector_store = QdrantVectorStore()
        await vector_store.delete_by_document(str(doc.session_id), str(doc.id))

    await session.delete(doc)
    await session.commit()
    
    return {"status": "deleted", "document_id": document_id}


@router.post("/upload/{session_id}", response_model=List[DocumentRead])
async def upload_documents_for_session(
    session_id: str,
    files: List[UploadFile] = File(...),
    session: AsyncSession = Depends(get_session),
) -> List[DocumentRead]:
    """
    Upload one or more documents for a given chat session.

    For each file:
    - Create a Document row
    - Save original file under ./storage/sessions/{session_id}/{document_id}/{filename}
    - Chunk + embed + store in the per-session Qdrant collection
    """
    ingestion_service = IngestionService()
    uploaded_docs: List[DocumentRead] = []

    # Validate session_id is non-empty
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    for file in files:
        contents = await file.read()
        size_bytes = len(contents)
        doc = Document(
            workspace_id="default",
            session_id=str(session_id),
            filename=file.filename,
            mime_type=file.content_type or "application/octet-stream",
            size_bytes=size_bytes,
        )
        session.add(doc)
        await session.flush()  # ensure doc.id is populated

        # Ingest into vector store (updates status/num_chunks and commits)
        await ingestion_service.ingest_uploaded_document(session, doc, contents)

        uploaded_docs.append(DocumentRead.model_validate(doc))

    # The ingestion service already commits; a final commit here is harmless but ensures consistency
    await session.commit()
    return uploaded_docs


@router.post("/ingest", response_model=IngestionRunRead)
async def trigger_ingestion() -> IngestionRunRead:
    from app.services.ingestion import IngestionService
    service = IngestionService()
    run = await service.ingest_all()
    return IngestionRunRead.model_validate(run)


@router.get("/ingestion-runs", response_model=List[IngestionRunRead])
async def list_ingestion_runs(session: AsyncSession = Depends(get_session)) -> List[IngestionRunRead]:
    result = await session.execute(select(IngestionRun).order_by(IngestionRun.started_at.desc()))
    runs = result.scalars().all()
    return [IngestionRunRead.model_validate(r) for r in runs]
