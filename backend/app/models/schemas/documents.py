from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class DocumentBase(BaseModel):
    workspace_id: UUID
    filename: str
    mime_type: str
    size_bytes: int
    session_id: UUID | None = None


class DocumentCreate(DocumentBase):
    pass


class DocumentRead(DocumentBase):
    id: UUID
    status: str
    num_chunks: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
