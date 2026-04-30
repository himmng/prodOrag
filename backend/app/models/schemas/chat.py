from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel


class ChatSessionCreate(BaseModel):
    workspace_id: Optional[UUID] = None
    config_profile_id: Optional[UUID] = None
    title: Optional[str] = None


class ChatSessionRead(BaseModel):
    id: UUID
    workspace_id: UUID
    config_profile_id: Optional[UUID] = None
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ChatMessageCreate(BaseModel):
    content: str


class ChatMessageRead(BaseModel):
    id: UUID
    session_id: UUID
    role: str
    content: str
    token_count: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


class ChatSessionWithMessages(ChatSessionRead):
    messages: List[ChatMessageRead]
