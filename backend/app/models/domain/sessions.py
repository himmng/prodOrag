import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String

from app.infrastructure.db import Base


def _uuid_str() -> str:
    return str(uuid.uuid4())


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String, primary_key=True, default=_uuid_str)
    workspace_id = Column(String, nullable=False)
    config_profile_id = Column(String, nullable=True)
    title = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True, default=_uuid_str)
    session_id = Column(String, nullable=False)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    token_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
