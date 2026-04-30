import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String

from app.infrastructure.db import Base


def _uuid_str() -> str:
    return str(uuid.uuid4())


class ConfigProfile(Base):
    __tablename__ = "config_profiles"

    id = Column(String, primary_key=True, default=_uuid_str)
    name = Column(String, nullable=False, unique=True)
    is_default = Column(Boolean, default=False, nullable=False)
    llm_provider = Column(String, nullable=False)
    llm_base_url = Column(String, nullable=False)
    llm_model_id = Column(String, nullable=False)
    llm_api_key = Column(String, nullable=True)
    embedding_provider = Column(String, nullable=False)
    embedding_base_url = Column(String, nullable=False)
    embedding_model_name = Column(String, nullable=False)
    embedding_api_key = Column(String, nullable=True)
    retrieval_top_k = Column(Integer, nullable=False, default=8)
    retrieval_score_threshold = Column(Float, nullable=True)
    hybrid_search_enabled = Column(Boolean, default=False, nullable=False)
    reranker_model = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
