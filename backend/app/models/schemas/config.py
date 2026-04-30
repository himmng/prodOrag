from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ConfigProfileBase(BaseModel):
    name: str
    is_default: bool = False
    llm_provider: str
    llm_base_url: str
    llm_model_id: str
    llm_api_key: Optional[str] = None
    embedding_provider: str
    embedding_base_url: str
    embedding_model_name: str
    embedding_api_key: Optional[str] = None
    retrieval_top_k: int = Field(default=8, ge=1)
    retrieval_score_threshold: Optional[float] = None
    hybrid_search_enabled: bool = False
    reranker_model: Optional[str] = None


class ConfigProfileCreate(ConfigProfileBase):
    pass


class ConfigProfileUpdate(ConfigProfileBase):
    pass


class ConfigProfileRead(ConfigProfileBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
