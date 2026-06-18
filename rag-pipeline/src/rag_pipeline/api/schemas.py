"""API request/response schemas.

Vendor-neutral — no LangChain types leak out of the API. Clients see
plain JSON dicts that match these Pydantic models exactly.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    retriever: Literal["hybrid_reranked", "dense", "bm25", "ensemble"] = "hybrid_reranked"


class Citation(BaseModel):
    n:             int
    source_path:   str
    page_number:   Optional[int]  = None
    section_title: Optional[str]  = None
    score:         float


class AnswerResponse(BaseModel):
    question:    str
    answer:      str
    citations:   list[Citation]
    retriever:   str
    latency_ms:  float


class HealthResponse(BaseModel):
    status:     Literal["healthy", "degraded", "unhealthy"]
    components: dict[str, str]