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
    fetch_k:   Optional[int]   = Field(default=None, ge=5, le=50)
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    collections: list[Literal["IPC", "BNS"]] = Field(default=["IPC", "BNS"])
    # Opt-in interpretive commentary (committee report / SOR). Off by default so
    # general statute Q&A stays authoritative.
    include_context: bool = False


class ContextSnippet(BaseModel):
    """A retrieved chunk of NON-authoritative interpretive commentary."""
    text:         str
    doc_type:     str                     # "committee_report" | "sor"
    source:       str                     # human-readable display name
    page_number:  Optional[int] = None
    score:        float


class Citation(BaseModel):
    n:             int
    source_path:   str
    page_number:   Optional[int]  = None
    section_title: Optional[str]  = None
    score:         float

    act:           Optional[str]    = None    # "IPC" | "BNS"
    section:       Optional[str]    = None    # "302", "498A", etc.
    corresponds_to: Optional[str]   = None    # cross-act mapping
    change_status: Optional[str]    = None    # "new"|"changed"|"deleted"|"unchanged"

class CrossReference(BaseModel):
    concordance_row: Optional[int] = None
    ipc_section:     Optional[str] = None
    bns_section:     Optional[str] = None
    status:          Optional[str] = None
    page_number:     Optional[int] = None
    ipc_citation:    Optional[Citation] = None   # real IPC section text
    bns_citation:    Optional[Citation] = None   # real BNS section text

class AnswerResponse(BaseModel):
    question:       str
    answer:         str
    citations:      list[Citation]
    cross_reference: Optional[CrossReference] = None   # ← new
    context:        list["ContextSnippet"] = []        # interpretive commentary
    retriever:      str
    latency_ms:     float


class HealthResponse(BaseModel):
    status:     Literal["healthy", "degraded", "unhealthy"]
    components: dict[str, str]


class DocInfo(BaseModel):
    doc_id:       str
    filename:     str
    char_count:   int
    uploaded_at:  str
    section_refs: list[str]


class DocListResponse(BaseModel):
    documents: list[DocInfo]
    total:     int


class DeleteResponse(BaseModel):
    doc_id:  str
    deleted: bool


class CaseExcerptOut(BaseModel):
    text:          str
    score:         float
    page_number:   Optional[int] = None
    section_title: Optional[str] = None


class CaseAnswerRequest(BaseModel):
    doc_id:   str = Field(..., min_length=1)
    question: str = Field(..., min_length=1, max_length=2000)
    top_k:    int = Field(default=5, ge=1, le=20)
    case_k:   int = Field(default=5, ge=1, le=20)   # case excerpts to inject
    collections: list[Literal["IPC", "BNS"]] = Field(default=["IPC", "BNS"])
    # Interpretation is the whole point of case Q&A → commentary ON by default.
    include_context: bool = True


class CaseAnswerResponse(BaseModel):
    question:        str
    answer:          str
    citations:       list[Citation]            # IPC/BNS statutory citations
    case_excerpts:   list[CaseExcerptOut]      # from the uploaded case (isolated)
    cross_reference: Optional[CrossReference] = None
    context:         list["ContextSnippet"] = []   # interpretive commentary
    latency_ms:      float

class EvalRetrievalRequest(BaseModel):
    retriever: Literal["hybrid_reranked", "dense", "bm25", "ensemble"] = "hybrid_reranked"
    top_k:     int = Field(default=5, ge=1, le=20)


class EvalRetrievalResponse(BaseModel):
    retriever:    str
    n_examples:   int
    top_k:        int
    metrics:      dict             # hit@k, recall@k, mrr, snippet@k
    by_difficulty: dict             # same metrics, sliced
    latency_ms:   float


class ThresholdSweepRequest(BaseModel):
    thresholds: list[float] = Field(
        default=[0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
    )
    top_k:      int = Field(default=5, ge=1, le=20)


class ThresholdSweepResponse(BaseModel):
    top_k:       int
    rows:        list[dict]   # one per threshold
    recommended: dict         # threshold with highest f1
    latency_ms:  float