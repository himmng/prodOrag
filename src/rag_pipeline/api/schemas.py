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
    # Empty = all corpus acts; validated at runtime against the loaded corpus.
    collections: list[str] = Field(default_factory=list)
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
    """A cross-reference mapping between two corpus acts. Act labels are DATA
    (source_act/target_act), supplied by the corpus config — not hardcoded."""
    concordance_row: Optional[int] = None
    source_act:      Optional[str] = None        # e.g. "IPC"
    target_act:      Optional[str] = None        # e.g. "BNS"
    source_section:  Optional[str] = None
    target_section:  Optional[str] = None
    status:          Optional[str] = None
    page_number:     Optional[int] = None
    source_citation: Optional[Citation] = None   # real source-act section text
    target_citation: Optional[Citation] = None   # real target-act section text

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


class CrossRefMeta(BaseModel):
    source_act: str
    target_act: str
    pdf_label:  str


class MetaResponse(BaseModel):
    """Corpus metadata so clients (dashboard) render without hardcoding IPC/BNS."""
    corpus:          str
    display_name:    str
    acts:            list[str]
    pdf_acts:        list[str]              # acts renderable via /documents/page-image
    context_enabled: bool
    cross_reference: Optional[CrossRefMeta] = None


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
    # Empty = all corpus acts; validated at runtime against the loaded corpus.
    collections: list[str] = Field(default_factory=list)
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
    by_category:   dict = {}
    by_category_difficulty: dict = {}
    negatives:     dict = {}
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