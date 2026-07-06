""" Universal chunk schema - the unit of retrieval that flow throught the piepline.

Every parser produces `RagChunk` objects; every retriever consumes them. The schema is vendor-neutral.
The ONLY point where we couple to a vector store is `to_langchain_metadata()` - thats's the entire chorma boundary.

`chunk_id` defaults to `content_hash` (SHA-256 of source_path + text, truncated to 16 hex chars). This means re-parsing the same source
produces the same chunk_ids -> chroma ingest becomes idempotent at the chunk level.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator

SourceFormat = Literal[
    "pdf", "docx", "pptx", "html", "md", "txt",
    "csv", "xlsx", "xls", "json", "jsonl", "ndjson",
]

ChunkType = Literal[
    "definition",     # Sections 1-3 of either act, defining terms
    "offence",        # The substantive offence text
    "punishment",     # "Whoever ... shall be punished with..."
    "exception",      # "This section does not apply to..."
    "illustration",   # "Illustration: A puts jewels..."
    "explanation",    # "Explanation 1.— For the purposes..."
    "proviso",        # "Provided that..."
    "other",          # Fallback for unclassifiable spans
]


def _content_hash(source_path: str, text: str, n: int = 16) -> str:
    """SHA-256 of (source_path \0 text), truncated to `n` hex chars."""
    h = hashlib.sha256()
    h.update(source_path.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:n]

class RagChunk(BaseModel):
    """A single retrieval unit. Produced by parsers, consumed by retrievers."""

    # core content
    text: str
    source_path: str
    source_format: SourceFormat

    # computed identity (auto-filled by default_factory)
    content_hash: str = ""
    chunk_id: str = ""

    # optional provenance - populated by format-aware parsers
    page_number: Optional[int] = None #PDfs
    slide_number: Optional[int] = None #PPTX
    sheet_name: Optional[str] = None    #Excel
    section_header: Optional[str] = None #All formats
    section_title: Optional[str] = None  #All formats
    row_range: Optional[str] = None       #CSV/Excel, e.g. "A1:C10"
    element_type: Optional[str] = None      #HTML, e.g. "p", "h1", "li"

    # free-from extras (must be primitives scalers to survive chroma)
    extras: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _fill_identity(self):
        """Compute content_hash and chunk_id if not provided."""
        if not self.content_hash:
            self.content_hash = _content_hash(self.source_path, self.text)
        if not self.chunk_id:
            self.chunk_id = self.content_hash
        return self
    
    # boundary methods

    def to_langchain_metadata(self) -> dict[str, Any]:
        """Flatten to chroma-compatible primitives. Optional fields included only when set."""
        meta: dict[str, Any] = {
            "source_path": self.source_path,
            "source_format": self.source_format,
            "chunk_id": self.chunk_id,
            "content_hash": self.content_hash,
        }

        for key, val in [
            ("page_number", self.page_number),
            ("slide_number", self.slide_number),
            ("sheet_name", self.sheet_name),
            ("section_header", self.section_header),
            ("section_title", self.section_title),
            ("row_range", self.row_range),
            ("element_type", self.element_type),
        ]:
            if val is not None:
                meta[key] = val
        # flatten extras (one level deep; only primitive survive)
        for k, v in self.extras.items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
        return meta
    
    def to_langchain_document(self):
        """Convert to langchain Document for vectorstore ingest."""
        from langchain_core.documents import Document
        return Document(
            page_content=self.text,
            metadata=self.to_langchain_metadata(),
        )



class StatuteChunk(BaseModel):
    """A chunk of statutory text (IPC or BNS), enriched with cross-references.

    chunk_id is content-addressed: SHA-256 of (act + section + text)[:16].
    This makes ingest idempotent — same content → same id, no duplicates.
    """

    chunk_id:        str

    # Provenance
    source_path:     str
    page_number:     int

    # Statute structure
    act:             str                              # corpus-defined act label
    section:         str                              # "302", "302A", "498A"
    subsection:      Optional[str]    = None         # "(1)", "(2)(a)"
    parent_section:  Optional[str]    = None         # if subsection chunk, link to parent
    section_title:   str
    chapter_number:  Optional[str]    = None
    chapter_title:   Optional[str]    = None

    # Classification
    chunk_type:      ChunkType        = "other"

    # Cross-reference (joined from concordance at ingest)
    corresponds_to:  Optional[str]    = None         # IPC chunk → BNS section, vice versa
    change_status:   Optional[Literal[
                       "new", "changed", "deleted", "unchanged"
                     ]] = None

    # Content
    text:            str
    char_count:      int               = Field(default=0)

    def model_post_init(self, __context) -> None:
        if self.char_count == 0:
            self.char_count = len(self.text)
    
    def to_langchain_metadata(self) -> dict:
        """Return metadata dict suitable for langchain Document (excludes text)."""
        data = self.model_dump()
        data.pop("text", None)
        # Chroma/langchain don't accept None values
        return {k: v for k, v in data.items() if v is not None}

    @classmethod
    def make_id(cls, act: str, section: str, text: str) -> str:
        """Content-addressed ID. Two chunks with identical (act, section, text)
        will collide — callers can disambiguate by including subsection or
        chunk_type in the `section` argument."""
        h = hashlib.sha256(f"{act}|{section}|{text}".encode("utf-8")).hexdigest()
        return h[:16]