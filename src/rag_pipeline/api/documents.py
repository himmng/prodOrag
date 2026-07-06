"""Session-scoped uploaded-case store with per-case ISOLATED vector collections.

Each uploaded case file is parsed, chunked, embedded, and stored in its OWN
ephemeral Chroma collection named ``case_{session_id}_{doc_id}``. Guarantees:

- **Never mixed with the corpus.** Case chunks never land in IPC_Corpus/BNS_Corpus
  (enforced in ``vectorstore.make_case_vectorstore`` via a reserved-name guard).
- **Never leaked across sessions.** The store is keyed by session token first, so
  one client cannot read, search, or delete another client's case — even with a
  known doc_id. Each case is also a physically separate collection.
- **Bounded.** Every case carries a TTL; expiry drops both the metadata and the
  underlying Chroma collection. Resets fully on process restart.

The interface is deliberately Redis/managed-vector-store friendly for the cloud
path: session_id -> key prefix, expires_at -> server-side expiry.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import TYPE_CHECKING

from rag_pipeline.config import log
from rag_pipeline.vectorstore import make_case_vectorstore, drop_collection

if TYPE_CHECKING:
    from rag_pipeline.schemas import RagChunk

DEFAULT_TTL_SECONDS = 3600  # cases expire 1h after upload


# Corpus-agnostic section reference patterns in uploaded documents:
# "Section 302", "Sec. 376(2)", "§ 304B", "S.420"
_SECTION_RE = re.compile(
    r"(?:section|sec\.?|§|S\.)\s*(\d+[A-Z]*(?:\(\d+\))?)",
    re.IGNORECASE,
)


def extract_section_refs(text: str) -> list[str]:
    """Pull unique statute section references mentioned in the document text."""
    seen: dict[str, None] = {}
    for m in _SECTION_RE.findall(text):
        key = m.upper().replace(" ", "")
        seen.setdefault(key, None)
    return list(seen.keys())


def slugify_filename(filename: str, maxlen: int = 40) -> str:
    """Filesystem/Chroma-safe slug from an upload filename (no extension).

    Chroma collection names allow only [a-zA-Z0-9._-]; we reduce to that,
    collapse runs, trim, and cap the length so `case_<slug>_<doc_id>` stays
    within Chroma's 63-char limit. Falls back to 'file' if nothing survives.
    """
    stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    stem = re.sub(r"_+", "_", stem)[:maxlen].strip("_")
    return stem or "file"


@dataclass
class CaseExcerpt:
    """One retrieved chunk from a case's isolated collection."""
    text:          str
    score:         float
    page_number:   int | None = None
    section_title: str | None = None


@dataclass
class UploadedDoc:
    doc_id:          str
    session_id:      str
    filename:        str
    collection_name: str
    char_count:      int
    n_chunks:        int
    uploaded_at:     str
    expires_at:      datetime
    section_refs:    list[str] = field(default_factory=list)

    def is_expired(self, now: datetime | None = None) -> bool:
        return (now or datetime.now(timezone.utc)) >= self.expires_at


class DocumentStore:
    """Thread-safe, session-scoped case store backed by isolated vector collections.

    Two-level map: session_id -> {doc_id -> UploadedDoc}. Every op is scoped to a
    session token. TTL is enforced lazily on access and via purge_expired().
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self._sessions: dict[str, dict[str, UploadedDoc]] = {}
        self._ttl = ttl_seconds
        self._lock = Lock()

    # ── writes ────────────────────────────────────────────────────────────
    def add(self, session_id: str, filename: str, chunks: list["RagChunk"]) -> UploadedDoc:
        """Embed the case's chunks into a fresh isolated collection and register it."""
        now = datetime.now(timezone.utc)
        doc_id = uuid.uuid4().hex[:12]
        # Human-readable collection/dir name: case_<filename-slug>_<doc_id>.
        # doc_id keeps it unique; session isolation is enforced by the store map.
        collection_name = f"case_{slugify_filename(filename)}_{doc_id}"
        full_text = "\n\n".join(c.text for c in chunks)

        vs = make_case_vectorstore(collection_name)
        vs.add_texts(
            texts=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id":        doc_id,
                    "session_id":    session_id,
                    "filename":      filename,
                    "chunk_index":   i,
                    "page_number":   c.page_number if c.page_number is not None else -1,
                    "section_title": c.section_title or "",
                }
                for i, c in enumerate(chunks)
            ],
        )

        doc = UploadedDoc(
            doc_id=doc_id,
            session_id=session_id,
            filename=filename,
            collection_name=collection_name,
            char_count=len(full_text),
            n_chunks=len(chunks),
            uploaded_at=now.isoformat(timespec="seconds"),
            expires_at=now + timedelta(seconds=self._ttl),
            section_refs=extract_section_refs(full_text),
        )
        with self._lock:
            self._sessions.setdefault(session_id, {})[doc_id] = doc
        log.info(f"case uploaded: session={session_id[:8]} doc={doc_id} chunks={len(chunks)}")
        return doc

    # ── reads ─────────────────────────────────────────────────────────────
    def get(self, session_id: str, doc_id: str) -> UploadedDoc | None:
        now = datetime.now(timezone.utc)
        with self._lock:
            doc = self._sessions.get(session_id, {}).get(doc_id)
            if doc is None:
                return None
            if doc.is_expired(now):
                self._drop(session_id, doc_id)
                return None
            return doc

    def list_all(self, session_id: str) -> list[UploadedDoc]:
        now = datetime.now(timezone.utc)
        with self._lock:
            bucket = self._sessions.get(session_id, {})
            for did in list(bucket):
                if bucket[did].is_expired(now):
                    self._drop(session_id, did)
            return list(self._sessions.get(session_id, {}).values())

    def search_case(
        self, session_id: str, doc_id: str, query: str, top_k: int = 5
    ) -> list[CaseExcerpt]:
        """Semantic search WITHIN one case's isolated collection."""
        doc = self.get(session_id, doc_id)
        if doc is None:
            return []
        vs = make_case_vectorstore(doc.collection_name)
        results = vs.similarity_search_with_score(query, k=top_k)
        out: list[CaseExcerpt] = []
        for d, dist in results:
            page = d.metadata.get("page_number", -1)
            out.append(
                CaseExcerpt(
                    text=d.page_content,
                    score=1.0 - (dist ** 2) / 2,  # L2 → cosine, matches DenseRetriever
                    page_number=page if page not in (-1, None) else None,
                    section_title=d.metadata.get("section_title") or None,
                )
            )
        return out

    # ── deletes / GC ──────────────────────────────────────────────────────
    def delete(self, session_id: str, doc_id: str) -> bool:
        with self._lock:
            return self._drop(session_id, doc_id)

    def purge_expired(self) -> int:
        """Drop all expired cases across sessions; returns count removed."""
        now = datetime.now(timezone.utc)
        removed = 0
        with self._lock:
            for sid in list(self._sessions):
                for did in list(self._sessions[sid]):
                    if self._sessions[sid][did].is_expired(now):
                        self._drop(sid, did)
                        removed += 1
        return removed

    def _drop(self, session_id: str, doc_id: str) -> bool:
        """Remove one case + its collection; prune empty session. Caller holds lock."""
        bucket = self._sessions.get(session_id)
        if not bucket:
            return False
        doc = bucket.pop(doc_id, None)
        if doc is None:
            return False
        drop_collection(doc.collection_name)  # tears down the isolated vectors
        if not bucket:
            self._sessions.pop(session_id, None)
        return True
