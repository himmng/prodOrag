"""In-memory uploaded document store.

Documents uploaded via /documents/upload live here for the API process
lifetime. Strictly isolated from the IPC ChromaDB collection — there is
NO code path that writes uploaded content into the corpus vectorstore.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock


# Common IPC section reference patterns in case files:
# "Section 302", "Sec. 376(2)", "§ 304B", "S.420 IPC"
_SECTION_RE = re.compile(
    r"(?:section|sec\.?|§|S\.)\s*(\d+[A-Z]*(?:\(\d+\))?)\s*(?:of\s+)?(?:IPC|I\.P\.C\.)?",
    re.IGNORECASE,
)


def extract_section_refs(text: str) -> list[str]:
    """Pull unique IPC section references mentioned in the document text."""
    seen: dict[str, None] = {}
    for m in _SECTION_RE.findall(text):
        key = m.upper().replace(" ", "")
        seen.setdefault(key, None)
    return list(seen.keys())


@dataclass
class UploadedDoc:
    doc_id:       str
    filename:     str
    text:         str
    char_count:   int
    uploaded_at:  str
    section_refs: list[str] = field(default_factory=list)


class DocumentStore:
    """Thread-safe in-memory document store. Resets on process restart."""

    def __init__(self):
        self._docs: dict[str, UploadedDoc] = {}
        self._lock = Lock()

    def add(self, filename: str, text: str) -> UploadedDoc:
        doc = UploadedDoc(
            doc_id=uuid.uuid4().hex[:12],
            filename=filename,
            text=text,
            char_count=len(text),
            uploaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            section_refs=extract_section_refs(text),
        )
        with self._lock:
            self._docs[doc.doc_id] = doc
        return doc

    def get(self, doc_id: str) -> UploadedDoc | None:
        with self._lock:
            return self._docs.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            return self._docs.pop(doc_id, None) is not None

    def list_all(self) -> list[UploadedDoc]:
        with self._lock:
            return list(self._docs.values())