"""
BM25 lexical retriever -- in-memory, rank_bm25 backed.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rag_pipeline.retrievers.base import BaseRetriever

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from rag_pipeline.schemas import RagChunk

## Tokenizer keeps hyphens / digits - preserves section IDs like "498A", "108-A"
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")

def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]

class BM25Retriever(BaseRetriever):
    """In-memory BM25 over a chunk corpus, Build Once, query multiple"""

    def __init__(self, chunks: list["RagChunk"]):
        from rank_bm25 import BM25Okapi
        from langchain_core.documents import Document

        self.chunks = chunks
        self.documents = [
            Document(page_content=c.text, metadata=c.to_langchain_metadata())
            for c in chunks
        ]
        self.bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple["Document", float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.documents, scores),
                        key=lambda x: x[1],
                        reverse=True,)
        return ranked[:top_k]