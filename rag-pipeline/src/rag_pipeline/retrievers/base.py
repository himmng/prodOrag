""" Retriever interface + Reciprocal Rank Fusion.

Every retriever: retriever(query, top_k) -> list[(Document, score)]     high score = better 

RRF (paper: Cormack 2009) fuses ranking without score normalization - robust to different scoring scales

(dense cosine, BM25 tf-idf, reranker sigmoid)

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document

class BaseRetriever(ABC):
    """
    All retrievers in the pipeline share this interface"""

    @abstractmethod
    def retrieve(
        self, 
        query: str,
        top_k: int = 5,
    ) -> list[tuple["Document", float]]:
        """Return (document, score) tuples in score-descending order."""

def reciprocal_rank_fusion(
    rankings: list[list["Document"]],
    k: int = 60,
    weights: list[float] | None = None,

) -> list[tuple["Document", float]]:
    """RRF: score(document) = sum (weight) / sum (k + rank(document))
    
    Dedups by chunk_id (falls back to content_hash, then text hash)"""

    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"weights={len(weights)} != rankings={len(rankings)}")

    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, "Document"] = {}

    for ranking, w in zip(rankings, weights):
        for rank, doc in enumerate(ranking):
            doc_id = str(
                doc.metadata.get("chunk_id")
                or doc.metadata.get("content_hash")
                or hash(doc.page_content)
            )
            scores[doc_id] += w / (k + rank + 1)
            doc_map.setdefault(doc_id, doc)

    return sorted(
        ((doc_map[d], s) for d, s in scores.items()),
        key=lambda x: x[1],
        reverse=True,
    )