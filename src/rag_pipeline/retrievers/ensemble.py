""" Ensemble retriever -- RRF fusion over multiple base retriever """

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_pipeline.retrievers.base import BaseRetriever, reciprocal_rank_fusion

if TYPE_CHECKING:
    from langchain_core.documents import Document

class EnsembleRetriever(BaseRetriever):
    """ RRF over an arbitrary list of base retrievers.
    
    Each base is queried independently with fetch_k results.
    Their rankings are then fused with Reciprocal Rank Fusion.
    
    """

    def __init__(
            self,
            retrievers: list[BaseRetriever],
            fetch_k: int = 20,
            weights: list[float] | None = None,
            rrf_k: int = 60,
    ):
        if not retrievers:
            raise ValueError("EnsembleRetriever requires at least one retriever")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError(f"weights ({len(weights)}) != retriever ({len(retrievers)})")
        
        self.retrievers = retrievers
        self.fetch_k = fetch_k
        self.weights = weights or [1.0] * len(retrievers)
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple["Document", float]]:
        rankings: list[list["Document"]] = []
        for r in self.retrievers:
            results = r.retrieve(query, top_k=self.fetch_k)
            rankings.append([doc for doc, _ in results])
        
        fused = reciprocal_rank_fusion(rankings, k= self.rrf_k, weights=self.weights)
        return fused[:top_k]