""" Cross-encoder reranker + two-stage retriever.

A reranker scores (query, document) pair jointly - more accurate but slower than the bi-encoder used for initial retrieval.
Use as a second stage: get fetch_k candidates from a fast retriever, thank rerank to top_k.

Default: BAAI/bge-reranker-base (~100MB, CPU friendly)"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag_pipeline.config import log
from rag_pipeline.retrievers.base import BaseRetriever

if TYPE_CHECKING:
    from langchain_core.documents import Document


class Reranker:
    """Cross-encoder reranker. Returns calibrated [0, 1] scores when normalize=True."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        normalize: bool = True,
    ):
        from FlagEmbedding import FlagReranker
        self.model_name = model_name
        self.normalize = normalize
        log.info(f"Loading reranker: {model_name}")
        self._model = FlagReranker(model_name, use_fp16=False)

    def score(self, query: str, documents: list["Document"]) -> list[float]:
        if not documents:
            return []
        pairs = [[query, d.page_content] for d in documents]
        scores = self._model.compute_score(pairs, normalize=self.normalize)
        if isinstance(scores, float):
            scores = [scores]
        return list(scores)


class RerankedRetriever(BaseRetriever):
    """Two-stage: fast first-pass -> cross-encoder rerank.

    `min_score` (optional) drops candidates below the threshold — useful
    as a safety filter against irrelevant results passing to the LLM.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker: Reranker,
        fetch_k: int = 20,
        min_score: float | None = None,
    ):
        self.base = base_retriever
        self.reranker = reranker
        self.fetch_k = fetch_k
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple["Document", float]]:
        candidates = self.base.retrieve(query, top_k=self.fetch_k)
        if not candidates:
            return []

        docs = [d for d, _ in candidates]
        scores = self.reranker.score(query, docs)

        rescored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        if self.min_score is not None:
            rescored = [(d, s) for d, s in rescored if s >= self.min_score]
        return rescored[:top_k]