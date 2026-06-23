""" Cross-encoder reranker + two-stage retriever.

A reranker scores (query, document) pair jointly - more accurate but slower than the bi-encoder used for initial retrieval.
Use as a second stage: get fetch_k candidates from a fast retriever, thank rerank to top_k.

Default: BAAI/bge-reranker-base (~100MB, CPU friendly)"""

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from rag_pipeline.config import log
from rag_pipeline.retrievers.base import BaseRetriever
import numpy as np
from sentence_transformers import CrossEncoder


if TYPE_CHECKING:
    from langchain_core.documents import Document


class Reranker:
    """Wraps a HuggingFace cross-encoder for relevance scoring."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        normalize: bool = True,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading reranker: {model_name} (device={device})")
        self._model = CrossEncoder(model_name, device=device, trust_remote_code=False)
        self.normalize = normalize

    def score(self, query: str, docs: list) -> list[float]:
        """Score (query, doc) pairs. Returns floats; higher = more relevant."""
        # Accept Document objects or plain strings
        texts = [d.page_content if hasattr(d, "page_content") else str(d) for d in docs]
        pairs = [[query, t] for t in texts]

        scores = self._model.predict(pairs, show_progress_bar=False)
        scores = np.asarray(scores, dtype=np.float32)

        if self.normalize:
            # Sigmoid → [0, 1] (BGE reranker outputs raw logits)
            scores = 1.0 / (1.0 + np.exp(-scores))

        return scores.tolist()



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