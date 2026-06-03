""" Dense (semantic) retriever - thin wrapper around the vectorstore."""

from __future__ import annotations
from typing import TYPE_CHECKING

from rag_pipeline.retrievers.base import BaseRetriever
from rag_pipeline.vectorstore import get_vectorstore

if TYPE_CHECKING:
    from langchain_core.documents import Document

class DenseRetriever(BaseRetriever):
    """Semantic similarity search via the configured vectorstore.
    
    Return SIMILARITY score (higher = better), converted from Chroma's or Azure, qdrant (future updates) cosine distance. 
    
    """

    def __init__(self, collection_name: str = "rag_default"):
        self.collection_name = collection_name
        self.vectorstore = get_vectorstore(collection_name)
    
    def retrieve(self, query: str, top_k: int = 5) -> list[tuple["Document", float]]:
        results =  self.vectorstore.similarity_search_with_score(query, k=top_k)
        # chroma -> chroma distance (lower=better) -> convert to similarity (higher=better)
        # Chroma default = L2 distance over (typically) unit-length embeddings.
        # Convert L2 → cosine similarity: cos = 1 - dist²/2
        return [(doc, 1.0 - (dist ** 2) / 2) for doc, dist in results]