""" vectorstore factory.

The single place in the codebase that talks to a specific vector DB.

Current: Chroma DB (can handle upto 10M chunks)

Future: Azure AI search / Qdrant / pgvector 

upgrade will require to change only this file, everything else like retrievers, ingest, eval remain unaffected.

"""
import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")

from __future__ import annotations
from functools import lru_cache
from typing import TYPE_CHECKING

from rag_pipeline.config import cfg, log
from rag_pipeline.providers import get_embeddings

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore

@lru_cache(maxsize=4)
def get_vectorstore(collection_name: str = "rag_default") -> "VectorStore":
    """ Return a persistent vectorstore for the given collection."""
    # To do later:
    # cfg.vectorsotre_provider = "azure_ai_search" or "qdrant"
    provider = "chroma"
    log.info(f"initializing vectorstore (provider = {provider}, collection={collection_name})")

    if provider == "chroma":
        from langchain_chroma import Chroma
        return Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=str(cfg.CHROMA_PERSIST_DIR),
        )
    raise ValueError(f"Unknown VECTORSTORE_PROVIDER: {provider!r}")