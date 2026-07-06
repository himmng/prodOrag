""" vectorstore factory.

The single place in the codebase that talks to a specific vector DB.

Current: Chroma DB (can handle upto 10M chunks)

Future: Azure AI search / Qdrant / pgvector 

upgrade will require to change only this file, everything else like retrievers, ingest, eval remain unaffected.

"""


from __future__ import annotations
from functools import lru_cache
from typing import TYPE_CHECKING
import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
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


# ── Ephemeral, per-session case collections ──────────────────────────────
# These are ISOLATED from the corpus collections (IPC_Corpus/BNS_Corpus) and
# from each other. They are NOT lru_cached — each upload gets a fresh handle,
# and the collection is dropped on TTL/delete. Reserved corpus names may never
# be created or dropped through this path.

_RESERVED_COLLECTIONS = {"IPC_Corpus", "BNS_Corpus", "rag_default"}


def make_case_vectorstore(collection_name: str) -> "VectorStore":
    """Return a fresh (uncached) vectorstore for an isolated case upload."""
    if collection_name in _RESERVED_COLLECTIONS:
        raise ValueError(f"Refusing to use reserved collection name: {collection_name!r}")
    from langchain_chroma import Chroma
    log.info(f"creating isolated case collection: {collection_name}")
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(cfg.CHROMA_PERSIST_DIR),
    )


def drop_collection(collection_name: str) -> None:
    """Delete an ephemeral case collection. Refuses reserved corpus names."""
    if collection_name in _RESERVED_COLLECTIONS:
        raise ValueError(f"Refusing to drop reserved collection: {collection_name!r}")
    from langchain_chroma import Chroma
    try:
        vs = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=str(cfg.CHROMA_PERSIST_DIR),
        )
        vs.delete_collection()
        log.info(f"dropped isolated case collection: {collection_name}")
    except Exception as e:  # best-effort GC — never crash a request on cleanup
        log.warning(f"failed to drop collection {collection_name}: {e}")