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

# On-disk layout under chroma_db/ is split so it is obvious which data is which:
#   chroma_db/corpus/            → the persistent corpus (IPC/BNS/Context), one store
#   chroma_db/cases/<name>/      → one ISOLATED store per uploaded case
# Corpus and case data never share a directory or a sqlite file.
CORPUS_DIR = cfg.CHROMA_PERSIST_DIR / "corpus"
CASES_DIR  = cfg.CHROMA_PERSIST_DIR / "cases"


def _case_dir(collection_name: str):
    return CASES_DIR / collection_name


@lru_cache(maxsize=4)
def get_vectorstore(collection_name: str = "rag_default") -> "VectorStore":
    """ Return a persistent vectorstore for a CORPUS collection."""
    # To do later:
    # cfg.vectorsotre_provider = "azure_ai_search" or "qdrant"
    provider = "chroma"
    log.info(f"initializing vectorstore (provider = {provider}, collection={collection_name})")

    if provider == "chroma":
        from langchain_chroma import Chroma
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=str(CORPUS_DIR),
        )
    raise ValueError(f"Unknown VECTORSTORE_PROVIDER: {provider!r}")


# ── Ephemeral, per-session case collections ──────────────────────────────
# Each uploaded case gets its OWN directory chroma_db/cases/<name>/ — fully
# isolated from the corpus (chroma_db/corpus/) and from every other case. They
# are NOT lru_cached, and deleting a case removes its whole directory, so no
# stray vectors survive. Reserved corpus names may never be used here.

_RESERVED_COLLECTIONS = {"IPC_Corpus", "BNS_Corpus", "BNS_Context", "rag_default"}


def make_case_vectorstore(collection_name: str) -> "VectorStore":
    """Return a fresh (uncached) vectorstore in the case's own directory."""
    if collection_name in _RESERVED_COLLECTIONS:
        raise ValueError(f"Refusing to use reserved collection name: {collection_name!r}")
    from langchain_chroma import Chroma
    case_dir = _case_dir(collection_name)
    case_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"creating isolated case store: {case_dir}")
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(case_dir),
    )


def drop_collection(collection_name: str) -> None:
    """Delete a case: remove its entire directory (physical, guaranteed cleanup)."""
    if collection_name in _RESERVED_COLLECTIONS:
        raise ValueError(f"Refusing to drop reserved collection: {collection_name!r}")
    import shutil
    case_dir = _case_dir(collection_name)
    # Safety: only ever delete inside chroma_db/cases/.
    if CASES_DIR not in case_dir.parents:
        log.warning(f"refusing to drop path outside cases dir: {case_dir}")
        return
    try:
        shutil.rmtree(case_dir, ignore_errors=True)
        log.info(f"dropped isolated case store: {case_dir}")
    except Exception as e:  # best-effort GC — never crash a request on cleanup
        log.warning(f"failed to drop case store {case_dir}: {e}")