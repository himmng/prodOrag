from typing import List, Dict, Any

from pathlib import Path

import chromadb
from chromadb.config import Settings

from backend.core.config import AppConfig


class VectorStore:
    """Wrapper around a ChromaDB collection stored in a specific directory."""

    def __init__(self, persist_directory: str):
        self._persist_directory = persist_directory
        self._client = chromadb.Client(
            Settings(persist_directory=persist_directory)
        )
        self._collection = self._client.get_or_create_collection("protoRAG_docs")

    def add_documents(
        self,
        doc_id: str,
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        """Add documents and their embeddings to the collection."""

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=[m.get("text", "") for m in metadatas],
        )

    def query(self, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Query the collection by a single embedding and return top_k results."""

        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]
        ids = result.get("ids", [[]])[0]

        outputs: List[Dict[str, Any]] = []
        for mid, m, d in zip(ids, metadatas, documents):
            entry = dict(m or {})
            entry.update({"id": mid, "text": d})
            outputs.append(entry)
        return outputs


# Cache of VectorStore instances keyed by their persist directory
_vector_store_cache: Dict[str, VectorStore] = {}


def get_conversation_vector_dir(config: AppConfig, conversation_id: str | None) -> str:
    """Return the directory where vectors for a conversation should be stored.

    If conversation_id is None, use the base vector_dir.
    Otherwise, create/use a subdirectory per conversation.
    """

    base_dir = Path(config.storage.vector_dir)
    if conversation_id is None:
        return str(base_dir)

    conv_dir = base_dir / conversation_id
    conv_dir.mkdir(parents=True, exist_ok=True)
    return str(conv_dir)


def get_vector_store_for_dir(config: AppConfig, vector_dir: str) -> VectorStore:
    """Return a cached VectorStore for the given directory.

    This ensures each directory (e.g., per conversation) has its own Chroma
    client and persisted data, while avoiding repeated client construction.
    """

    key = vector_dir
    store = _vector_store_cache.get(key)
    if store is None:
        store = VectorStore(vector_dir)
        _vector_store_cache[key] = store
    return store


def get_vector_store(config: AppConfig) -> VectorStore:
    """Backward-compatible helper using the base vector_dir from config."""

    vector_dir = config.storage.vector_dir
    return get_vector_store_for_dir(config, vector_dir)
