from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings

from backend.core.config import AppConfig


class VectorStore:
    def __init__(self, config: AppConfig):
        self._config = config
        self._client = chromadb.Client(
            Settings(persist_directory=config.storage.vector_dir)
        )
        self._collection = self._client.get_or_create_collection("protoRAG_docs")

    def add_documents(
        self,
        doc_id: str,
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=[m.get("text", "") for m in metadatas],
        )

    def query(self, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
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


_vector_store_cache: Dict[str, VectorStore] = {}


def get_vector_store(config: AppConfig) -> VectorStore:
    key = config.storage.vector_dir
    store = _vector_store_cache.get(key)
    if store is None:
        store = VectorStore(config)
        _vector_store_cache[key] = store
    return store
