from typing import List, Dict, Any, Union

from pathlib import Path

import chromadb

from backend.core.config import AppConfig


class VectorStore:
    def __init__(self, config_or_directory: Union[AppConfig, str]):
        if isinstance(config_or_directory, AppConfig):
            persist_directory = config_or_directory.storage.vector_dir
        else:
            persist_directory = str(config_or_directory)
        self._persist_directory = persist_directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_directory)
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


def get_conversation_vector_dir(config: AppConfig, conversation_id: str | None) -> str:
    base_dir = Path(config.storage.vector_dir)
    per_conversation = getattr(config.storage, "per_conversation", True)
    if not per_conversation or conversation_id is None:
        base_dir.mkdir(parents=True, exist_ok=True)
        return str(base_dir)

    conv_dir = base_dir / conversation_id
    conv_dir.mkdir(parents=True, exist_ok=True)
    return str(conv_dir)


def get_vector_store_for_dir(config: AppConfig, vector_dir: str) -> VectorStore:
    backend = getattr(config.storage, "vector_backend", "chroma")
    if backend != "chroma":
        raise ValueError(f"Unsupported vector backend: {backend}")

    key = vector_dir
    store = _vector_store_cache.get(key)
    if store is None:
        store = VectorStore(vector_dir)
        _vector_store_cache[key] = store
    return store


def get_vector_store(config: AppConfig) -> VectorStore:
    vector_dir = config.storage.vector_dir
    return get_vector_store_for_dir(config, vector_dir)
