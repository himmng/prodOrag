from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from backend.core.config import AppConfig


class VectorStore:
    def __init__(self, persist_directory: str, collection_name: str = "protoRAG_docs"):
        self._persist_directory = persist_directory
        self._collection_name = collection_name

        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Embedded (in-process) Qdrant with on-disk storage in persist_directory
        self._client = QdrantClient(
            path=persist_directory,
            prefer_grpc=False,
        )

        # Lazily ensure collection exists when we first see embeddings
        self._collection_created = False

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_created:
            return

        collections = self._client.get_collections().collections
        names = {c.name for c in collections}
        if self._collection_name not in names:
            self._client.recreate_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
        self._collection_created = True

    def add_documents(
        self,
        doc_id: str,
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> None:
        if not embeddings:
            return

        # Ensure collection exists with proper vector size
        vector_size = len(embeddings[0])
        self._ensure_collection(vector_size)

        points: List[PointStruct] = []
        for eid, vec, meta in zip(ids, embeddings, metadatas):
            payload = dict(meta or {})
            payload.setdefault("doc_id", doc_id)
            # Qdrant local collections expect UUID ids; generate a stable UUID from our string id
            # and keep the original id in the payload for debugging/lookup.
            payload.setdefault("chunk_id", eid)
            qdrant_id = uuid.uuid5(uuid.NAMESPACE_URL, f"protoRAG:{eid}")
            points.append(
                PointStruct(
                    id=str(qdrant_id),
                    vector=vec,
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

    def query(self, embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        if not embedding:
            return []

        collections = self._client.get_collections().collections
        names = {c.name for c in collections}
        if self._collection_name not in names:
            return []

        search_result = self._client.search(
            collection_name=self._collection_name,
            query_vector=embedding,
            limit=top_k,
        )

        outputs: List[Dict[str, Any]] = []
        for scored_point in search_result:
            payload = dict(scored_point.payload or {})
            point_id = scored_point.id
            entry = dict(payload)
            entry.update(
                {
                    "id": point_id,
                    "text": payload.get("text", ""),
                }
            )
            outputs.append(entry)

        return outputs


_vector_store_cache: Dict[str, VectorStore] = {}


def get_vector_store_for_dir(config: AppConfig, vector_dir: str) -> VectorStore:
    key = vector_dir
    store = _vector_store_cache.get(key)
    if store is None:
        store = VectorStore(vector_dir)
        _vector_store_cache[key] = store
    return store


def get_vector_store(config: AppConfig) -> VectorStore:
    return get_vector_store_for_dir(config, config.storage.vector_dir)


def get_conversation_vector_dir(config: AppConfig, conversation_id: Optional[str]) -> str:
    if not conversation_id:
        return config.storage.vector_dir
    return str(Path(config.storage.vector_dir) / conversation_id)
