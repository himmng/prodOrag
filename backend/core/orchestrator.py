from typing import List, Dict, Any, Optional

from backend.core.config import AppConfig
from backend.core.embedding_client import create_embedding_client, EmbeddingClient
from backend.core.llm_client import create_llm_client, LLMClient
from backend.core.vector_store import get_vector_store, VectorStore


class RAGOrchestrator:
    def __init__(
        self,
        config: AppConfig,
        llm_client: Optional[LLMClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self._config = config
        self._llm_client = llm_client or create_llm_client(config)
        self._embedding_client = embedding_client or create_embedding_client(config)
        self._vector_store = vector_store or get_vector_store(config)

    async def chat(self, user_message: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        embeddings = await self._embedding_client.embed([user_message])
        if not embeddings:
            answer = ""
            sources: List[Dict[str, Any]] = []
            return {"answer": answer, "sources": sources}

        query_embedding = embeddings[0]
        results = self._vector_store.query(query_embedding, self._config.rag.top_k)

        context_parts = []
        for r in results:
            text = r.get("text") or ""
            context_parts.append(text)
        context = "\n\n".join(context_parts)

        messages: List[Dict[str, Any]] = []
        messages.append(
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions using the provided context.",
            }
        )
        if context:
            messages.append(
                {
                    "role": "system",
                    "content": f"Context:\n{context}",
                }
            )

        for h in history:
            role = h.get("role", "user")
            content = h.get("content", "")
            messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_message})

        answer_text = await self._llm_client.chat(messages)

        return {"answer": answer_text, "sources": results}


_orchestrator_cache: Dict[str, RAGOrchestrator] = {}


def get_orchestrator(config: AppConfig) -> RAGOrchestrator:
    key = f"{config.llm.provider}:{config.embeddings.provider}:{config.storage.vector_dir}"
    orch = _orchestrator_cache.get(key)
    if orch is None:
        orch = RAGOrchestrator(config)
        _orchestrator_cache[key] = orch
    return orch
