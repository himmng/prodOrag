from abc import ABC, abstractmethod
from typing import List

import httpx

from backend.core.config import AppConfig


class EmbeddingClient(ABC):
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        ...


class OpenAICompatibleEmbeddingClient(EmbeddingClient):
    def __init__(self, base_url: str, api_key: str | None, model: str):
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        self._base_url = base
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(timeout=30.0)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "input": texts,
        }

        url = f"{self._base_url}/v1/embeddings"
        resp = await self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]


def create_embedding_client(config: AppConfig) -> EmbeddingClient:
    emb_cfg = config.embeddings
    return OpenAICompatibleEmbeddingClient(
        base_url=emb_cfg.base_url,
        api_key=emb_cfg.api_key,
        model=emb_cfg.model,
    )
