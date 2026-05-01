from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import httpx

from backend.core.config import AppConfig, EmbeddingsConfig


class EmbeddingClient(ABC):
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        ...


class OpenAICompatibleEmbeddingClient(EmbeddingClient):
    def __init__(self, base_url: str, api_key: Optional[str], model: str):
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


def _resolve_embedding_config(cfg: EmbeddingsConfig) -> Tuple[str, Optional[str], str]:
    provider = (cfg.provider or "").lower()
    base_url = cfg.base_url
    api_key = cfg.api_key
    model = cfg.model

    if provider == "ollama":
        if not base_url:
            base_url = "http://localhost:11434"
        if not model:
            model = "nomic-embed-text"
    elif provider in ("lmstudio", "lm_studio", "lm-studio"):
        if not base_url:
            base_url = "http://localhost:1234"
    elif provider in ("openai", "openai-compatible", "openai_compatible"):
        if not base_url:
            base_url = "https://api.openai.com"
    else:
        if not base_url:
            base_url = "http://localhost:11434"

    return str(base_url), api_key, model


def create_embedding_client(config: AppConfig) -> EmbeddingClient:
    emb_cfg = config.embeddings
    base_url, api_key, model = _resolve_embedding_config(emb_cfg)
    return OpenAICompatibleEmbeddingClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
    )
