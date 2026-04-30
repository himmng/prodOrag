from abc import ABC, abstractmethod
from typing import List, Dict, Any

import httpx

from backend.core.config import AppConfig


class LLMClient(ABC):
    @abstractmethod
    async def chat(self, messages: List[Dict[str, Any]]) -> str:
        ...


class OpenAICompatibleLLMClient(LLMClient):
    def __init__(self, base_url: str, api_key: str | None, model: str):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(timeout=30.0)

    async def chat(self, messages: List[Dict[str, Any]]) -> str:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "messages": messages,
        }

        url = f"{self._base_url}/v1/chat/completions"
        resp = await self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def create_llm_client(config: AppConfig) -> LLMClient:
    llm_cfg = config.llm
    return OpenAICompatibleLLMClient(
        base_url=llm_cfg.base_url,
        api_key=llm_cfg.api_key,
        model=llm_cfg.model,
    )
