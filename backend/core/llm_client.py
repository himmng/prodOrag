from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple

import httpx

from backend.core.config import AppConfig, LLMConfig


class LLMClient(ABC):
    @abstractmethod
    async def chat(self, messages: List[Dict[str, Any]]) -> str:
        ...


class OpenAICompatibleLLMClient(LLMClient):
    def __init__(self, base_url: str, api_key: Optional[str], model: str):
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        self._base_url = base
        self._api_key = api_key
        self._model = model
        self._client = httpx.AsyncClient(timeout=30.0)

    async def chat(self, messages: List[Dict[str, Any]]) -> str:
        headers: Dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "messages": messages,
        }

        url = f"{self._base_url}/v1/chat/completions"
        try:
            resp = await self._client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise exc

        if resp.status_code != 200:
            return f"LLM request failed with status {resp.status_code}"

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return str(data)


def _resolve_llm_config(cfg: LLMConfig) -> Tuple[str, Optional[str], str]:
    provider = (cfg.provider or "").lower()
    base_url = cfg.base_url
    api_key = cfg.api_key
    model = cfg.model

    if provider == "ollama":
        if not base_url:
            base_url = "http://localhost:11434"
        if not model:
            model = "llama3"
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


def create_llm_client(config: AppConfig) -> LLMClient:
    llm_cfg = config.llm
    base_url, api_key, model = _resolve_llm_config(llm_cfg)
    return OpenAICompatibleLLMClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
    )
