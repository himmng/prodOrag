from typing import Iterable, List, Dict, Any

from openai import OpenAI

from ..config.models import AppConfig
from .embeddings import OpenAICompatibleEmbeddings


class OpenAICompatibleClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._llm_client = OpenAI(
            base_url=str(config.llm.base_url),
            api_key=config.llm.api_key or "dummy",
        )
        self.embeddings = OpenAICompatibleEmbeddings(config)

    def chat_completion_stream(self, messages: List[Dict[str, str]]) -> Iterable[str]:
        stream = self._llm_client.chat.completions.create(
            model=self._config.llm.model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
