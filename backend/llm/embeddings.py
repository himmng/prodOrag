from typing import List

from openai import OpenAI

from ..config.models import AppConfig


class OpenAICompatibleEmbeddings:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client = OpenAI(
            base_url=str(config.embeddings.base_url),
            api_key=config.embeddings.api_key or "dummy",
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(
            model=self._config.embeddings.model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
