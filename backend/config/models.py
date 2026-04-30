from pydantic import BaseModel, Field, HttpUrl
from typing import Optional


class OpenAIEndpointConfig(BaseModel):
    provider_name: str = Field(description="Human label, e.g. ollama, openai, lmstudio")
    base_url: HttpUrl
    api_key: Optional[str] = None
    model: str


class AppConfig(BaseModel):
    llm: OpenAIEndpointConfig
    embeddings: OpenAIEndpointConfig
    vector_store_path: str = "./data/vector_store/chroma"
    file_store_path: str = "./data/files"
    sqlite_path: str = "./data/meta.db"
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
