from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    model: str = "llama3"


class EmbeddingsConfig(BaseModel):
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    model: str = "nomic-embed-text"


class StorageConfig(BaseModel):
    vector_dir: str = "./data/vector_store"
    docs_dir: str = "./data/documents"


class RagConfig(BaseModel):
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64


class AppConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    storage: StorageConfig = StorageConfig()
    rag: RagConfig = RagConfig()


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(path: Optional[str] = None) -> AppConfig:
    config_path = Path(path) if path else _CONFIG_PATH
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = AppConfig()
        save_config(config, str(config_path))
        return config

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return AppConfig(**data)


def save_config(config: AppConfig, path: Optional[str] = None) -> None:
    global _cached_config
    config_path = Path(path) if path else _CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.dict(), f, sort_keys=False)
    _cached_config = config


_cached_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _cached_config
    if _cached_config is None:
        _cached_config = load_config()
    return _cached_config
