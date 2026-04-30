import json
from pathlib import Path
from typing import Optional

from .models import AppConfig, OpenAIEndpointConfig


DEFAULT_CONFIG_PATH = Path("./data/config.json")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        default = AppConfig(
            llm=OpenAIEndpointConfig(
                provider_name="ollama",
                base_url="http://localhost:11434/v1",
                api_key=None,
                model="llama3",
            ),
            embeddings=OpenAIEndpointConfig(
                provider_name="ollama",
                base_url="http://localhost:11434/v1",
                api_key=None,
                model="nomic-embed-text",
            ),
        )
        save_config(default, path)
        return default

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return AppConfig.model_validate(data)


def save_config(config: AppConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config.model_dump(), f, indent=2)
