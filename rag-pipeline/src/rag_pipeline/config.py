""" Configuration file for RAG pipeline. singleton + project-wide logger

PROJECT_ROOT is auto-detected by walking up from this file until pyproject.toml is found. All paths in `cfg` are absolute
 and derived from PROJECT_ROOT. Enviroment variables are loaded from .env at the project root.
 """

from __future__ import annotations # for forward references in type hints (e.g. in dataclasses)

import logging # for logging
from pathlib import Path # for filesystem paths
from typing import ClassVar, Literal, Optional # for class variables in dataclasses
from pydantic_settings import BaseSettings, SettingsConfigDict # for configuration management with environment variable support

def _find_project_root(marker: str = "pyproject.toml") -> Path:
    """Walk up from this file's location until `marker` is found."""
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find {marker} in any parent of {here}")

PROJECT_ROOT = _find_project_root()

class Config(BaseSettings):
    """Project configuration. Override any field via .env or environment variable."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="allow", # allow extra fields in .env without erroring
    )

    # Paths (ClassVar -> not user-overrideable, derived from PROJECT_ROOT)
    PROJECT_ROOT: ClassVar[Path] = PROJECT_ROOT
    DATA_RAW_DIR: ClassVar[Path] = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR: ClassVar[Path] = PROJECT_ROOT / "data" / "processed"
    CHROMA_PERSIST_DIR: ClassVar[Path] = PROJECT_ROOT / "chroma_db"
    EVAL_DIR: ClassVar[Path] = PROJECT_ROOT / "eval"
    EVAL_SET_PATH: ClassVar[Path] = EVAL_DIR / "eval_set.json"
    EVAL_RESULTS_DIR: ClassVar[Path] = EVAL_DIR / "results"

    # provider switch
    MODEL_PROVIDER: Literal["ollama", "azure", "openai", "gcp", "aws"] = "ollama"

    # ollama (env-overrideable)
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma-4-e4b:latest"
    OLLAMA_EMBEDDING_MODEL: str = "embeddinggemma:latest"

    # azure openai
    AZURE_OPENAI_ENDPOINT:             Optional[str] = None
    AZURE_OPENAI_API_KEY:              Optional[str] = None
    AZURE_OPENAI_API_VERSION:          str = "2024-10-21"
    AZURE_OPENAI_DEPLOYMENT:           Optional[str] = None   # chat deployment name
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: Optional[str] = None   # embeddings deployment name

    # ── AWS Bedrock ──────────────────────────────────────────────────────
    AWS_REGION:                    str = "us-east-1"
    AWS_BEDROCK_MODEL_ID:           Optional[str] = None      # e.g. anthropic.claude-3-5-sonnet-20241022-v2:0
    AWS_BEDROCK_EMBEDDING_MODEL_ID: Optional[str] = None      # e.g. amazon.titan-embed-text-v2:0

    # ── GCP Vertex AI ────────────────────────────────────────────────────
    GCP_PROJECT_ID:              Optional[str] = None
    GCP_REGION:                  str = "us-central1"
    GCP_VERTEX_MODEL:            Optional[str] = None         # e.g. gemini-2.0-pro
    GCP_VERTEX_EMBEDDING_MODEL:  Optional[str] = None         # e.g. text-embedding-005

    # ── OpenAI (direct, optional) ────────────────────────────────────────
    OPENAI_API_KEY:         Optional[str] = None
    OPENAI_MODEL:           str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120

    # retrieval
    TOP_K: int = 5

    # logging
    LOG_LEVEL: str = "INFO"

    def ensure_dirs(self) -> None:
        """Create output directories if they don't exist."""
        for p in [self.DATA_RAW_DIR, self.DATA_PROCESSED_DIR, self.CHROMA_PERSIST_DIR, self.EVAL_DIR, self.EVAL_RESULTS_DIR]:
            p.mkdir(parents=True, exist_ok=True)

# Singletons

cfg = Config()
cfg.ensure_dirs()

def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure the project-wide 'rag' logger. Idempotent."""
    log = logging.getLogger("rag")
    if log.handlers:
        return log # already configured
    log.setLevel(level or cfg.LOG_LEVEL)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)-7s | %(name)s - %(message)s"))
    log.addHandler(handler)
    log.propagate = False # prevent double logging if root logger is also configured
    return log

log = setup_logging()
log.info(f"Project root: {cfg.PROJECT_ROOT}")
log.info(f"Ollama: {cfg.OLLAMA_HOST}, LLM model: {cfg.OLLAMA_MODEL} | Embedding model: {cfg.OLLAMA_EMBEDDING_MODEL}")