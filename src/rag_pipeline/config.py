""" Configuration file for RAG pipeline. singleton + project-wide logger

PROJECT_ROOT is auto-detected by walking up from this file until pyproject.toml is found. All paths in `cfg` are absolute
 and derived from PROJECT_ROOT. Enviroment variables are loaded from .env at the project root.
 """

from __future__ import annotations # for forward references in type hints (e.g. in dataclasses)
import os
import logging # for logging
from pathlib import Path # for filesystem paths
from typing import ClassVar, Literal, Optional # for class variables in dataclasses
from pydantic_settings import BaseSettings, SettingsConfigDict # for configuration management with environment variable support
from pydantic import model_validator
def _find_project_root(marker: str = "pyproject.toml") -> Path:
    # 1. Explicit env override (containers, deployed environments)
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    # 2. Walk up from this file looking for the marker (editable installs)
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / marker).exists():
            return parent

    # 3. Last resort — current working directory
    return Path.cwd()

PROJECT_ROOT = _find_project_root()

class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    # Paths (unchanged)
    PROJECT_ROOT: ClassVar[Path] = PROJECT_ROOT
    DATA_RAW_DIR: ClassVar[Path] = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR: ClassVar[Path] = PROJECT_ROOT / "data" / "processed"
    CHROMA_PERSIST_DIR: ClassVar[Path] = PROJECT_ROOT / "chroma_db"

    EVAL_DIR:          ClassVar[Path] = PROJECT_ROOT / "eval"
    EVAL_SETS_DIR:     ClassVar[Path] = PROJECT_ROOT / "eval" / "eval_sets"
    EVAL_RESULTS_DIR:  ClassVar[Path] = PROJECT_ROOT / "eval" / "results"
    EVAL_SET_FILE: str = "eval_set_v3.json"   # override in .env

    # retrieval result subdirs
    RETRIEVAL_SUMMARY_DIR:      ClassVar[Path] = EVAL_RESULTS_DIR / "retrieval" / "summary"
    RETRIEVAL_PERQ_DIR:         ClassVar[Path] = EVAL_RESULTS_DIR / "retrieval" / "per_question"
    RETRIEVAL_PLOTS_DIR:        ClassVar[Path] = EVAL_RESULTS_DIR / "retrieval" / "plots"

    # ragas result subdirs
    RAGAS_SUMMARY_DIR:  ClassVar[Path] = EVAL_RESULTS_DIR / "ragas" / "summary"
    RAGAS_PERQ_DIR:     ClassVar[Path] = EVAL_RESULTS_DIR / "ragas" / "per_question"
    RAGAS_PLOTS_DIR:    ClassVar[Path] = EVAL_RESULTS_DIR / "ragas" / "plots"

    API_KEYS: str = ""
    RAG_CORPUS: str = "ipc_bns"

    # Provider switches — independent LLM and embedding providers.
    LLM_PROVIDER:       Literal["ollama", "azure", "openai", "gcp", "aws"] = "ollama"
    EMBEDDING_PROVIDER: Literal["ollama", "azure", "openai", "gcp", "aws"] = "ollama"

    # Ollama
    OLLAMA_HOST:            str = "http://localhost:11434"
    OLLAMA_MODEL:           str = "gemma-4-e4b:latest"
    OLLAMA_EMBEDDING_MODEL: str = "embeddinggemma:latest"

    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT:             Optional[str] = None
    AZURE_OPENAI_API_KEY:              Optional[str] = None
    AZURE_OPENAI_API_VERSION:          str = "2024-10-21"
    AZURE_OPENAI_DEPLOYMENT:           Optional[str] = None
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: Optional[str] = None

    # AWS Bedrock
    AWS_REGION:                     str = "us-east-1"
    AWS_BEDROCK_MODEL_ID:           Optional[str] = None
    AWS_BEDROCK_EMBEDDING_MODEL_ID: Optional[str] = None

    # GCP Vertex
    GCP_PROJECT_ID:             Optional[str] = None
    GCP_REGION:                 str = "us-central1"
    GCP_VERTEX_MODEL:           Optional[str] = None
    GCP_VERTEX_EMBEDDING_MODEL: Optional[str] = None

    # OpenAI direct
    OPENAI_API_KEY:         Optional[str] = None
    OPENAI_MODEL:           str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Chunking / retrieval / logging
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 5
    LOG_LEVEL: str = "INFO"
    
    # act tie-breaker
    PREFERRED_ACT: str = ""          # e.g. "BNS"; set per-corpus in .env
    ACT_TIE_DELTA: float = 0.05

    # ── Resolved model names (computed from the active provider) ──────────
    @property
    def MODEL(self) -> Optional[str]:
        return {
            "ollama": self.OLLAMA_MODEL,
            "azure":  self.AZURE_OPENAI_DEPLOYMENT,
            "openai": self.OPENAI_MODEL,
            "gcp":    self.GCP_VERTEX_MODEL,
            "aws":    self.AWS_BEDROCK_MODEL_ID,
        }[self.LLM_PROVIDER]

    @property
    def EVAL_SET_PATH(self) -> Path:
        return self.EVAL_SETS_DIR / self.EVAL_SET_FILE
    
    @property
    def EMBEDDING_MODEL(self) -> Optional[str]:
        return {
            "ollama": self.OLLAMA_EMBEDDING_MODEL,
            "azure":  self.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            "openai": self.OPENAI_EMBEDDING_MODEL,
            "gcp":    self.GCP_VERTEX_EMBEDDING_MODEL,
            "aws":    self.AWS_BEDROCK_EMBEDDING_MODEL_ID,
        }[self.EMBEDDING_PROVIDER]

    @property
    def llm_provider(self) -> str:
        return self.LLM_PROVIDER

    @property
    def embedding_provider(self) -> str:
        return self.EMBEDDING_PROVIDER

    @model_validator(mode="after")
    def _validate_active_provider_config(self):
        """Fail fast if the selected provider is missing required creds."""
        if self.LLM_PROVIDER == "azure":
            missing = [n for n, v in [
                ("AZURE_OPENAI_ENDPOINT", self.AZURE_OPENAI_ENDPOINT),
                ("AZURE_OPENAI_API_KEY", self.AZURE_OPENAI_API_KEY),
                ("AZURE_OPENAI_DEPLOYMENT", self.AZURE_OPENAI_DEPLOYMENT),
            ] if not v]
            if missing:
                raise ValueError(f"LLM_PROVIDER=azure requires: {', '.join(missing)}")
        # (add similar blocks for openai/gcp/aws if you want strict validation)
        return self

    def ensure_dirs(self) -> None:
        for p in [self.DATA_RAW_DIR, self.DATA_PROCESSED_DIR, self.CHROMA_PERSIST_DIR,
                  self.EVAL_DIR, self.EVAL_SETS_DIR, self.EVAL_RESULTS_DIR,
                  self.RETRIEVAL_SUMMARY_DIR, self.RETRIEVAL_PERQ_DIR, self.RETRIEVAL_PLOTS_DIR,
                  self.RAGAS_SUMMARY_DIR, self.RAGAS_PERQ_DIR, self.RAGAS_PLOTS_DIR]:
            p.mkdir(parents=True, exist_ok=True)

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
log.info(f"MODEL PROVIDER: {cfg.LLM_PROVIDER}, LLM model: {cfg.MODEL} | Embedding model: {cfg.EMBEDDING_MODEL}")