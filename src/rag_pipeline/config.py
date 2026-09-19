""" Configuration file for RAG pipeline. singleton + project-wide logger

PROJECT_ROOT is auto-detected by walking up from this file until pyproject.toml is found. All paths in `cfg` are absolute
 and derived from PROJECT_ROOT. Enviroment variables are loaded from .env at the project root.
 """

from __future__ import annotations # for forward references in type hints (e.g. in dataclasses)
import os
import logging # for logging
import logging.handlers # for RotatingFileHandler type annotations
from datetime import datetime # for per-session log filenames
from pathlib import Path # for filesystem paths
from typing import ClassVar, Literal, Optional # for class variables in dataclasses
from pydantic_settings import BaseSettings, SettingsConfigDict # for configuration management with environment variable support
from pydantic import SecretStr, model_validator
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
    LOGS_DIR: ClassVar[Path] = PROJECT_ROOT / "logs"
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

    # RAGAS role models (blank = fall back to the global azure deployment)
    RAGAS_GEN_DEPLOYMENT:    str = ""      # generator (answers being judged)
    RAGAS_JUDGE_DEPLOYMENT: str = ""      # all judges
    RAGAS_GEN_TEMPERATURE:         float = 1.0   # generator temperature
    RAGAS_JUDGE_TEMPERATURE:       float = 1.0   # judge temperature

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
    OLLAMA_ENDPOINT:            str = "http://localhost:11434"
    OLLAMA_LLM_MODEL:           str = "gemma-4-e4b:latest"
    OLLAMA_LLM_TEMPERATURE: float = 1.0
    OLLAMA_EMBEDDING_MODEL: str = "embeddinggemma:latest"
    OLLAMA_EMBEDDING_TEMPERATURE: float = 1.0

    # Azure OpenAI
    AZURE_FOUNDRY_ENDPOINT:             Optional[str] = None
    AZURE_FOUNDRY_API_KEY:              Optional[SecretStr] = None
    AZURE_FOUNDRY_SDK_API_VERSION:          str = "2024-10-21"
    AZURE_FOUNDRY_LLM_MODEL:           Optional[str] = None
    AZURE_FOUNDRY_EMBEDDING_MODEL: Optional[str] = None
    AZURE_FOUNDRY_EMBEDDING_TEMPERATURE:      float = 1.0
    AZURE_FOUNDRY_LLM_TEMPERATURE:      float = 1.0

    # AWS Bedrock
    AWS_REGION:                     str = "us-east-1"
    AWS_BEDROCK_ENDPOINT:           Optional[str] = None
    AWS_BEDROCK_EMBEDDING_MODEL:    Optional[str] = None
    AWS_BEDROCK_LLM_MODEL:              Optional[str] = None
    AWS_BEDROCK_API_KEY:            Optional[SecretStr] = None
    AWS_BEDROCK_LLM_TEMPERATURE:      float = 1.0
    AWS_BEDROCK_EMBEDDING_TEMPERATURE: float = 1.0

    # GCP Vertex
    GCP_PROJECT_ID:             Optional[str] = None
    GCP_REGION:                 str = "us-central1"
    GCP_VERTEX_MODEL:           Optional[str] = None
    GCP_VERTEX_EMBEDDING_MODEL: Optional[str] = None
    GCP_VERTEX_TEMPERATURE:     float = 1.0
    GCP_VERTEX_EMBEDDING_TEMPERATURE: float = 1.0

    # OpenAI direct
    OPENAI_API_KEY:         Optional[SecretStr] = None
    OPENAI_LLM_MODEL:           str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_LLM_TEMPERATURE:         float = 1.0
    OPENAI_EMBEDDING_TEMPERATURE: float = 1.0

    # Reranker (cross-encoder, always local/HuggingFace regardless of LLM_PROVIDER)
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"

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
            "ollama": self.OLLAMA_LLM_MODEL,
            "azure":  self.AZURE_FOUNDRY_LLM_MODEL,
            "openai": self.OPENAI_LLM_MODEL,
            "gcp":    self.GCP_VERTEX_MODEL,
            "aws":    self.AWS_BEDROCK_LLM_MODEL,
        }[self.LLM_PROVIDER]

    @property
    def EVAL_SET_PATH(self) -> Path:
        return self.EVAL_SETS_DIR / self.EVAL_SET_FILE
    
    @property
    def EMBEDDING_MODEL(self) -> Optional[str]:
        return {
            "ollama": self.OLLAMA_EMBEDDING_MODEL,
            "azure":  self.AZURE_FOUNDRY_EMBEDDING_MODEL,
            "openai": self.OPENAI_EMBEDDING_MODEL,
            "gcp":    self.GCP_VERTEX_EMBEDDING_MODEL,
            "aws":    self.AWS_BEDROCK_EMBEDDING_MODEL,
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
        if self.LLM_PROVIDER == "ollama":
            missing = [n for n, v in [
                ("OLLAMA_ENDPOINT", self.OLLAMA_ENDPOINT),
                ("OLLAMA_LLM_MODEL", self.OLLAMA_LLM_MODEL),
            ] if not v]
        elif self.LLM_PROVIDER == "azure":
            missing = [n for n, v in [
                ("AZURE_FOUNDRY_ENDPOINT", self.AZURE_FOUNDRY_ENDPOINT),
                ("AZURE_FOUNDRY_LLM_MODEL_API_KEY", self.AZURE_FOUNDRY_API_KEY),
                ("AZURE_FOUNDRY_LLM_MODEL", self.AZURE_FOUNDRY_LLM_MODEL),
                ("AZURE_FOUNDRY_LLM_TEMPERATURE", self.AZURE_FOUNDRY_LLM_TEMPERATURE),
            ] if not v]
        elif self.LLM_PROVIDER == "aws":
            missing = [n for n, v in [
                ("AWS_REGION", self.AWS_REGION),
                ("AWS_BEDROCK_LLM_MODEL", self.AWS_BEDROCK_LLM_MODEL),
                ("AWS_BEDROCK_API_KEY", self.AWS_BEDROCK_API_KEY),
                ("AWS_BEDROCK_LLM_MODEL_TEMPERATURE", self.AWS_BEDROCK_LLM_TEMPERATURE),
                ("AWS_BEDROCK_ENDPOINT", self.AWS_BEDROCK_ENDPOINT),
            ] if not v]
        elif self.LLM_PROVIDER == "gcp":
            missing = [n for n, v in [
                ("GCP_PROJECT_ID", self.GCP_PROJECT_ID),
                ("GCP_REGION", self.GCP_REGION),
                ("GCP_VERTEX_MODEL", self.GCP_VERTEX_MODEL),
            ] if not v]
        elif self.LLM_PROVIDER == "openai":
            missing = [n for n, v in [
                ("OPENAI_API_KEY", self.OPENAI_API_KEY),
                ("OPENAI_LLM_MODEL", self.OPENAI_LLM_MODEL),
            ] if not v]
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.LLM_PROVIDER}")

        if missing:
            raise ValueError(f"LLM_PROVIDER={self.LLM_PROVIDER} requires: {', '.join(missing)}")
        return self

    def ensure_dirs(self) -> None:
        for p in [self.DATA_RAW_DIR, self.DATA_PROCESSED_DIR, self.CHROMA_PERSIST_DIR,
                  self.EVAL_DIR, self.EVAL_SETS_DIR, self.EVAL_RESULTS_DIR,
                  self.RETRIEVAL_SUMMARY_DIR, self.RETRIEVAL_PERQ_DIR, self.RETRIEVAL_PLOTS_DIR,
                  self.RAGAS_SUMMARY_DIR, self.RAGAS_PERQ_DIR, self.RAGAS_PLOTS_DIR,
                  self.LOGS_DIR]:
            p.mkdir(parents=True, exist_ok=True)

cfg = Config()
cfg.ensure_dirs()

# Path of the current session's log file — read by install_json_logging()
# so the API server's JSON logging still lands in this same file.
SESSION_LOG_PATH: Path | None = None


def _session_log_path(corpus_name: str) -> Path:
    """logs/protorag_<corpus_name>_DDMMYYYY_HH_MM_SS.log"""
    stamp = datetime.now().strftime("%d%m%Y_%H_%M_%S")
    return cfg.LOGS_DIR / f"protorag_{corpus_name}_{stamp}.log"


def _make_file_handler(corpus_name: str, fmt: logging.Formatter) -> "logging.handlers.RotatingFileHandler":
    global SESSION_LOG_PATH
    from logging.handlers import RotatingFileHandler

    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_LOG_PATH = _session_log_path(corpus_name)
    handler = RotatingFileHandler(
        SESSION_LOG_PATH,
        maxBytes=10 * 1024 * 1024,   # 10 MB per file
        backupCount=5,               # keep 5 rotations (~50 MB total)
        encoding="utf-8",
    )
    handler.setFormatter(fmt)
    return handler


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure the project-wide 'rag' logger. Structured JSON, console + one
    timestamped file per session. Idempotent."""
    from rag_pipeline.logging_utils import JSONLogFormatter, quiet_noisy_loggers

    log = logging.getLogger("rag")
    if log.handlers:
        return log  # already configured
    log.setLevel(level or cfg.LOG_LEVEL)

    fmt = JSONLogFormatter()

    # Console
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    log.addHandler(console)

    # One fresh file per run: logs/protorag_<corpus>_<DDMMYYYY_HH_MM_SS>.log
    log.addHandler(_make_file_handler(cfg.RAG_CORPUS, fmt))

    log.propagate = False
    quiet_noisy_loggers()
    return log


def reconfigure_file_log(corpus_name: str) -> None:
    """Swap the session's log file to match the actual corpus being processed.

    Call this once the real corpus name is known (e.g. after argparse), so the
    file matches `--corpus`/document set rather than the RAG_CORPUS default
    picked at import time.
    """
    from rag_pipeline.logging_utils import JSONLogFormatter

    log = logging.getLogger("rag")
    fmt = JSONLogFormatter()
    for handler in [h for h in log.handlers if isinstance(h, logging.FileHandler)]:
        log.removeHandler(handler)
        handler.close()
    log.addHandler(_make_file_handler(corpus_name, fmt))

log = setup_logging()
log.info(f"Project root: {cfg.PROJECT_ROOT}")
log.info(f"MODEL PROVIDER: {cfg.LLM_PROVIDER}, LLM model: {cfg.MODEL} | Embedding model: {cfg.EMBEDDING_MODEL}")