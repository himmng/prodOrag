"""
LLM + embedding 

The single source of truth for *which* model the pipeline runs against.
Everything else in the codebase calls `get_llm()` or `get_embedding()` - no module ever instantiates a model directly.

Provider is selected via `cfg.MODEL_PROVIDER`. Required env vars are validated lazily, and provider SDKs are imported only when their 
provider is active (so you don't need langchain unless you use the provider.)

"""

from __future__ import annotations # for forward references in type hints (e.g. in dataclasses)
from functools import lru_cache # for caching the provider instances
from typing import TYPE_CHECKING # for type checking without circular imports

from rag_pipeline.config import cfg, log # project-wide config and logger

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel


# Public API - call these, cached as singletons.

@lru_cache(maxsize=1)
def get_llm(provider: str | None = None, deployment: str | None = None) -> "BaseChatModel":
    """Return the chat LLM. Defaults to the active provider from config;
    pass provider/deployment to override (for per-role models like RAGAS judges)."""
    provider = provider or cfg.llm_provider
    log.info(f"Instantiating LLM for provider: {provider}" + (f" (deployment={deployment})" if deployment else ""))
    dispatch = {
        "ollama": _ollama_llm,
        "azure": _azure_llm,
        "openai": _openai_llm,
        "gcp": _gcp_llm,
        "aws": _aws_llm,
    }
    if provider not in dispatch:
        raise ValueError(f"Unsupported provider: {provider}")
    return dispatch[provider](deployment) if provider == "azure" else dispatch[provider]()

@lru_cache(maxsize=1)
def get_embeddings() -> "Embeddings":
    """Return the embedding model for the active provider."""
    provider = cfg.embedding_provider
    log.info(f"Instantiating embedding model for provider: {provider}")
    dispatch = {
        "ollama": _ollama_emb,
        "azure": _azure_emb,
        "openai": _openai_emb,
        "gcp": _gcp_emb,
        "aws": _aws_emb,
    }
    if provider not in dispatch:
        raise ValueError(f"Unsupported MODEL_PROVIDER: {provider}")
    return dispatch[provider]()

# Helpers

def _require(value, name: str) -> None:
    if not value:
        raise ValueError(
            f"{name} is required for provider {cfg.LLM_PROVIDER}."
            f"Please set it via environment variable or in your .env file."
        )
    
# ollama (local)
def _ollama_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=cfg.OLLAMA_MODEL,
        base_url=cfg.OLLAMA_HOST,
        temperature=cfg.OLLAMA_MODEL_TEMPERATURE,

    )

def _ollama_emb():
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(
        model=cfg.OLLAMA_EMBEDDING_MODEL,
        base_url=cfg.OLLAMA_HOST,
        temperature=cfg.OLLAMA_EMBEDDING_TEMPERATURE,
    )

# azure openai
def _azure_llm(deployment: str | None = None):
    from langchain_openai import AzureChatOpenAI
    _require(cfg.AZURE_OPENAI_ENDPOINT, "AZURE_OPENAI_ENDPOINT")
    _require(cfg.AZURE_OPENAI_API_KEY, "AZURE_OPENAI_API_KEY")
    dep = deployment or cfg.AZURE_OPENAI_DEPLOYMENT
    _require(dep, "AZURE_OPENAI_DEPLOYMENT")
    return AzureChatOpenAI(
        azure_endpoint=cfg.AZURE_OPENAI_ENDPOINT,
        azure_deployment=dep,
        api_version=cfg.AZURE_OPENAI_API_VERSION,
        api_key=cfg.AZURE_OPENAI_API_KEY,
        temperature=cfg.AZURE_OPENAI_DEPLOYMENT_TEMPERATURE,
    )
def _azure_emb():
    from langchain_openai import AzureOpenAIEmbeddings
    _require(cfg.AZURE_OPENAI_ENDPOINT, "AZURE_OPENAI_ENDPOINT")
    _require(cfg.AZURE_OPENAI_API_KEY, "AZURE_OPENAI_API_KEY")
    _require(cfg.AZURE_OPENAI_EMBEDDING_DEPLOYMENT, "AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    return AzureOpenAIEmbeddings(
        azure_endpoint=cfg.AZURE_OPENAI_ENDPOINT,
        azure_deployment=cfg.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        api_version=cfg.AZURE_OPENAI_API_VERSION,     # was azure_api_version
        api_key=cfg.AZURE_OPENAI_API_KEY,             # was azure_api_key
        temperature=cfg.AZURE_OPENAI_EMBEDDING_TEMPERATURE,  # was azure_embedding_temperature
    )

# openai direct (optional)
def _openai_llm():
    from langchain_openai import ChatOpenAI
    _require(cfg.OPENAI_API_KEY, "OPENAI_API_KEY")
    return ChatOpenAI(
        model=cfg.OPENAI_MODEL,
        openai_api_key=cfg.OPENAI_API_KEY,
        temperature=cfg.OPENAI_TEMPERATURE,
    )

def _openai_emb():
    from langchain_openai import OpenAIEmbeddings
    _require(cfg.OPENAI_API_KEY, "OPENAI_API_KEY")
    return OpenAIEmbeddings(
        model=cfg.OPENAI_EMBEDDING_MODEL,
        openai_api_key=cfg.OPENAI_API_KEY,
        temperature=cfg.OPENAI_EMBEDDING_TEMPERATURE,
    )

# AWS Bedrock

def _aws_llm():
    from langchain_aws import ChatBedrock
    _require(cfg.AWS_BEDROCK_MODEL_ID, "AWS_BEDROCK_MODEL_ID")
    return ChatBedrock(
        model_id=cfg.AWS_BEDROCK_MODEL_ID,
        region_name=cfg.AWS_REGION,
        temperature=cfg.AWS_BEDROCK_TEMPERATURE,
    )

def _aws_emb():
    from langchain_aws import BedrockEmbeddings
    _require(cfg.AWS_BEDROCK_EMBEDDING_MODEL_ID, "AWS_BEDROCK_EMBEDDING_MODEL_ID")
    return BedrockEmbeddings(
        model_id=cfg.AWS_BEDROCK_EMBEDDING_MODEL_ID,
        region_name=cfg.AWS_REGION,
        temperature=cfg.AWS_BEDROCK_EMBEDDING_TEMPERATURE,
    )


# GCP Vertex AI

def _gcp_llm():
    from langchain_google_vertexai import ChatVertexAI
    _require(cfg.GCP_PROJECT_ID, "GCP_PROJECT_ID")
    _require(cfg.GCP_VERTEX_MODEL, "GCP_VERTEX_MODEL")
    return ChatVertexAI(
        model=cfg.GCP_VERTEX_MODEL,
        project=cfg.GCP_PROJECT_ID,
        location=cfg.GCP_REGION,
        temperature=cfg.GCP_VERTEX_TEMPERATURE,
    )

def _gcp_emb():
    from langchain_google_vertexai import VertexAIEmbeddings
    _require(cfg.GCP_PROJECT_ID, "GCP_PROJECT_ID")
    _require(cfg.GCP_VERTEX_EMBEDDING_MODEL, "GCP_VERTEX_EMBEDDING_MODEL")
    return VertexAIEmbeddings(
        model=cfg.GCP_VERTEX_EMBEDDING_MODEL,
        project=cfg.GCP_PROJECT_ID,
        location=cfg.GCP_REGION,
        temperature=cfg.GCP_VERTEX_EMBEDDING_TEMPERATURE,
    )
