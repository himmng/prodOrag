"""End-to-end RAG answer pipeline.

Composes: retrieve -> contextualize -> LLM invocation -> response dict.
Single canonical entry point for asking a question. Works with any
BaseRetriever implementation (dense, hybrid, reranked, multi-query, ...).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from rag_pipeline.config import cfg
from rag_pipeline.generation.context import build_context
from rag_pipeline.prompts import get_prompt

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from rag_pipeline.retrievers.base import BaseRetriever


REFUSAL_TEXT = "I don't have that information in the provided documents."


def answer(
    query: str,
    retriever: "BaseRetriever",
    llm: "BaseChatModel",
    top_k: int | None = None,
    system_prompt_name: str = "system",
) -> dict:
    """Run the full RAG pipeline.

    Args:
        query:              user question
        retriever:          any BaseRetriever implementation
        llm:                a LangChain chat model
        top_k:              # of chunks to retrieve (defaults to cfg.TOP_K)
        system_prompt_name: which YAML prompt to use as the system message

    Returns:
        dict with keys: question, answer, citations, retriever (class name)
    """
    top_k = top_k or cfg.TOP_K
    results = retriever.retrieve(query, top_k=top_k)

    # Early-exit refusal if retrieval returned nothing — saves an LLM call
    if not results:
        return {
            "question":  query,
            "answer":    REFUSAL_TEXT,
            "citations": [],
            "retriever": type(retriever).__name__,
        }

    context_block, citations = build_context(results)
    system_prompt = get_prompt(system_prompt_name)
    user_msg = f"CONTEXT:\n{context_block}\n\nQUESTION: {query}\n\nANSWER:"

    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg),
    ])

    return {
        "question":  query,
        "answer":    resp.content,
        "citations": citations,
        "retriever": type(retriever).__name__,
    }