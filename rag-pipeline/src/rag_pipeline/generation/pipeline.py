"""End-to-end RAG answer pipeline.

Composes: retrieve -> contextualize -> LLM invocation -> response dict.
Single canonical entry point for asking a question. Works with any
BaseRetriever implementation (dense, hybrid, reranked, multi-query, ...).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

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

def answer_stream(
    query: str,
    retriever,
    llm,
    top_k: int = 5,
) -> Iterator[dict]:
    """Streaming variant of answer().

    Yields events (as plain dicts) in this order:
      1. {"event": "citations", "data": {...}}      — fires after retrieval
      2. {"event": "token",     "data": {"text": ...}}  — one per LLM chunk
      3. {"event": "done",      "data": {...}}      — terminal frame
      4. {"event": "error",     "data": {...}}      — on failure (terminal)

    The API layer is responsible for wire-format (SSE). This function
    stays transport-agnostic.
    """
    import time
    from rag_pipeline.generation.context import build_context

    start = time.perf_counter()

    try:
        # 1. Retrieve
        docs_with_scores = retriever.retrieve(query, top_k=top_k)
    except Exception as e:
        yield {"event": "error", "data": {"stage": "retrieve", "message": str(e)}}
        return

    # 2. Refusal path — no docs survived the filter
    if not docs_with_scores:
        yield {"event": "citations", "data": {"citations": []}}
        refusal = "I could not find relevant information in the IPC corpus to answer that."
        yield {"event": "token", "data": {"text": refusal}}
        yield {"event": "done", "data": {
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "retriever":  getattr(retriever, "name", "unknown"),
            "refused":    True,
        }}
        return

    # 3. Emit citations immediately (before LLM call)
    citations = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        meta = doc.metadata or {}
        citations.append({
            "n":             i,
            "source_path":   meta.get("source_path", ""),
            "page_number":   meta.get("page_number"),
            "section_title": meta.get("section_title"),
            "score":         float(score),
        })
    yield {"event": "citations", "data": {"citations": citations}}

    # 4. Build prompt + stream LLM
    docs = [d for d, _ in docs_with_scores]
    context = build_context(docs_with_scores)
    prompt = (
        f"You are a legal assistant grounded ONLY in the IPC corpus excerpts below. "
        f"Cite sources by their [n] marker. If the answer is not in the excerpts, say so.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    )

    try:
        for chunk in llm.stream(prompt):
            # Handle multiple chunk shapes:
            #   - str (OllamaLLM)
            #   - AIMessageChunk with .content (ChatOllama)
            if isinstance(chunk, str):
                text = chunk
            elif hasattr(chunk, "content"):
                text = chunk.content or ""
            else:
                text = ""
            if text:                       # skip empty/heartbeat chunks
                yield {"event": "token", "data": {"text": text}}
    except Exception as e:
        yield {"event": "error", "data": {"stage": "llm", "message": str(e)}}
        return

    # 5. Done
    yield {"event": "done", "data": {
        "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        "retriever":  getattr(retriever, "name", "unknown"),
        "refused":    False,
    }}