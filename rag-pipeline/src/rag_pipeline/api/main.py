"""FastAPI app — production entry point for the RAG pipeline.

Lifespan-managed singletons (chunks, retrievers, LLM) — loaded ONCE at
startup, reused for every request. Avoids per-request reranker reload
(~20s) and chunks deserialization (~1s).
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from rag_pipeline.api.schemas import (
    AnswerRequest, AnswerResponse, Citation, HealthResponse,
)
from rag_pipeline.config import cfg, log
from rag_pipeline.generation import answer
from rag_pipeline.parsers import load_chunks_cache
from rag_pipeline.providers import get_llm
from rag_pipeline.retrievers import (
    BM25Retriever, DenseRetriever, EnsembleRetriever,
    Reranker, RerankedRetriever,
)


# Process-wide singletons populated by the lifespan handler
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy objects ONCE at startup."""
    log.info("API starting up — loading pipeline...")
    chunks_path = cfg.PROJECT_ROOT / "data" / "processed" / "phase1_chunks.json"
    if not chunks_path.exists():
        raise RuntimeError(
            f"Chunks cache missing: {chunks_path}. "
            f"Run the ingest notebook first."
        )

    chunks   = load_chunks_cache(chunks_path)
    dense    = DenseRetriever(collection_name="IPC_Corpus")
    bm25     = BM25Retriever(chunks)
    ensemble = EnsembleRetriever([dense, bm25], fetch_k=20)
    reranker = Reranker()
    hybrid_r = RerankedRetriever(ensemble, reranker, fetch_k=20, min_score=0.5)
    llm      = get_llm()

    _state.update({
        "chunks":    chunks,
        "dense":     dense,
        "bm25":      bm25,
        "ensemble":  ensemble,
        "reranker":  reranker,
        "hybrid_r":  hybrid_r,
        "llm":       llm,
    })
    log.info(f"API ready — {len(chunks)} chunks loaded")
    yield
    log.info("API shutting down")
    _state.clear()


app = FastAPI(
    title="RAG Pipeline API",
    description="Hybrid retrieval-augmented generation over the IPC corpus",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Readiness probe: returns 'healthy' iff every dep is reachable."""
    components = {
        "chunks":    "ok" if _state.get("chunks") else "missing",
        "retriever": "ok" if _state.get("hybrid_r") else "missing",
        "llm":       "ok" if _state.get("llm") else "missing",
    }
    try:
        _state["llm"].invoke("ping")
        components["ollama"] = "ok"
    except Exception as e:
        components["ollama"] = f"unreachable: {str(e)[:80]}"

    all_ok = all(v == "ok" for v in components.values())
    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        components=components,
    )


@app.post("/answer", response_model=AnswerResponse)
def answer_route(req: AnswerRequest) -> AnswerResponse:
    """End-to-end RAG: retrieve → contextualize → LLM → return cited answer."""
    retriever_map = {
        "hybrid_reranked": _state["hybrid_r"],
        "dense":           _state["dense"],
        "bm25":            _state["bm25"],
        "ensemble":        _state["ensemble"],
    }
    retriever = retriever_map.get(req.retriever)
    if retriever is None:
        raise HTTPException(422, detail=f"Unknown retriever: {req.retriever}")

    start = time.perf_counter()
    response = answer(
        query=req.query,
        retriever=retriever,
        llm=_state["llm"],
        top_k=req.top_k,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0

    return AnswerResponse(
        question=response["question"],
        answer=response["answer"],
        citations=[Citation(**c) for c in response["citations"]],
        retriever=response["retriever"],
        latency_ms=latency_ms,
    )