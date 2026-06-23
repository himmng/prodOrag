"""FastAPI app — production entry point for the RAG pipeline.

Lifespan-managed singletons (chunks, retrievers, LLM) — loaded ONCE at
startup, reused for every request. Avoids per-request reranker reload
(~20s) and chunks deserialization (~1s).
"""
import os
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
import json
import json as _json
from rag_pipeline.api.schemas import (
    EvalRetrievalRequest, EvalRetrievalResponse,
    ThresholdSweepRequest, ThresholdSweepResponse,
)
from rag_pipeline.eval.retrieval import threshold_sweep
# Add this if you don't already have it:
from rag_pipeline.eval.retrieval import evaluate_retriever, threshold_sweep
from rag_pipeline.eval.schema   import load_eval_set 
import time
import tempfile
from fastapi import UploadFile, File
from rag_pipeline.api.documents import DocumentStore, UploadedDoc
from rag_pipeline.api.schemas import (
    DocInfo, DocListResponse, DeleteResponse,
)
from rag_pipeline.parsers.docling import DoclingHybridParser
from fastapi.responses import StreamingResponse
from rag_pipeline.api.sse import sse_event
from rag_pipeline.generation import answer_stream
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, Response
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

# At the top, after the existing imports
from rag_pipeline.api.middleware.logging import (
    RequestLoggingMiddleware,
    install_json_logging,
)
# auth, ratelimits
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from rag_pipeline.api.middleware.auth import verify_api_key, auth_enabled
from rag_pipeline.api.middleware.rate_limit import limiter

# Process-wide singletons populated by the lifespan handler
_state: dict = {}


# Call once at module import — before app = FastAPI(...)
install_json_logging(level="INFO")

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
    hybrid_r = RerankedRetriever(ensemble, reranker, fetch_k=20, min_score=0.01)  # this is important, I changed min_score from 0.5 to None to now 0.01 as calibrated currently.
    hybrid_r_nofilter = RerankedRetriever(ensemble, reranker, fetch_k=20, min_score=None)
    _state["hybrid_r_nofilter"] = hybrid_r_nofilter
    llm      = get_llm()
    _state["doc_store"] = DocumentStore()
    _state["uploaded_parser"] = DoclingHybridParser()  # reuse one instance
    _state.update({
        "chunks":    chunks,
        "dense":     dense,
        "bm25":      bm25,
        "ensemble":  ensemble,
        "reranker":  reranker,
        "hybrid_r":  hybrid_r,
        "llm":       llm,
    })
    eval_path = cfg.PROJECT_ROOT / "eval" / "eval_set.json"
    _state["eval_set"] = load_eval_set(eval_path)
    log.info(f"Loaded {len(_state['eval_set'])} eval examples")
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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

log.info(f"API auth: {'ENABLED' if auth_enabled() else 'DISABLED (dev mode)'}")
app.add_middleware(RequestLoggingMiddleware)
# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    components = {
        "chunks":    "ok" if _state.get("chunks") else "missing",
        "retriever": "ok" if _state.get("hybrid_r") else "missing",
        "llm":       "ok" if _state.get("llm") else "missing",
    }
    try:
        import socket
        host = cfg.OLLAMA_HOST.replace("http://", "").replace("https://", "").split("/")[0]
        port = int(cfg.OLLAMA_PORT)
        with socket.create_connection((host, port), timeout=2):
            components["ollama"] = "ok"
    except Exception as e:
        components["ollama"] = f"unreachable: {str(e)[:60]}"

    all_ok = all(v == "ok" for v in components.values())
    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        components=components,
    )

def _build_retriever_for_request(req: AnswerRequest):
    """Build a retriever instance honoring any per-request overrides.
    Reuses the cached underlying components (dense, bm25, reranker)."""
    fetch_k = req.fetch_k or 20

    if req.retriever == "bm25":
        return _state["bm25"]
    if req.retriever == "dense":
        return _state["dense"]
    if req.retriever == "ensemble":
        return EnsembleRetriever([_state["dense"], _state["bm25"]], fetch_k=fetch_k)
    if req.retriever == "hybrid_reranked":
        ensemble = EnsembleRetriever([_state["dense"], _state["bm25"]], fetch_k=fetch_k)
        return RerankedRetriever(
            ensemble, _state["reranker"],
            fetch_k=fetch_k,
            min_score=req.min_score if req.min_score is not None else 0.01,
        )
    raise HTTPException(422, detail=f"Unknown retriever: {req.retriever}")

@app.post(
    "/answer",
    response_model=AnswerResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("60/minute")
def answer_route(
    request: Request,
    response: Response,        # ← add this
    req: AnswerRequest,
) -> AnswerResponse:
    """End-to-end RAG: retrieve → contextualize → LLM → return cited answer."""
    retriever_map = {
        "hybrid_reranked": _state["hybrid_r"],
        "dense":           _state["dense"],
        "bm25":            _state["bm25"],
        "ensemble":        _state["ensemble"],
    }
    retriever = _build_retriever_for_request(req)
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

def _to_doc_info(d: UploadedDoc) -> DocInfo:
    return DocInfo(
        doc_id=d.doc_id,
        filename=d.filename,
        char_count=d.char_count,
        uploaded_at=d.uploaded_at,
        section_refs=d.section_refs,
    )

@app.post(
    "/answer/stream",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("20/minute")
def answer_stream_route(
    request: Request,
    response: Response,        # ← add this
    req: AnswerRequest,
):
    """Streaming variant of /answer. Returns text/event-stream.

    Client receives:
      • citations event ASAP (~1.5s after request)
      • token events as the LLM generates
      • done event when complete
      • error event on any failure (terminal)
    """
    retriever_map = {
        "hybrid_reranked": _state["hybrid_r"],
        "dense":           _state["dense"],
        "bm25":            _state["bm25"],
        "ensemble":        _state["ensemble"],
    }
    retriever = _build_retriever_for_request(req)
    if retriever is None:
        raise HTTPException(422, detail=f"Unknown retriever: {req.retriever}")

    def event_generator():
        for evt in answer_stream(
            query=req.query,
            retriever=retriever,
            llm=_state["llm"],
            top_k=req.top_k,
        ):
            yield sse_event(evt["event"], evt["data"])

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",   # disable nginx/proxy buffering
        },
    )

@app.post(
    "/eval/retrieval",
    response_model=EvalRetrievalResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("5/minute")
def eval_retrieval_route(
    request: Request,
    response: Response,
    req: EvalRetrievalRequest,
) -> EvalRetrievalResponse:
    retriever_map = {
        "hybrid_reranked": _state["hybrid_r_nofilter"],
        "dense":           _state["dense"],
        "bm25":            _state["bm25"],
        "ensemble":        _state["ensemble"],
    }
    retriever = retriever_map.get(req.retriever)
    if retriever is None:
        raise HTTPException(422, detail=f"Unknown retriever: {req.retriever}")

    start = time.perf_counter()
    result = evaluate_retriever(
        retriever,
        _state["eval_set"],
        top_k=req.top_k,
    )
    return EvalRetrievalResponse(
        retriever=req.retriever,
        n_examples=result["n_positive"],     # only positives are scored
        top_k=req.top_k,
        metrics=result["overall"],
        by_difficulty=result["by_difficulty"],
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )


@app.post(
    "/eval/threshold-sweep",
    response_model=ThresholdSweepResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("2/minute")
def threshold_sweep_route(
    request: Request,
    response: Response,
    req: ThresholdSweepRequest,
) -> ThresholdSweepResponse:
    start = time.perf_counter()
    rows = threshold_sweep(
        _state["hybrid_r_nofilter"],
        _state["eval_set"],
        thresholds=req.thresholds,
        top_k=req.top_k,
    )
    recommended = max(rows, key=lambda r: r["f1"]) if rows else {}
    return ThresholdSweepResponse(
        top_k=req.top_k,
        rows=rows,
        recommended=recommended,
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )

@app.post("/documents/upload", response_model=DocInfo)
async def upload_document(file: UploadFile = File(...)) -> DocInfo:
    """Parse and ingest a case file into the in-memory doc store.
    
    Never touches the IPC ChromaDB collection.
    """
    if not file.filename:
        raise HTTPException(422, detail="filename required")

    suffix = (file.filename.rsplit(".", 1)[-1] or "").lower()
    if suffix not in {"pdf", "txt", "md"}:
        raise HTTPException(415, detail=f"Unsupported file type: .{suffix}")

    raw = await file.read()

    if suffix == "pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        chunks = _state["uploaded_parser"].parse(tmp_path)
        text = "\n\n".join(c.text for c in chunks)
    else:
        text = raw.decode("utf-8", errors="replace")

    if not text.strip():
        raise HTTPException(422, detail="Document had no extractable text")

    doc = _state["doc_store"].add(filename=file.filename, text=text)
    return _to_doc_info(doc)


@app.get("/documents", response_model=DocListResponse)
def list_documents() -> DocListResponse:
    docs = _state["doc_store"].list_all()
    return DocListResponse(
        documents=[_to_doc_info(d) for d in docs],
        total=len(docs),
    )


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str) -> DeleteResponse:
    if not _state["doc_store"].delete(doc_id):
        raise HTTPException(404, detail=f"Document not found: {doc_id}")
    return DeleteResponse(doc_id=doc_id, deleted=True)