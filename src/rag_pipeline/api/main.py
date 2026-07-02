"""FastAPI app — production entry point for the RAG pipeline.

Lifespan-managed singletons (chunks, retrievers, LLM) — loaded ONCE at
startup, reused for every request. Avoids per-request reranker reload
(~20s) and chunks deserialization (~1s).
"""
import os
import re
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
from typing import Optional
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
from rag_pipeline.generation.context import build_context
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

_SECTION_MENTION = re.compile(
    r"(?:(IPC|BNS)\s*)?"
    r"(?:section|sec\.?|s\.?|§)?\s*"
    r"(\d+[A-Z]?)"
    r"(?:\s*(IPC|BNS))?",
    re.IGNORECASE,
)

def _concordance_context(query: str) -> str:
    """If the query mentions a section, look up its cross-reference and
    return a context string the LLM can cite. Empty string if no match."""
    concordance = _state.get("concordance")
    if concordance is None:
        return ""

    notes = []
    for m in _SECTION_MENTION.finditer(query):
        act_before, section, act_after = m.group(1), m.group(2), m.group(3)
        act = (act_before or act_after or "").upper()
        if not section:
            continue

        row = None
        if act == "IPC":
            row = concordance.lookup_ipc(section)
        elif act == "BNS":
            row = concordance.lookup_bns(section)
        else:
            # No act specified — try both
            row = concordance.lookup_ipc(section) or concordance.lookup_bns(section)

        if row:
            note = (
                f"CROSS-REFERENCE: IPC Section {row.ipc_section or '—'} "
                f"corresponds to BNS Section {row.bns_section or '—'} "
                f"(status: {row.status})."
            )
            if row.ipc_title:
                note += f" IPC title: {row.ipc_title}."
            if row.bns_title:
                note += f" BNS title: {row.bns_title}."
            notes.append(note)

    return "\n".join(notes)

def _retrieve_across_collections(
    query: str,
    acts: list[str],
    retriever_kind: str,
    top_k: int,
    min_score: Optional[float],
) -> list[tuple]:
    """Retrieve from one or more act collections, merge by score, return top_k.

    Each act has its own pre-built retrievers in _state["by_act"][act].
    When the user picks both IPC and BNS, we retrieve top_k from each then
    sort the union by reranker score.
    """
    retriever_attr = {
        "hybrid_reranked": "hybrid_r",
        "dense":           "dense",
        "bm25":            "bm25",
        "ensemble":        "ensemble",
    }.get(retriever_kind)

    if retriever_attr is None:
        raise HTTPException(422, detail=f"Unknown retriever: {retriever_kind}")

    if not acts:
        raise HTTPException(422, detail="At least one act required in `collections`")

    # When the user requests min_score=None on hybrid, swap in the no-filter variant
    if retriever_kind == "hybrid_reranked" and min_score is None:
        retriever_attr = "hybrid_r_nofilter"

    merged: list[tuple] = []
    per_collection_k = top_k if len(acts) == 1 else max(top_k, 5)

    for act in acts:
        setup = _state["by_act"].get(act)
        if setup is None:
            log.warning(f"Unknown act in request: {act}")
            continue
        retriever = setup[retriever_attr]
        results = retriever.retrieve(query, top_k=per_collection_k)
        merged.extend(results)

    merged.sort(key=lambda r: r[1], reverse=True)
    return merged[:top_k]


def _enrich_citation(doc, score: float, n: int) -> Citation:
    """Build a Citation from (Document, score), pulling fields from metadata."""
    meta = doc.metadata or {}
    return Citation(
        n=n,
        source_path=meta.get("source_path", ""),
        page_number=meta.get("page_number"),
        section_title=meta.get("section_title"),
        score=float(score),
        act=meta.get("act"),
        section=meta.get("section"),
        corresponds_to=meta.get("corresponds_to"),
        change_status=meta.get("change_status"),
    )


def _to_doc_info(d: UploadedDoc) -> DocInfo:
    return DocInfo(
        doc_id=d.doc_id,
        filename=d.filename,
        char_count=d.char_count,
        uploaded_at=d.uploaded_at,
        section_refs=d.section_refs,
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
        # Prefer a cached hybrid retriever (tests inject this) to avoid
        # constructing a RerankedRetriever when no reranker is available.
        if _state.get("hybrid_r") is not None:
            return _state["hybrid_r"]

        ensemble = EnsembleRetriever([_state["dense"], _state["bm25"]], fetch_k=fetch_k)
        # If the process doesn't have a reranker (e.g. tests), fall back
        # to the ensemble rather than constructing a broken reranker wrapper.
        if _state.get("reranker") is None:
            return ensemble

        return RerankedRetriever(
            ensemble, _state["reranker"],
            fetch_k=fetch_k,
            min_score=req.min_score if req.min_score is not None else 0.01,
        )
    raise HTTPException(422, detail=f"Unknown retriever: {req.retriever}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("API starting up — loading pipeline...")

    # 1. Shared reranker FIRST (used by all collections)
    shared_reranker = Reranker()

    # 2. Per-collection retrievers
    collection_setups = {}
    for act, collection in [("IPC", "IPC_Corpus"), ("BNS", "BNS_Corpus")]:
        log.info(f"Loading {act} retrievers ({collection})...")
        
        chunks_path = cfg.PROJECT_ROOT / "data" / "processed" / f"{act.lower()}_chunks.json"
        if not chunks_path.exists():
            raise RuntimeError(f"Chunks file missing: {chunks_path}. Run `rag-ingest --corpus ipc_bns`.")
        
        import json
        with open(chunks_path) as f:
            raw_chunks = json.load(f)
        from rag_pipeline.schemas import StatuteChunk
        chunks = [StatuteChunk(**c) for c in raw_chunks]

        dense = DenseRetriever(collection_name=collection)
        bm25  = BM25Retriever(chunks)
        ensemble = EnsembleRetriever([dense, bm25], fetch_k=20)
        
        collection_setups[act] = {
            "dense":             dense,
            "bm25":              bm25,
            "ensemble":          ensemble,
            "hybrid_r":          RerankedRetriever(ensemble, shared_reranker, fetch_k=20, min_score=0.01),
            "hybrid_r_nofilter": RerankedRetriever(ensemble, shared_reranker, fetch_k=20, min_score=None),
            "n_chunks":          len(chunks),
        }
        log.info(f"  {act}: {len(chunks)} chunks loaded")

    # 3. Concordance
    from rag_pipeline.corpus.concordance import Concordance
    conc_path = cfg.PROJECT_ROOT / "data" / "processed" / "concordance.json"
    concordance = Concordance.from_json(conc_path) if conc_path.exists() else None
    if concordance:
        log.info("Loaded concordance for cross-references")

    _state.update({
        "by_act":      collection_setups,
        "reranker":    shared_reranker,
        "llm":         get_llm(),
        "concordance": concordance,
    })

    # 4. Eval set (optional, may not exist for fresh corpus)
    eval_path = cfg.PROJECT_ROOT / "eval" / "eval_set.json"
    if eval_path.exists():
        _state["eval_set"] = load_eval_set(eval_path)
        log.info(f"Loaded {len(_state['eval_set'])} eval examples")

    log.info("API ready")
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


# ── /answer (non-streaming) ──────────────────────────────────────────

@app.post(
    "/answer",
    response_model=AnswerResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("60/minute")
def answer_route(
    request: Request,
    response: Response,
    req: AnswerRequest,
) -> AnswerResponse:
    """End-to-end RAG: retrieve → contextualize → LLM → return cited answer."""
    start = time.perf_counter()

    docs_with_scores = _retrieve_across_collections(
        query=req.query,
        acts=req.collections,
        retriever_kind=req.retriever,
        top_k=req.top_k,
        min_score=req.min_score,
    )

    if not docs_with_scores:
        return AnswerResponse(
            question=req.query,
            answer="I don't have that information in the provided documents.",
            citations=[],
            retriever=req.retriever,
            latency_ms=(time.perf_counter() - start) * 1000.0,
        )

    context = build_context(docs_with_scores)

    # Inject cross-reference mapping if the query mentions a section
    xref = _concordance_context(req.query)
    if xref:
        context = f"{xref}\n\n{context}"

    prompt = (
        f"You are a legal assistant grounded ONLY in the Indian Penal Code / "
        f"Bharatiya Nyaya Sanhita excerpts below. Use the CROSS-REFERENCE lines "
        f"to answer questions about section correspondences. Cite sources by [n]. "
        f"If the answer is not in the excerpts, say so.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {req.query}\n\nANSWER:"
    )
    llm_resp = _state["llm"].invoke(prompt)
    answer_text = getattr(llm_resp, "content", None) or str(llm_resp)

    citations = [
        _enrich_citation(doc, score, i + 1)
        for i, (doc, score) in enumerate(docs_with_scores)
    ]

    return AnswerResponse(
        question=req.query,
        answer=answer_text,
        citations=citations,
        retriever=req.retriever,
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )


# ── /answer/stream (SSE) ─────────────────────────────────────────────

@app.post(
    "/answer/stream",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("20/minute")
def answer_stream_route(
    request: Request,
    response: Response,
    req: AnswerRequest,
):
    """Streaming variant of /answer. Returns text/event-stream."""
    if req.retriever not in {"hybrid_reranked", "dense", "bm25", "ensemble"}:
        raise HTTPException(422, detail=f"Unknown retriever: {req.retriever}")

    def event_generator():
        start = time.perf_counter()
        try:
            docs_with_scores = _retrieve_across_collections(
                query=req.query,
                acts=req.collections,
                retriever_kind=req.retriever,
                top_k=req.top_k,
                min_score=req.min_score,
            )
        except HTTPException as e:
            yield sse_event("error", {"stage": "retrieve", "message": e.detail})
            return
        except Exception as e:
            yield sse_event("error", {"stage": "retrieve", "message": str(e)})
            return

        if not docs_with_scores:
            yield sse_event("citations", {"citations": []})
            yield sse_event("token", {"text": "I don't have that information in the provided documents."})
            yield sse_event("done", {
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "retriever":  req.retriever,
                "refused":    True,
            })
            return

        citations = [
            _enrich_citation(doc, score, i + 1).model_dump()
            for i, (doc, score) in enumerate(docs_with_scores)
        ]
        yield sse_event("citations", {"citations": citations})

        context = build_context(docs_with_scores)

        # Inject cross-reference mapping if the query mentions a section
        xref = _concordance_context(req.query)
        if xref:
            context = f"{xref}\n\n{context}"

        prompt = (
            f"You are a legal assistant grounded ONLY in the Indian Penal Code / "
            f"Bharatiya Nyaya Sanhita excerpts below. Use the CROSS-REFERENCE lines "
            f"to answer questions about section correspondences. Cite sources by [n]. "
            f"If the answer is not in the excerpts, say so.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {req.query}\n\nANSWER:"
        )
        try:
            for chunk in _state["llm"].stream(prompt):
                if isinstance(chunk, str):
                    text = chunk
                elif hasattr(chunk, "content"):
                    text = chunk.content or ""
                else:
                    text = ""
                if text:
                    yield sse_event("token", {"text": text})
        except Exception as e:
            yield sse_event("error", {"stage": "llm", "message": str(e)})
            return

        yield sse_event("done", {
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "retriever":  req.retriever,
            "refused":    False,
        })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── /eval/* (point at IPC by default for now) ────────────────────────

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
    if "eval_set" not in _state:
        raise HTTPException(503, detail="Eval set not loaded. Run Stage E eval rebuild first.")

    # Eval against IPC for now; Stage E rebuilds a combined eval set
    setup = _state["by_act"].get("IPC")
    retriever_attr = {
        "hybrid_reranked": "hybrid_r_nofilter",
        "dense":           "dense",
        "bm25":            "bm25",
        "ensemble":        "ensemble",
    }.get(req.retriever)
    if retriever_attr is None:
        raise HTTPException(422, detail=f"Unknown retriever: {req.retriever}")
    retriever = setup[retriever_attr]

    start = time.perf_counter()
    result = evaluate_retriever(retriever, _state["eval_set"], top_k=req.top_k)
    return EvalRetrievalResponse(
        retriever=req.retriever,
        n_examples=result["n_positive"],
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
    if "eval_set" not in _state:
        raise HTTPException(503, detail="Eval set not loaded. Run Stage E eval rebuild first.")

    start = time.perf_counter()
    rows = threshold_sweep(
        _state["by_act"]["IPC"]["hybrid_r_nofilter"],
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


# ── /documents/* (uploaded case files — unchanged) ───────────────────

@app.post("/documents/upload", response_model=DocInfo)
async def upload_document(file: UploadFile = File(...)) -> DocInfo:
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