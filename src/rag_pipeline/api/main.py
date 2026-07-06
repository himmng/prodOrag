"""FastAPI app — production entry point for the RAG pipeline.

Lifespan-managed singletons (chunks, retrievers, LLM) — loaded ONCE at
startup, reused for every request. Avoids per-request reranker reload
(~20s) and chunks deserialization (~1s).
"""
import os
import re
from functools import lru_cache
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
import json
import json as _json
from fastapi import Query
from fastapi.responses import Response as FastAPIResponse
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
    CaseAnswerRequest, CaseAnswerResponse, CaseExcerptOut,
    ContextSnippet,
)
from rag_pipeline.parsers.docling import DoclingHybridParser
from rag_pipeline.schemas import RagChunk
from fastapi import Header
from fastapi.responses import StreamingResponse
from rag_pipeline.api.sse import sse_event
from rag_pipeline.generation import answer_stream
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from rag_pipeline.api.schemas import (
    AnswerRequest, AnswerResponse, Citation, HealthResponse, CrossReference,
    MetaResponse, CrossRefMeta,
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

@lru_cache(maxsize=8)
def _section_mention_re(act_a: str, act_b: str) -> "re.Pattern":
    """Build a section-mention regex from the corpus's two concordance act labels."""
    alt = f"{re.escape(act_a)}|{re.escape(act_b)}"
    return re.compile(
        rf"(?:({alt})\s*)?"
        r"(?:section|sec\.?|s\.?|§)?\s*"
        r"(\d+[A-Z]?)"
        rf"(?:\s*({alt}))?",
        re.IGNORECASE,
    )


def _concordance_context(query: str) -> tuple[str, list[dict], list]:
    """Detect section mentions, look up cross-references (config-driven labels).

    Returns (prompt_text, resolved, rows):
      - prompt_text: CROSS-REFERENCE lines for the LLM
      - resolved:    [{act, section}] pointing at real section texts to cite
      - rows:        matched ConcordanceRow objects (for table-location citation)
    """
    concordance = _state.get("concordance")
    corpus = _state.get("corpus")
    if concordance is None or corpus is None or not corpus.has_concordance:
        return "", [], []

    cc = corpus.concordance
    act_a, act_b = cc.act_a, cc.act_b
    rx = _section_mention_re(act_a, act_b)

    notes, resolved, rows = [], [], []
    for m in rx.finditer(query):
        act_before, section, act_after = m.group(1), m.group(2), m.group(3)
        act = (act_before or act_after or "").upper()
        if not section:
            continue

        if act == act_a.upper():
            row = concordance.lookup_a(section)
        elif act == act_b.upper():
            row = concordance.lookup_b(section)
        else:
            row = concordance.lookup_a(section) or concordance.lookup_b(section)

        if row:
            note = (
                f"CROSS-REFERENCE: {act_a} Section {row.section_a or '—'} "
                f"corresponds to {act_b} Section {row.section_b or '—'} "
                f"(status: {row.status})."
            )
            if row.title_a:
                note += f" {act_a} title: {row.title_a}."
            if row.title_b:
                note += f" {act_b} title: {row.title_b}."
            notes.append(note)
            rows.append(row)
            if row.section_a:
                resolved.append({"act": act_a, "section": row.section_a})
            if row.section_b:
                resolved.append({"act": act_b, "section": row.section_b})

    return "\n".join(notes), resolved, rows


# Prose commentary similarity tops out ~0.3 with embeddinggemma → low floor.
CONTEXT_FLOOR = 0.15


def _retrieve_context(query: str, top_k: int = 3) -> list:
    """Retrieve interpretive commentary (committee report / SOR). Empty if disabled."""
    ctx = _state.get("context_retriever")
    if ctx is None:
        return []
    hits = ctx.retrieve(query, top_k=top_k)
    return [(d, s) for d, s in hits if s >= CONTEXT_FLOOR]


def _context_block(hits: list) -> str:
    """Format commentary hits as a labeled, clearly-non-authoritative prompt block."""
    if not hits:
        return ""
    lines = []
    for i, (d, s) in enumerate(hits, 1):
        src = d.metadata.get("display_name") or d.metadata.get("doc_type", "commentary")
        page = d.metadata.get("page_number")
        loc = f", p.{page}" if page and page != -1 else ""
        lines.append(f"[C{i}] ({src}{loc}) {d.page_content}")
    return (
        "LEGISLATIVE CONTEXT / COMMITTEE COMMENTARY (interpretive background — "
        "NOT statute; do not cite as binding law, use only to explain intent):\n"
        + "\n\n".join(lines)
    )


def _context_snippets(hits: list) -> list["ContextSnippet"]:
    out = []
    for d, s in hits:
        page = d.metadata.get("page_number")
        out.append(ContextSnippet(
            text=d.page_content,
            doc_type=d.metadata.get("doc_type", "commentary"),
            source=d.metadata.get("display_name") or d.metadata.get("doc_type", "commentary"),
            page_number=page if page not in (None, -1) else None,
            score=s,
        ))
    return out


def _fetch_section_chunk(act: str, section: str):
    """Return (Document, score) for an exact act+section, or None. Vectorstore-agnostic."""
    c = _state.get("section_index", {}).get((act, section))
    if c is None:
        return None
    from langchain_core.documents import Document
    return (Document(page_content=c.text, metadata=c.to_langchain_metadata()), 1.0)


def _concordance_citation(row, n: int) -> Citation:
    """Turn a ConcordanceRow into a citation showing its table location."""
    corpus = _state.get("corpus")
    cc = corpus.concordance if (corpus and corpus.has_concordance) else None
    act_a = cc.act_a if cc else "A"
    act_b = cc.act_b if cc else "B"
    label = cc.pdf_label if cc else "CONCORDANCE"
    src = cc.pdf_path.name if cc else "concordance.pdf"
    return Citation(
        n=n,
        source_path=src,
        page_number=row.page_number,
        section_title=(
            f"Concordance Row {row.row_index or '?'} · "
            f"{act_b} §{row.section_b or '—'} ↔ {act_a} §{row.section_a or '—'}"
        ),
        score=1.0,
        act=label,
        section=f"row {row.row_index or '?'}",
        corresponds_to=(row.section_b if row.section_a else row.section_a),
        change_status=row.status,
    )
def _corpus_display() -> str:
    corpus = _state.get("corpus")
    return corpus.display_name if corpus else "the provided corpus"


def _resolve_acts(requested: list[str]) -> list[str]:
    """Empty request → all corpus acts; otherwise validate against the corpus."""
    corpus = _state.get("corpus")
    all_acts = corpus.acts if corpus else []
    if not requested:
        return list(all_acts)
    unknown = [a for a in requested if a not in all_acts]
    if unknown:
        raise HTTPException(422, detail=f"Unknown act(s) {unknown}; available: {all_acts}")
    return requested


def _answer_prompt(query: str, context: str, ctx_block: str) -> str:
    """Grounded-QA prompt, corpus name pulled from config (no IPC/BNS hardcoding)."""
    return (
        f"You are an assistant grounded ONLY in the {_corpus_display()} excerpts "
        f"below. Use any CROSS-REFERENCE lines to answer questions about section "
        f"correspondences. Cite sources by [n]. Treat any LEGISLATIVE CONTEXT as "
        f"non-binding background only. If the answer is not in the excerpts, say so.\n\n"
        f"CONTEXT:\n{context}"
        + (f"\n\n{ctx_block}" if ctx_block else "")
        + f"\n\nQUESTION: {query}\n\nANSWER:"
    )


def _build_cross_reference(rows, resolved) -> Optional[CrossReference]:
    """Assemble the CrossReference response block using config act labels."""
    if not rows:
        return None
    corpus = _state.get("corpus")
    cc = corpus.concordance if (corpus and corpus.has_concordance) else None
    act_a = cc.act_a if cc else None
    act_b = cc.act_b if cc else None
    row = rows[0]
    src_cit = tgt_cit = None
    for r in resolved:
        hit = _fetch_section_chunk(r["act"], r["section"])
        if hit:
            cit = _enrich_citation(hit[0], hit[1], 0)
            if r["act"] == act_a:
                src_cit = cit
            elif r["act"] == act_b:
                tgt_cit = cit
    return CrossReference(
        concordance_row=row.row_index,
        source_act=act_a,
        target_act=act_b,
        source_section=row.section_a,
        target_section=row.section_b,
        page_number=row.page_number,
        status=row.status,
        source_citation=src_cit,
        target_citation=tgt_cit,
    )


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

    # 0. Load the active corpus config — the single source of truth for the
    #    serving layer. Everything below is driven by it (no IPC/BNS hardcoding).
    from rag_pipeline.corpus import load_corpus
    corpus = load_corpus(cfg.RAG_CORPUS)
    log.info(f"Active corpus: {corpus.name} — {corpus.display_name} (acts: {corpus.acts})")

    # 1. Shared reranker FIRST (used by all collections)
    shared_reranker = Reranker()

    # 2. Per-collection retrievers — one per corpus source
    from rag_pipeline.schemas import StatuteChunk
    collection_setups = {}
    for source in corpus.sources:
        act, collection = source.act, source.collection
        log.info(f"Loading {act} retrievers ({collection})...")

        chunks_path = source.chunks_output
        if not chunks_path.exists():
            raise RuntimeError(
                f"Chunks file missing: {chunks_path}. Run `rag-ingest --corpus {corpus.name}`."
            )

        import json
        with open(chunks_path) as f:
            raw_chunks = json.load(f)
        chunks = [StatuteChunk(**c) for c in raw_chunks]
        section_index = _state.setdefault("section_index", {})
        for c in chunks:
            section_index.setdefault((act, c.section), c)
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

    # 3. Concordance / cross-reference — optional, only if the corpus declares it
    concordance = None
    if corpus.has_concordance:
        from rag_pipeline.corpus.concordance import Concordance
        conc_path = corpus.concordance.output_json
        concordance = Concordance.from_json(conc_path) if conc_path.exists() else None
        if concordance:
            log.info(f"Loaded concordance for cross-references "
                     f"({corpus.concordance.act_a}↔{corpus.concordance.act_b})")

    # 4. Interpretive context layer — optional, from corpus.context_sources
    context_retriever = None
    for coll in corpus.context_collections:
        try:
            ctx = DenseRetriever(collection_name=coll)
            if ctx.vectorstore._collection.count() > 0:
                context_retriever = ctx
                log.info(f"Loaded context layer: {coll} ({ctx.vectorstore._collection.count()} vectors)")
                break
        except Exception as e:
            log.warning(f"Context layer '{coll}' unavailable: {e}")

    _state.update({
        "corpus":            corpus,
        "by_act":            collection_setups,
        "reranker":          shared_reranker,
        "llm":               get_llm(),
        "concordance":       concordance,
        "doc_store":         DocumentStore(),        # session-scoped case uploads (isolated)
        "uploaded_parser":   DoclingHybridParser(),  # parses uploaded case files
        "context_retriever": context_retriever,      # commentary (non-authoritative)
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
    description="Corpus-agnostic hybrid retrieval-augmented generation API",
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
    by_act = _state.get("by_act", {})
    any_retriever = any(s.get("hybrid_r") for s in by_act.values())
    components = {
        "chunks":    "ok" if by_act else "missing",
        "retriever": "ok" if any_retriever else "missing",
        "llm":       "ok" if _state.get("llm") else "missing",
    }
    # Ollama reachability check — OLLAMA_HOST already includes host:port
    import httpx
    ollama_url = cfg.OLLAMA_HOST.rstrip("/")
    try:
        r = httpx.get(f"{ollama_url}/api/tags", timeout=2.0)
        ollama_status = "ok" if r.status_code == 200 else f"http {r.status_code}"
    except Exception as e:
        ollama_status = f"unreachable: {e}"

    all_ok = all(v == "ok" for v in components.values())
    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        components=components,
    )


@app.get("/meta", response_model=MetaResponse)
def meta() -> MetaResponse:
    """Active-corpus metadata so clients render without hardcoding act labels."""
    corpus = _state.get("corpus")
    if corpus is None:
        raise HTTPException(503, detail="Corpus not loaded")
    xref = None
    if corpus.has_concordance:
        cc = corpus.concordance
        xref = CrossRefMeta(source_act=cc.act_a, target_act=cc.act_b, pdf_label=cc.pdf_label)
    return MetaResponse(
        corpus=corpus.name,
        display_name=corpus.display_name,
        acts=corpus.acts,
        pdf_acts=list(corpus.pdf_map().keys()),
        context_enabled=_state.get("context_retriever") is not None,
        cross_reference=xref,
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
        acts=_resolve_acts(req.collections),
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
    MAIN_CITATION_FLOOR = 0.55
    docs_with_scores = [
        (d, s) for d, s in docs_with_scores if s >= MAIN_CITATION_FLOOR
    ]
    context = build_context(docs_with_scores)

    # Cross-reference detection
    xref, resolved, rows = _concordance_context(req.query)
    if xref:
        context = f"{xref}\n\n{context}"

    # Interpretive commentary (opt-in for general /answer)
    ctx_hits = _retrieve_context(req.query) if req.include_context else []
    ctx_block = _context_block(ctx_hits)

    # Priority citations: concordance row(s) + real section texts
    priority = []
    n = 1
    for row in rows:
        priority.append(_concordance_citation(row, n)); n += 1

    priority_docs = []
    for r in resolved:
        hit = _fetch_section_chunk(r["act"], r["section"])
        if hit:
            priority_docs.append(hit)

    prompt = _answer_prompt(req.query, context, ctx_block)
    llm_resp = _state["llm"].invoke(prompt)
    answer_text = getattr(llm_resp, "content", None) or str(llm_resp)

    # Assemble: concordance rows → section texts → deduped semantic
    seen_ids = {d.metadata.get("chunk_id") for d, _ in priority_docs}
    section_citations = [_enrich_citation(d, s, 0) for d, s in priority_docs]
    semantic_citations = [
        _enrich_citation(d, s, 0) for d, s in docs_with_scores
        if d.metadata.get("chunk_id") not in seen_ids
    ]
    keep = priority + section_citations + semantic_citations[: max(0, req.top_k)]
    # Semantic citations ONLY — capped at top_k
    citations = [
        _enrich_citation(doc, score, i + 1)
        for i, (doc, score) in enumerate(docs_with_scores[:req.top_k])
    ]

    # Cross-reference lives in its own field, not mixed into citations
    cross_ref = _build_cross_reference(rows, resolved)

    return AnswerResponse(
        question=req.query,
        answer=answer_text,
        citations=citations,
        cross_reference=cross_ref,
        context=_context_snippets(ctx_hits),
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
    """Streaming variant of /answer. Returns text/event-stream.

    Events:
      • cross_reference — concordance mapping + both section texts (if query names a section)
      • citations       — semantic RAG results only, capped at top_k
      • token           — one per LLM chunk
      • done            — terminal frame with latency
      • error           — terminal frame on failure
    """
    if req.retriever not in {"hybrid_reranked", "dense", "bm25", "ensemble"}:
        raise HTTPException(422, detail=f"Unknown retriever: {req.retriever}")

    def event_generator():
        start = time.perf_counter()

        # 1. Retrieve
        try:
            docs_with_scores = _retrieve_across_collections(
                query=req.query,
                acts=_resolve_acts(req.collections),
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
        MAIN_CITATION_FLOOR = 0.55
        docs_with_scores = [
            (d, s) for d, s in docs_with_scores if s >= MAIN_CITATION_FLOOR
        ]
        # 2. Cross-reference detection (fires even if semantic retrieval is weak)
        xref, resolved, rows = _concordance_context(req.query)

        # 3. Refusal — only if BOTH semantic and concordance are empty
        if not docs_with_scores and not rows:
            yield sse_event("cross_reference", {"cross_reference": None})
            yield sse_event("citations", {"citations": []})
            yield sse_event("token", {"text": "I don't have that information in the provided documents."})
            yield sse_event("done", {
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                "retriever":  req.retriever,
                "refused":    True,
            })
            return

        # 4. Build context
        context = build_context(docs_with_scores) if docs_with_scores else ""
        if xref:
            context = f"{xref}\n\n{context}"

        # 4b. Interpretive commentary (opt-in)
        ctx_hits = _retrieve_context(req.query) if req.include_context else []
        ctx_block = _context_block(ctx_hits)

        # 5. Cross-reference block (separate from citations)
        cross_ref = _build_cross_reference(rows, resolved)
        yield sse_event(
            "cross_reference",
            {"cross_reference": cross_ref.model_dump() if cross_ref else None},
        )

        # 6. Semantic citations ONLY — capped at top_k
        citations = [
            _enrich_citation(doc, score, i + 1).model_dump()
            for i, (doc, score) in enumerate(docs_with_scores[: req.top_k])
        ]
        yield sse_event("citations", {"citations": citations})

        # 6b. Interpretive commentary as its own event
        yield sse_event(
            "context",
            {"context": [s.model_dump() for s in _context_snippets(ctx_hits)]},
        )

        # 7. Build prompt + stream LLM tokens
        prompt = _answer_prompt(req.query, context, ctx_block)
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

        # 8. Done
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
# ── /eval/* (evaluate against the corpus's first act by default) ─────────

def _default_eval_setup():
    """The retriever setup to evaluate against — first act of the active corpus."""
    by_act = _state.get("by_act", {})
    if not by_act:
        raise HTTPException(503, detail="No corpus collections loaded.")
    return next(iter(by_act.values()))



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

    # Eval against the corpus's first act; Stage E rebuilds a combined eval set
    setup = _default_eval_setup()
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
        _default_eval_setup()["hybrid_r_nofilter"],
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


# ── /documents/* (uploaded case files — session-scoped, isolated vectors) ──
#
# Every op requires an X-Session-Id token. Uploads are keyed by that token and
# embedded into a PER-CASE isolated Chroma collection (case_{session}_{doc});
# they never touch IPC_Corpus/BNS_Corpus and are invisible across sessions.

def _text_to_chunks(text: str, filename: str, window: int = 800) -> list[RagChunk]:
    """Naive paragraph-aware chunker for txt/md uploads (~`window` chars each)."""
    chunks, buf = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) + 2 > window and buf:
            chunks.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    return [
        RagChunk(text=t, source_path=filename, source_format="txt",
                 element_type="text-window")
        for t in chunks
    ]


@app.post("/documents/upload", response_model=DocInfo)
async def upload_document(
    file: UploadFile = File(...),
    x_session_id: str = Header(..., description="Per-client session token"),
) -> DocInfo:
    if not file.filename:
        raise HTTPException(422, detail="filename required")

    suffix = (file.filename.rsplit(".", 1)[-1] or "").lower()
    if suffix not in {"pdf", "txt", "md"}:
        raise HTTPException(415, detail=f"Unsupported file type: .{suffix}")

    raw = await file.read()
    if suffix == "pdf":
        from pathlib import Path
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        chunks = _state["uploaded_parser"].parse(Path(tmp_path))
    else:
        text = raw.decode("utf-8", errors="replace")
        chunks = _text_to_chunks(text, file.filename)

    if not chunks or not any(c.text.strip() for c in chunks):
        raise HTTPException(422, detail="Document had no extractable text")

    doc = _state["doc_store"].add(
        session_id=x_session_id, filename=file.filename, chunks=chunks
    )
    return _to_doc_info(doc)


@app.get("/documents", response_model=DocListResponse)
def list_documents(
    x_session_id: str = Header(..., description="Per-client session token"),
) -> DocListResponse:
    docs = _state["doc_store"].list_all(x_session_id)
    return DocListResponse(
        documents=[_to_doc_info(d) for d in docs],
        total=len(docs),
    )

@app.get("/documents/page-image", dependencies=[Depends(verify_api_key)])
def page_image(
    act: str = Query(..., description="Act label or concordance label from the active corpus"),
    page: int = Query(..., ge=1),
):
    """Render one page of a corpus PDF to PNG for inline preview."""
    import pdfplumber, io

    corpus = _state.get("corpus")
    pdf_map = corpus.pdf_map() if corpus else {}
    pdf_path = pdf_map.get(act)
    if not pdf_path or not pdf_path.exists():
        raise HTTPException(404, detail=f"PDF not found for '{act}'; available: {list(pdf_map)}")

    with pdfplumber.open(pdf_path) as pdf:
        if page > len(pdf.pages):
            raise HTTPException(404, detail=f"Page {page} out of range")
        pil_img = pdf.pages[page - 1].to_image(resolution=120).original
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return FastAPIResponse(content=buf.getvalue(), media_type="image/png")

@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(
    doc_id: str,
    x_session_id: str = Header(..., description="Per-client session token"),
) -> DeleteResponse:
    if not _state["doc_store"].delete(x_session_id, doc_id):
        raise HTTPException(404, detail=f"Document not found: {doc_id}")
    return DeleteResponse(doc_id=doc_id, deleted=True)


# ── /answer/case — interpret an uploaded case against IPC/BNS ──────────────

@app.post(
    "/answer/case",
    response_model=CaseAnswerResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("30/minute")
def answer_case_route(
    request: Request,
    response: Response,
    req: CaseAnswerRequest,
    x_session_id: str = Header(..., description="Per-client session token"),
) -> CaseAnswerResponse:
    """Interpret an uploaded case: which IPC/BNS sections apply, why, and how.

    The case text is CONTEXT only — it lives in its own isolated collection and
    is never mixed into the corpus. We retrieve (a) the most relevant case
    excerpts, (b) relevant IPC/BNS statutory sections, (c) any IPC↔BNS
    cross-references implied by the question or sections named in the case.
    """
    start = time.perf_counter()

    doc = _state["doc_store"].get(x_session_id, req.doc_id)
    if doc is None:
        raise HTTPException(404, detail=f"Case not found for this session: {req.doc_id}")

    # 1. Retrieve from the case's OWN isolated collection
    case_hits = _state["doc_store"].search_case(
        x_session_id, req.doc_id, req.question, top_k=req.case_k
    )

    # 2. Retrieve relevant IPC/BNS statute sections (existing corpus pipeline)
    docs_with_scores = _retrieve_across_collections(
        query=req.question,
        acts=_resolve_acts(req.collections),
        retriever_kind="hybrid_reranked",
        top_k=req.top_k,
        min_score=None,
    )
    MAIN_CITATION_FLOOR = 0.55
    docs_with_scores = [(d, s) for d, s in docs_with_scores if s >= MAIN_CITATION_FLOOR]
    corpus_context = build_context(docs_with_scores) if docs_with_scores else ""

    # 3. Cross-reference: sections named in the question OR in the case text
    xref_query = f"{req.question} " + " ".join(doc.section_refs)
    xref, resolved, rows = _concordance_context(xref_query)

    # 3b. Interpretive commentary (committee report / SOR) — on by default here
    ctx_hits = _retrieve_context(req.question) if req.include_context else []
    ctx_block = _context_block(ctx_hits)

    # 4. Build the interpretation prompt
    case_context = "\n\n".join(
        f"[CASE {i + 1}] {ex.text}" for i, ex in enumerate(case_hits)
    ) or "(no case excerpts retrieved)"
    xref_block = f"\n\nCROSS-REFERENCE:\n{xref}" if xref else ""

    prompt = (
        "You are an assistant analysing a REAL case/document a user uploaded. "
        f"Ground your answer ONLY in (a) the CASE EXCERPTS, (b) the STATUTORY "
        f"EXCERPTS from {_corpus_display()}, (c) any CROSS-REFERENCE lines, and "
        "(d) any LEGISLATIVE CONTEXT (commentary — background only, NOT binding "
        "law). Explain WHICH sections apply, WHY they apply (tie the facts of the "
        "case to the elements of each provision), and HOW to interpret the case, "
        "drawing on the legislative context for intent where relevant. Cite "
        "sources by [n]. Do NOT invent sections; if the excerpts do not support "
        "a point, say so.\n\n"
        f"CASE EXCERPTS:\n{case_context}\n\n"
        f"STATUTORY EXCERPTS:\n{corpus_context or '(none retrieved)'}"
        f"{xref_block}"
        + (f"\n\n{ctx_block}" if ctx_block else "")
        + f"\n\nQUESTION: {req.question}\n\nANSWER:"
    )
    llm_resp = _state["llm"].invoke(prompt)
    answer_text = getattr(llm_resp, "content", None) or str(llm_resp)

    # 5. Assemble response
    citations = [
        _enrich_citation(d, s, i + 1)
        for i, (d, s) in enumerate(docs_with_scores[: req.top_k])
    ]
    case_excerpts = [
        CaseExcerptOut(
            text=ex.text, score=ex.score,
            page_number=ex.page_number, section_title=ex.section_title,
        )
        for ex in case_hits
    ]

    cross_ref = _build_cross_reference(rows, resolved)

    return CaseAnswerResponse(
        question=req.question,
        answer=answer_text,
        citations=citations,
        case_excerpts=case_excerpts,
        cross_reference=cross_ref,
        context=_context_snippets(ctx_hits),
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )