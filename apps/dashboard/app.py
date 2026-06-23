"""IPC Legal RAG — Streamlit dashboard.

Run:
    streamlit run apps/dashboard/app.py
"""

import json
import time
from typing import Iterator

import requests
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="IPC Legal RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main .block-container { padding-top: 1.5rem; max-width: 100%; }
        div[data-testid="stMetricValue"] { font-size: 1.25rem; }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7f8fa 0%, #ffffff 100%);
            width: 320px !important;
        }
        .sample-btn button { text-align: left !important; height: 3.5rem; font-size: 0.9rem; }
        .monitor-panel {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 0.75rem;
        }
        .monitor-event {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.78rem;
            padding: 0.15rem 0;
            color: #495057;
        }
        .flow-stage {
            display: inline-block;
            background: #e7f1ff;
            border: 1px solid #b6d4fe;
            color: #084298;
            padding: 0.4rem 0.7rem;
            margin: 0 0.25rem;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 500;
            white-space: nowrap;
        }
        .flow-arrow {
            color: #adb5bd;
            font-size: 1.2rem;
            margin: 0 0.1rem;
        }
        .flow-container {
            overflow-x: auto;
            padding: 0.6rem 0;
            text-align: center;
            white-space: nowrap;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ────────────────────────────────────────────────────

DEFAULTS = {
    "messages":       [],
    "api_url":        "http://localhost:4040",
    "api_key":        "dev-key-123",
    "retriever":      "hybrid_reranked",
    "top_k":          5,
    "fetch_k":        20,
    "min_score":      0.01,
    "use_streaming":  True,
    "events":         [],   # live event log for the monitor
    "req_count":      0,
    "total_tokens":   0,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def log_event(kind: str, msg: str) -> None:
    """Append an event to the live monitor log."""
    st.session_state.events.append({
        "ts":   time.strftime("%H:%M:%S"),
        "kind": kind,
        "msg":  msg,
    })
    # Keep last 30
    st.session_state.events = st.session_state.events[-30:]


# ── API helpers ──────────────────────────────────────────────────────

@st.cache_data(ttl=10, show_spinner=False)
def check_health(api_url: str) -> dict:
    try:
        r = requests.get(f"{api_url}/health", timeout=None)
        return r.json() if r.status_code == 200 else {
            "status": "unhealthy", "components": {"http": str(r.status_code)}
        }
    except requests.exceptions.RequestException as e:
        return {"status": "unhealthy", "components": {"error": str(e)[:80]}}


def call_answer(api_url, api_key, query, retriever, top_k, fetch_k, min_score) -> dict:
    payload = {
        "query": query, "retriever": retriever, "top_k": top_k,
        "fetch_k": fetch_k, "min_score": min_score,
    }
    r = requests.post(
        f"{api_url}/answer",
        json=payload,
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        timeout=None,
    )
    if r.status_code != 200:
        raise RuntimeError(f"API {r.status_code}: {r.text[:200]}")
    return r.json()


def stream_answer(api_url, api_key, query, retriever, top_k, fetch_k, min_score) -> Iterator[dict]:
    payload = {
        "query": query, "retriever": retriever, "top_k": top_k,
        "fetch_k": fetch_k, "min_score": min_score,
    }
    with requests.post(
        f"{api_url}/answer/stream",
        json=payload,
        headers={"X-API-Key": api_key, "Content-Type": "application/json"},
        stream=True, timeout=None,
    ) as r:
        if r.status_code != 200:
            yield {"event": "error", "data": {"status": r.status_code, "text": r.text[:200]}}
            return
        current_event = None
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                current_event = line[6:].strip()
            elif line.startswith("data:"):
                try:
                    yield {"event": current_event, "data": json.loads(line[5:].strip())}
                except json.JSONDecodeError:
                    pass


def render_citations(citations: list, container) -> None:
    if not citations:
        return
    with container.container():
        with st.expander(f"📚 {len(citations)} citations", expanded=False):
            for c in citations:
                fname = (c.get("source_path") or "").split("/")[-1]
                st.markdown(
                    f"**[{c['n']}]** `{c.get('section_title', '—')}`  "
                    f"&nbsp;·&nbsp; score `{c.get('score', 0):.4f}`  "
                    f"&nbsp;·&nbsp; *{fname}*"
                )


# ── Sidebar — Summary, flow, settings ────────────────────────────────

with st.sidebar:
    st.markdown("### ⚖️ IPC Legal RAG")
    st.markdown(
        "Retrieval-augmented Q&A over the **Indian Penal Code** corpus "
        "(74 PDFs → 612 chunks). Hybrid retrieval combines dense vector "
        "search with BM25, then re-ranks with a cross-encoder before "
        "passing context to the LLM."
    )

    with st.expander("🛠 Tech stack", expanded=False):
        st.markdown(
            """
            | Layer | Tool |
            |---|---|
            | Parser | **Docling** (PDF → structured chunks) |
            | Embeddings | **embeddinggemma** (Ollama, 768-d) |
            | Vector store | **ChromaDB** |
            | Sparse | **BM25** (rank-bm25) |
            | Fusion | **RRF** (k=60) |
            | Reranker | **BGE-reranker-base** (sentence-transformers) |
            | LLM | **gemma-4-e4b** (Ollama) |
            | API | **FastAPI** + SSE streaming |
            | Auth | API-key + slowapi rate limit |
            """
        )

    with st.expander("🔗 Pipeline flow", expanded=False):
        st.markdown(
            """
            <div class="flow-container">
              <span class="flow-stage">Query</span>
              <span class="flow-arrow">→</span>
              <span class="flow-stage">Embed</span>
              <span class="flow-arrow">→</span>
              <span class="flow-stage">Dense + BM25</span>
              <span class="flow-arrow">→</span>
              <span class="flow-stage">RRF Fusion</span>
              <span class="flow-arrow">→</span>
              <span class="flow-stage">Rerank</span>
              <span class="flow-arrow">→</span>
              <span class="flow-stage">LLM</span>
              <span class="flow-arrow">→</span>
              <span class="flow-stage">Answer + Cite</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Query → Ollama embedding (768-d) → ChromaDB cosine search + "
            "BM25 keyword search → Reciprocal Rank Fusion → BGE cross-encoder "
            "scoring → top-k chunks to LLM → streamed answer with citations."
        )

    st.divider()
    st.markdown("### 🔌 Connection")
    st.session_state.api_url = st.text_input("API URL", value=st.session_state.api_url)
    st.session_state.api_key = st.text_input("API Key", value=st.session_state.api_key, type="password")

    health = check_health(st.session_state.api_url)
    status = health.get("status", "unknown")
    badge  = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴"}.get(status, "⚫")
    st.markdown(f"**Status:** {badge} `{status}`")
    with st.expander("Components", expanded=False):
        for k, v in health.get("components", {}).items():
            icon = "✅" if v == "ok" else "❌"
            st.markdown(f"{icon} **{k}**: `{v}`")

    st.divider()
    st.markdown("### 🔍 Retrieval (live tunable)")

    st.session_state.retriever = st.selectbox(
        "Retriever strategy",
        options=["hybrid_reranked", "dense", "bm25", "ensemble"],
        index=["hybrid_reranked","dense","bm25","ensemble"].index(st.session_state.retriever),
        help=(
            "**hybrid_reranked** (prod): BM25 + dense + cross-encoder rerank. "
            "Best recall + precision.\n\n"
            "**dense**: vector-only — fast, semantic match.\n\n"
            "**bm25**: keyword-only — best for legal section numbers.\n\n"
            "**ensemble**: RRF fusion of dense + BM25, no rerank — fast hybrid."
        ),
    )

    st.session_state.fetch_k = st.slider(
        "fetch_k (pre-rerank candidates)",
        min_value=5, max_value=50, value=st.session_state.fetch_k, step=5,
        help=(
            "How many candidates to retrieve **before** reranking. "
            "Higher → reranker sees more options, recall ↑, latency ↑. "
            "Default 20."
        ),
    )

    st.session_state.top_k = st.slider(
        "top_k (final context)",
        min_value=1, max_value=20, value=st.session_state.top_k,
        help=(
            "How many top-scored chunks reach the LLM. "
            "More → richer context, but bigger prompt → slower + may dilute focus. "
            "5 is a strong default."
        ),
    )

    st.session_state.min_score = st.slider(
        "min_score (reranker threshold)",
        min_value=0.0, max_value=1.0, value=float(st.session_state.min_score), step=0.01,
        help=(
            "Drop chunks scoring below this. "
            "Higher → stricter refusal of weak matches, recall ↓. "
            "Calibrated default: 0.01 (F1-optimal on the 96-example eval set)."
        ),
    )

    st.session_state.use_streaming = st.toggle(
        "Stream tokens (SSE)",
        value=st.session_state.use_streaming,
        help="Use /answer/stream. Citations land in ~1-2 s; tokens stream as LLM generates.",
    )

    st.divider()
    st.markdown("### 🛠 Server-side (read-only)")
    st.caption("Set at startup — change via `.env` + restart.")
    st.code(
        "LLM:        gemma-4-e4b (Ollama)\n"
        "Embeddings: embeddinggemma (768-d)\n"
        "Reranker:   BAAI/bge-reranker-base\n"
        "Vector DB:  ChromaDB (IPC_Corpus)\n"
        "Chunks:     612 (size=800, overlap=120)",
        language="yaml",
    )

    if st.button("🗑 Clear chat + monitor", use_container_width=True):
        st.session_state.messages = []
        st.session_state.events = []
        st.session_state.req_count = 0
        st.session_state.total_tokens = 0
        st.rerun()


# ── Main area: chat (left wide) + monitor (right narrow) ─────────────

chat_col, monitor_col = st.columns([4, 1.2], gap="medium")

# ─── Right column: Live monitor ──────────────────────────────────────

with monitor_col:
    st.markdown("### 📡 Live Monitor")

    badge = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴"}.get(
        check_health(st.session_state.api_url).get("status", "unknown"), "⚫"
    )
    st.markdown(f"**API:** {badge}")

    c1, c2 = st.columns(2)
    c1.metric("Requests", st.session_state.req_count)
    c2.metric("Tokens",   st.session_state.total_tokens)

    st.markdown("**Auth**")
    masked = (st.session_state.api_key[:4] + "…") if st.session_state.api_key else "—"
    st.caption(f"X-API-Key: `{masked}`")
    st.caption(f"Mode: {'SSE stream' if st.session_state.use_streaming else 'JSON POST'}")

    st.markdown("**Recent events**")
    events_box = st.container(height=380, border=True)
    with events_box:
        if not st.session_state.events:
            st.caption("_no events yet_")
        else:
            for e in reversed(st.session_state.events[-15:]):
                icon = {
                    "request": "▶",  "citations": "📚", "token": "·",
                    "done": "✅",    "error": "❌",    "info": "ℹ️",
                }.get(e["kind"], "•")
                st.markdown(
                    f'<div class="monitor-event">'
                    f'<code>{e["ts"]}</code> {icon} {e["msg"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ─── Left column: header, samples, chat ──────────────────────────────

with chat_col:
    st.markdown(
        "<h2 style='text-align:center;margin-bottom:0;'>⚖️ IPC Legal RAG</h2>"
        "<p style='text-align:center;color:#6c757d;margin-top:0;'>"
        "Hybrid retrieval over the Indian Penal Code"
        "</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.markdown("#### 💡 Try a question")
        samples = [
            "What is the punishment for theft under IPC?",
            "Define cruelty by husband under Section 498A.",
            "What is the difference between murder and culpable homicide?",
            "Explain criminal conspiracy under Section 120B.",
        ]
        cols = st.columns(2)
        for i, q in enumerate(samples):
            with cols[i % 2]:
                st.markdown('<div class="sample-btn">', unsafe_allow_html=True)
                if st.button(q, key=f"sample_{i}", use_container_width=True):
                    st.session_state._pending_query = q
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                render_citations(msg["citations"], st.empty())
            if msg.get("latency_ms"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Retriever", msg.get("retriever", "—"))
                c2.metric("Latency",   f"{msg['latency_ms']/1000:.1f} s")
                c3.metric("Citations", len(msg.get("citations", [])))

    pending = st.session_state.pop("_pending_query", None)
    user_query = pending or st.chat_input("Ask an IPC legal question…")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.req_count += 1
        log_event("request", f"POST /answer{'/stream' if st.session_state.use_streaming else ''}  q='{user_query[:35]}…'")

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            answer_box    = st.empty()
            citations_box = st.empty()
            metrics_box   = st.empty()

            full_answer    = ""
            citations      = []
            retriever_used = ""
            latency_ms     = 0
            token_count    = 0

            try:
                if st.session_state.use_streaming:
                    with st.spinner("Streaming…"):
                        for evt in stream_answer(
                            st.session_state.api_url,
                            st.session_state.api_key,
                            user_query,
                            st.session_state.retriever,
                            st.session_state.top_k,
                            st.session_state.fetch_k,
                            st.session_state.min_score,
                        ):
                            ev_t = evt.get("event")
                            data = evt.get("data", {})

                            if ev_t == "citations":
                                citations = data.get("citations", [])
                                render_citations(citations, citations_box)
                                log_event("citations", f"{len(citations)} chunks")
                            elif ev_t == "token":
                                tok = data.get("text", "")
                                full_answer += tok
                                token_count += 1
                                answer_box.markdown(full_answer + " ▌")
                            elif ev_t == "done":
                                retriever_used = data.get("retriever", "")
                                latency_ms = data.get("latency_ms", 0)
                                log_event("done", f"{token_count} tokens, {latency_ms/1000:.1f}s")
                            elif ev_t == "error":
                                log_event("error", str(data)[:60])
                                st.error(f"Server error: {data}")
                    answer_box.markdown(full_answer)
                else:
                    with st.spinner("Running RAG pipeline…"):
                        result = call_answer(
                            st.session_state.api_url,
                            st.session_state.api_key,
                            user_query,
                            st.session_state.retriever,
                            st.session_state.top_k,
                            st.session_state.fetch_k,
                            st.session_state.min_score,
                        )
                    full_answer    = result.get("answer", "")
                    citations      = result.get("citations", [])
                    retriever_used = result.get("retriever", "")
                    latency_ms     = result.get("latency_ms", 0)
                    token_count    = len(full_answer.split())  # approx
                    answer_box.markdown(full_answer)
                    render_citations(citations, citations_box)
                    log_event("done", f"~{token_count} words, {latency_ms/1000:.1f}s")

                st.session_state.total_tokens += token_count

                with metrics_box.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Retriever", retriever_used or st.session_state.retriever)
                    c2.metric("Latency",   f"{latency_ms/1000:.1f} s")
                    c3.metric("Citations", len(citations))

                st.session_state.messages.append({
                    "role":       "assistant",
                    "content":    full_answer,
                    "citations":  citations,
                    "retriever":  retriever_used or st.session_state.retriever,
                    "latency_ms": latency_ms,
                })

            except requests.exceptions.ConnectionError:
                log_event("error", "connection refused")
                st.error(f"❌ Cannot reach `{st.session_state.api_url}`. Is uvicorn running?")
            except Exception as e:
                log_event("error", str(e)[:60])
                st.error(f"Error: {e}")
        st.rerun()