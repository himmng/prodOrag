"""IPC Legal RAG — Streamlit dashboard.

Run:
    streamlit run apps/dashboard/app.py
"""
import os
import json
import time
import uuid
from pathlib import Path
from typing import Iterator

import requests
import streamlit as st
from dotenv import load_dotenv

# Streamlit doesn't auto-load .env like the FastAPI backend (pydantic-settings)
# does — load it explicitly so API_URL / DASHBOARD_API_KEY are picked up.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

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
    "api_url": os.environ.get("API_URL", "http://localhost:8000"),
    "api_key":        os.environ.get("DASHBOARD_API_KEY", ""),
    "retriever":      "hybrid_reranked",
    "top_k":          5,
    "fetch_k":        20,
    "min_score":      0.01,
    "use_streaming":  True,
    "collections":    [],           # empty = search all corpus acts
    "meta":           {},           # corpus metadata from /meta
    "events":         [],
    "req_count":      0,
    "total_tokens":   0,
    "session_id":     uuid.uuid4().hex,   # per-browser-session isolation token
    "active_case":    None,               # {"doc_id", "filename"} of selected case
    "case_mode":      False,              # route chat to /answer/case when True
    "include_context": False,             # opt-in legislative commentary on /answer
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
    st.session_state.events = st.session_state.events[-30:]


# ── API helpers ──────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def fetch_meta(api_url: str) -> dict:
    """Corpus metadata — drives the UI so nothing is hardcoded to IPC/BNS."""
    try:
        r = requests.get(f"{api_url}/meta", timeout=5)
        if r.status_code == 200:
            return r.json()
    except requests.exceptions.RequestException:
        pass
    # Fallback so the dashboard still renders if the API is down
    return {
        "corpus": "unknown", "display_name": "Corpus",
        "acts": ["IPC", "BNS"], "pdf_acts": [],
        "context_enabled": False, "cross_reference": None,
        "models": {
            "llm_provider": "?", "llm_model": None,
            "embedding_provider": "?", "embedding_model": None,
            "reranker_model": "?", "vector_store": "chromadb",
        },
    }


@st.cache_data(ttl=10, show_spinner=False)
def check_health(api_url: str) -> dict:
    try:
        r = requests.get(f"{api_url}/health", timeout=None)
        return r.json() if r.status_code == 200 else {
            "status": "unhealthy", "components": {"http": str(r.status_code)}
        }
    except requests.exceptions.RequestException as e:
        return {"status": "unhealthy", "components": {"error": str(e)[:80]}}


def call_answer(api_url, api_key, query, retriever, top_k, fetch_k, min_score, collections,
                include_context=False):
    payload = {
        "query": query, "retriever": retriever, "top_k": top_k,
        "fetch_k": fetch_k, "min_score": min_score,
        "collections": collections, "include_context": include_context,
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


def stream_answer(api_url, api_key, query, retriever, top_k, fetch_k, min_score, collections,
                  include_context=False):
    payload = {
        "query": query, "retriever": retriever, "top_k": top_k,
        "fetch_k": fetch_k, "min_score": min_score,
        "collections": collections, "include_context": include_context,
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


# ── Case-file API helpers (session-isolated) ─────────────────────────

def _case_headers() -> dict:
    return {
        "X-API-Key":    st.session_state.api_key,
        "X-Session-Id": st.session_state.session_id,
    }


def upload_case(api_url, file) -> dict:
    r = requests.post(
        f"{api_url}/documents/upload",
        files={"file": (file.name, file.getvalue())},
        headers=_case_headers(),
        timeout=None,
    )
    if r.status_code != 200:
        raise RuntimeError(f"API {r.status_code}: {r.text[:200]}")
    return r.json()


def list_cases(api_url) -> list:
    r = requests.get(f"{api_url}/documents", headers=_case_headers(), timeout=None)
    return r.json().get("documents", []) if r.status_code == 200 else []


def delete_case(api_url, doc_id) -> None:
    requests.delete(f"{api_url}/documents/{doc_id}", headers=_case_headers(), timeout=None)


def answer_case(api_url, doc_id, question, top_k, collections) -> dict:
    r = requests.post(
        f"{api_url}/answer/case",
        json={"doc_id": doc_id, "question": question,
              "top_k": top_k, "collections": collections},
        headers={**_case_headers(), "Content-Type": "application/json"},
        timeout=None,
    )
    if r.status_code != 200:
        raise RuntimeError(f"API {r.status_code}: {r.text[:200]}")
    return r.json()


# ── Rendering helpers ────────────────────────────────────────────────

_DOC_TYPE_LABEL = {
    "committee_report": "Standing Committee Report",
    "sor":              "Statement of Objects & Reasons",
}


def render_context(context: list, container, key_prefix: str = "live") -> None:
    """Render interpretive commentary — clearly marked non-authoritative."""
    if not context:
        return
    with container.container():
        with st.expander(f"🏛️ {len(context)} legislative-context notes (commentary, not law)", expanded=False):
            st.caption("Interpretive background from parliamentary materials — NOT binding statute.")
            for i, c in enumerate(context):
                label = _DOC_TYPE_LABEL.get(c.get("doc_type"), c.get("source", "commentary"))
                page  = c.get("page_number")
                st.markdown(
                    f"**[{label}]**"
                    + (f" · p.{page}" if page else "")
                    + f"  &nbsp;·&nbsp; score `{c.get('score', 0):.3f}`"
                )
                txt = c.get("text", "")
                st.caption(txt[:400] + ("…" if len(txt) > 400 else ""))


def render_case_excerpts(excerpts: list, container, key_prefix: str = "live") -> None:
    if not excerpts:
        return
    with container.container():
        with st.expander(f"📎 {len(excerpts)} case excerpts (your uploaded file)", expanded=False):
            for i, ex in enumerate(excerpts):
                title = ex.get("section_title") or "—"
                page  = ex.get("page_number")
                st.markdown(
                    f"**[C{i+1}]** `{title}`  &nbsp;·&nbsp; score `{ex.get('score', 0):.4f}`"
                    + (f"  &nbsp;·&nbsp; p.{page}" if page else "")
                )
                st.caption(ex.get("text", "")[:400] + ("…" if len(ex.get("text", "")) > 400 else ""))


def _view_page(act: str, page) -> None:
    """Fetch and display a PDF page image inline."""
    if not (act and page):
        return
    try:
        resp = requests.get(
            f"{st.session_state.api_url}/documents/page-image",
            params={"act": act, "page": page},
            headers={"X-API-Key": st.session_state.api_key},
            timeout=15,
        )
        if resp.status_code == 200:
            st.image(resp.content, caption=f"{act} p.{page}", use_container_width=True)
        else:
            st.error(f"Preview failed: {resp.status_code}")
    except Exception as e:
        st.error(f"Preview error: {e}")


def render_citations(citations: list, container, key_prefix: str = "live") -> None:
    if not citations:
        return
    with container.container():
        with st.expander(f"📚 {len(citations)} citations", expanded=False):
            for c in citations:
                fname = (c.get("source_path") or "").split("/")[-1]
                act   = c.get("act") or "—"
                page  = c.get("page_number")
                sec   = c.get("section") or "?"

                if act in ("IPC", "BNS", "CONCORDANCE") and page:
                    label = "concordance" if act == "CONCORDANCE" else act
                    if st.button(f"📄 View {label} p.{page}", key=f"{key_prefix}_pg_{c['n']}_{act}_{page}"):
                        _view_page(act, page)

                    

                # Cross-reference badge — "other act" comes from corpus meta
                xref_parts = []
                if c.get("corresponds_to"):
                    xm = (st.session_state.get("meta") or {}).get("cross_reference") or {}
                    a, b = xm.get("source_act"), xm.get("target_act")
                    other_act = b if act == a else (a if act == b else "↔")
                    xref_parts.append(f"↔ {other_act} §{c['corresponds_to']}")
                if c.get("change_status") and c["change_status"] != "unchanged":
                    badge = {"new": "🟢 new", "changed": "🟡 changed", "deleted": "🔴 deleted"}.get(
                        c["change_status"], c["change_status"]
                    )
                    xref_parts.append(badge)
                xref = "  ·  ".join(xref_parts)

                st.markdown(
                    f"**[{c['n']}] {act} §{sec}** — `{c.get('section_title', '—')}`  "
                    f"&nbsp;·&nbsp; score `{c.get('score', 0):.4f}`"
                )
                if xref:
                    st.caption(xref)
                st.caption(f"📄 {fname} · p.{c.get('page_number', '?')}")


def render_cross_reference(cross_ref: dict, container, key_prefix: str = "live") -> None:
    if not cross_ref:
        return
    with container.container():
        row    = cross_ref.get("concordance_row")
        src_act = cross_ref.get("source_act") or "A"
        tgt_act = cross_ref.get("target_act") or "B"
        src    = cross_ref.get("source_section")
        tgt    = cross_ref.get("target_section")
        status = cross_ref.get("status")
        page   = cross_ref.get("page_number")
        xmeta  = (st.session_state.get("meta") or {}).get("cross_reference") or {}
        pdf_label = xmeta.get("pdf_label", "CONCORDANCE")
        badge  = {"new": "🟢 new", "changed": "🟡 changed", "deleted": "🔴 deleted",
                  "unchanged": "⚪ unchanged"}.get(status, status or "")
        with st.expander("🔗 Related — corresponding sections (concordance)", expanded=True):
            st.markdown(f"**Concordance Row {row}** — {src_act} §{src} ↔ {tgt_act} §{tgt}  ·  {badge}")
            if page and st.button(f"📄 View concordance table (row {row})", key=f"{key_prefix}_xref_view_{row}"):
                _view_page(pdf_label, page)
            for side, key in [(src_act, "source_citation"), (tgt_act, "target_citation")]:
                cit = cross_ref.get(key)
                if not cit:
                    continue
                st.markdown(f"**{side} §{cit.get('section')}** — `{cit.get('section_title', '—')}`")
                cpage = cit.get("page_number")
                if cpage and st.button(f"📄 View {side} p.{cpage}", key=f"{key_prefix}_xref_{side}_{cpage}"):
                    _view_page(side, cpage)


# ── Sidebar — Summary, flow, settings ────────────────────────────────

with st.sidebar:
    st.markdown("### ⚖️ Legal RAG")
    st.markdown(
        "Corpus-agnostic retrieval-augmented Q&A. Hybrid retrieval combines dense "
        "vector search with BM25, then re-ranks with a cross-encoder before "
        "passing context to the LLM. When the active corpus provides a concordance, "
        "cross-references between acts are surfaced from that table."
    )

    with st.expander("🛠 Tech stack", expanded=False):
        _models = st.session_state.meta.get("models", {})
        _embed_model = _models.get("embedding_model") or "?"
        _llm_model = _models.get("llm_model") or "?"
        _llm_provider = _models.get("llm_provider") or "?"
        _embed_provider = _models.get("embedding_provider") or "?"
        _reranker = _models.get("reranker_model") or "?"
        _vstore = _models.get("vector_store") or "chromadb"
        st.markdown(
            f"""
            | Layer | Tool |
            |---|---|
            | Parser | **Docling** + **pdfplumber** |
            | Embeddings | **{_embed_model}** ({_embed_provider}) |
            | Vector store | **{_vstore}** |
            | Sparse | **BM25** (rank-bm25) |
            | Fusion | **RRF** (k=60) |
            | Reranker | **{_reranker}** |
            | LLM | **{_llm_model}** ({_llm_provider}) |
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
    st.session_state.meta = fetch_meta(st.session_state.api_url)
    status = health.get("status", "unknown")
    badge  = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴"}.get(status, "⚫")
    st.markdown(f"**Status:** {badge} `{status}`  ·  Corpus: `{st.session_state.meta.get('corpus','?')}`")
    with st.expander("Components", expanded=False):
        for k, v in health.get("components", {}).items():
            icon = "✅" if v == "ok" else "❌"
            st.markdown(f"{icon} **{k}**: `{v}`")

    st.divider()
    st.markdown("### 📎 Case files (isolated)")
    st.caption(
        "Upload a real case file (judgment / FIR / charge sheet). It is embedded "
        "into a **private, session-isolated** collection — never mixed with the "
        "main corpus or other sessions."
    )
    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

    up = st.file_uploader(
        "Upload case (pdf / txt / md)", type=["pdf", "txt", "md"],
        key="case_uploader",
    )
    if up is not None and st.button("⬆️ Upload & index", use_container_width=True):
        try:
            with st.spinner("Parsing + embedding case…"):
                info = upload_case(st.session_state.api_url, up)
            st.session_state.active_case = {"doc_id": info["doc_id"], "filename": info["filename"]}
            st.session_state.case_mode = True
            log_event("case", f"uploaded {info['filename']} → {info['doc_id']}")
            st.success(f"Indexed `{info['filename']}` ({info.get('char_count', 0)} chars)")
            st.rerun()
        except Exception as e:
            st.error(f"Upload failed: {e}")

    cases = list_cases(st.session_state.api_url)
    if cases:
        labels = {f"{c['filename']} · {c['doc_id'][:6]}": c for c in cases}
        active_did = (st.session_state.active_case or {}).get("doc_id")
        idx = next((i for i, c in enumerate(cases) if c["doc_id"] == active_did), 0)
        pick = st.selectbox("Active case", options=list(labels.keys()), index=idx)
        chosen = labels[pick]
        st.session_state.active_case = {"doc_id": chosen["doc_id"], "filename": chosen["filename"]}
        if chosen.get("section_refs"):
            st.caption("Sections named in case: " + ", ".join(chosen["section_refs"][:12]))
        st.session_state.case_mode = st.toggle(
            "🔎 Ask about this case", value=st.session_state.case_mode,
            help="When on, your chat question is interpreted against the uploaded case.",
        )
        if st.button("🗑️ Delete active case", use_container_width=True):
            delete_case(st.session_state.api_url, chosen["doc_id"])
            st.session_state.active_case = None
            st.session_state.case_mode = False
            log_event("case", f"deleted {chosen['doc_id']}")
            st.rerun()
    else:
        st.session_state.case_mode = False

    st.divider()
    st.markdown("### 🔍 Retrieval (live tunable)")
    _acts = st.session_state.meta.get("acts", [])
    st.session_state.collections = st.multiselect(
        "Search in",
        options=_acts,
        default=[a for a in st.session_state.collections if a in _acts],
        help="Which corpus acts to search. Leave empty to search all acts.",
    )

    st.session_state.retriever = st.selectbox(
        "Retriever strategy",
        options=["hybrid_reranked", "dense", "bm25", "ensemble"],
        index=["hybrid_reranked", "dense", "bm25", "ensemble"].index(st.session_state.retriever),
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
            "Higher → reranker sees more options, recall ↑, latency ↑. Default 20."
        ),
    )

    st.session_state.top_k = st.slider(
        "top_k (final context)",
        min_value=1, max_value=20, value=st.session_state.top_k,
        help=(
            "How many top-scored chunks reach the LLM. "
            "More → richer context, but bigger prompt → slower. 5 is a strong default."
        ),
    )

    st.session_state.min_score = st.slider(
        "min_score (reranker threshold)",
        min_value=0.0, max_value=1.0, value=float(st.session_state.min_score), step=0.01,
        help=(
            "Drop chunks scoring below this. "
            "Higher → stricter refusal of weak matches, recall ↓. Calibrated default: 0.01."
        ),
    )

    st.session_state.use_streaming = st.toggle(
        "Stream tokens (SSE)",
        value=st.session_state.use_streaming,
        help="Use /answer/stream. Citations land in ~1-2 s; tokens stream as LLM generates.",
    )
    st.session_state.include_context = st.toggle(
        "🏛️ Include legislative commentary",
        value=st.session_state.include_context,
        help="Add non-binding interpretive background (e.g. committee reports / "
             "statements of objects) to general answers. Case Q&A always includes it.",
    )

    st.divider()
    st.markdown("### 🛠 Server-side (read-only)")
    st.caption("Set at startup — change via `.env` + restart.")
    _models = st.session_state.meta.get("models", {})
    st.code(
        f"LLM:        {_models.get('llm_model') or '?'} ({_models.get('llm_provider') or '?'})\n"
        f"Embeddings: {_models.get('embedding_model') or '?'} ({_models.get('embedding_provider') or '?'})\n"
        f"Reranker:   {_models.get('reranker_model') or '?'}\n"
        f"Vector DB:  {_models.get('vector_store') or 'chromadb'}\n"
        f"Corpus:     {st.session_state.meta.get('corpus', '?')} "
        f"(acts: {', '.join(st.session_state.meta.get('acts', []))})",
        language="yaml",
    )

    if st.button("🗑 Clear chat + monitor", use_container_width=True):
        st.session_state.messages = []
        st.session_state.events = []
        st.session_state.req_count = 0
        st.session_state.total_tokens = 0
        st.session_state.pop("_last_xref", None)
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
    _disp = st.session_state.meta.get("display_name", "Legal RAG")
    st.markdown(
        f"<h2 style='text-align:center;margin-bottom:0;'>⚖️ {_disp}</h2>"
        "<p style='text-align:center;color:#6c757d;margin-top:0;'>"
        "Hybrid retrieval-augmented Q&A"
        "</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.markdown("#### 💡 Try a question")
        samples = [
            "What is the punishment for theft?",
            "Explain criminal conspiracy.",
            "What is the difference between murder and culpable homicide?",
            "Which sections cover cheating and fraud?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(samples):
            with cols[i % 2]:
                st.markdown('<div class="sample-btn">', unsafe_allow_html=True)
                if st.button(q, key=f"sample_{i}", use_container_width=True):
                    st.session_state._pending_query = q
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("case_excerpts"):
                render_case_excerpts(msg["case_excerpts"], st.empty(), key_prefix=f"hist{i}")
            if msg.get("context"):
                render_context(msg["context"], st.empty(), key_prefix=f"hist{i}")
            if msg.get("cross_reference"):
                render_cross_reference(msg["cross_reference"], st.empty(), key_prefix=f"hist{i}")
            if msg.get("citations"):
                render_citations(msg["citations"], st.empty(), key_prefix=f"hist{i}")
            if msg.get("latency_ms"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Retriever", msg.get("retriever", "—"))
                c2.metric("Latency",   f"{msg['latency_ms']/1000:.1f} s")
                c3.metric("Citations", len(msg.get("citations", [])))

    case_active = st.session_state.case_mode and st.session_state.active_case
    if case_active:
        st.info(
            f"🔎 **Case mode** — answering against uploaded case "
            f"`{st.session_state.active_case['filename']}` (isolated). "
            f"Toggle off in the sidebar for general corpus Q&A."
        )

    pending = st.session_state.pop("_pending_query", None)
    placeholder = (
        "Ask about the uploaded case…" if case_active else "Ask a legal question…"
    )
    user_query = pending or st.chat_input(placeholder)

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.req_count += 1
        log_event("request", f"POST /answer{'/stream' if st.session_state.use_streaming else ''}  q='{user_query[:35]}…'")

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            answer_box    = st.empty()
            citations_box = st.empty()
            xref_box      = st.empty()
            context_box   = st.empty()
            metrics_box   = st.empty()

            full_answer    = ""
            citations      = []
            cross_ref      = None
            case_excerpts  = []
            context        = []
            retriever_used = ""
            latency_ms     = 0
            token_count    = 0

            try:
                if case_active:
                    excerpts_box = st.empty()
                    with st.spinner("Interpreting case against the corpus…"):
                        result = answer_case(
                            st.session_state.api_url,
                            st.session_state.active_case["doc_id"],
                            user_query,
                            st.session_state.top_k,
                            st.session_state.collections,
                        )
                    full_answer   = result.get("answer", "")
                    citations     = result.get("citations", [])
                    cross_ref     = result.get("cross_reference")
                    case_excerpts = result.get("case_excerpts", [])
                    context       = result.get("context", [])
                    latency_ms    = result.get("latency_ms", 0)
                    retriever_used = "case"
                    token_count   = len(full_answer.split())
                    answer_box.markdown(full_answer)
                    render_case_excerpts(case_excerpts, excerpts_box, key_prefix="live")
                    render_cross_reference(cross_ref, xref_box, key_prefix="live")
                    render_context(context, context_box, key_prefix="live")
                    render_citations(citations, citations_box, key_prefix="live")
                    log_event("case", f"answered in {latency_ms/1000:.1f}s")
                elif st.session_state.use_streaming:
                    with st.spinner("Streaming…"):
                        for evt in stream_answer(
                            st.session_state.api_url,
                            st.session_state.api_key,
                            user_query,
                            st.session_state.retriever,
                            st.session_state.top_k,
                            st.session_state.fetch_k,
                            st.session_state.min_score,
                            st.session_state.collections,
                            st.session_state.include_context,
                        ):
                            ev_t = evt.get("event")
                            data = evt.get("data", {})
                            if ev_t == "citations":
                                citations = data.get("citations", [])
                                render_citations(citations, citations_box, key_prefix="live")
                                log_event("citations", f"{len(citations)} chunks")
                            elif ev_t == "context":
                                context = data.get("context", [])
                                render_context(context, context_box, key_prefix="live")
                            elif ev_t == "cross_reference":
                                cross_ref = data.get("cross_reference")
                                render_cross_reference(cross_ref, xref_box, key_prefix="live")
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
                            st.session_state.collections,
                            st.session_state.include_context,
                        )
                    full_answer    = result.get("answer", "")
                    citations      = result.get("citations", [])
                    cross_ref      = result.get("cross_reference")
                    context        = result.get("context", [])
                    retriever_used = result.get("retriever", "")
                    latency_ms     = result.get("latency_ms", 0)
                    token_count    = len(full_answer.split())
                    answer_box.markdown(full_answer)
                    render_cross_reference(cross_ref, xref_box, key_prefix="live")
                    render_context(context, context_box, key_prefix="live")
                    render_citations(citations, citations_box, key_prefix="live")
                    log_event("done", f"~{token_count} words, {latency_ms/1000:.1f}s")

                st.session_state.total_tokens += token_count

                with metrics_box.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Retriever", retriever_used or st.session_state.retriever)
                    c2.metric("Latency",   f"{latency_ms/1000:.1f} s")
                    c3.metric("Citations", len(citations))

                st.session_state.messages.append({
                    "role":            "assistant",
                    "content":         full_answer,
                    "citations":       citations,
                    "case_excerpts":   case_excerpts,
                    "context":         context,
                    "cross_reference": cross_ref,
                    "retriever":       retriever_used or st.session_state.retriever,
                    "latency_ms":      latency_ms,
                })

            except requests.exceptions.ConnectionError:
                log_event("error", "connection refused")
                st.error(f"❌ Cannot reach `{st.session_state.api_url}`. Is uvicorn running?")
            except Exception as e:
                log_event("error", str(e)[:60])
                st.error(f"Error: {e}")
        st.rerun()