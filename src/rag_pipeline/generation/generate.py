"""State-free answer generation — the single code path shared by the API's
/answer route and the RAGAS eval harness.

Everything here takes its dependencies (retriever, llm, concordance, corpus,
section_index) as plain arguments. No FastAPI, no process-wide `_state` dict —
callers own their own state and just pass it in. This is what lets the API
route and eval/ragas_runner.py call the exact same logic instead of each
reimplementing retrieve -> concordance-inject -> build_context -> generate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

from rag_pipeline.generation.context import build_context

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseChatModel
    from rag_pipeline.corpus.registry import CorpusConfig
    from rag_pipeline.corpus.concordance import Concordance, ConcordanceRow
    from rag_pipeline.retrievers.base import BaseRetriever


REFUSAL_TEXT = "I don't have that information in the provided documents."

# Interpretive commentary similarity tops out ~0.3 with embeddinggemma → low floor.
CONTEXT_FLOOR = 0.15


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


def concordance_context(
    query: str,
    concordance: Optional["Concordance"],
    corpus: Optional["CorpusConfig"],
) -> tuple[str, list[dict], list]:
    """Detect section mentions, look up cross-references (config-driven labels).

    Returns (prompt_text, resolved, rows):
      - prompt_text: CROSS-REFERENCE lines for the LLM
      - resolved:    [{act, section}] pointing at real section texts to cite
      - rows:        matched ConcordanceRow objects (for table-location citation)
    """
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


def fetch_section_chunk(section_index: dict, act: str, section: str):
    """Return (Document, score) for an exact act+section, or None. Vectorstore-agnostic."""
    c = section_index.get((act, section))
    if c is None:
        return None
    from langchain_core.documents import Document
    return (Document(page_content=c.text, metadata=c.to_langchain_metadata()), 1.0)


def context_block(hits: list) -> str:
    """Format interpretive-commentary hits as a labeled, non-authoritative prompt block."""
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


def _corpus_display(corpus: Optional["CorpusConfig"]) -> str:
    return corpus.display_name if corpus else "the provided corpus"


def build_answer_prompt(query: str, context: str, ctx_block: str, corpus: Optional["CorpusConfig"]) -> str:
    """Grounded-QA prompt, corpus name pulled from config (no IPC/BNS hardcoding)."""
    return (
        f"You are a legal assistant answering questions about {_corpus_display(corpus)}. "
        f"The CONTEXT below contains the FULL TEXT of the relevant statute sections "
        f"(each block starts with a [n] marker, then the section's complete text). "
        f"Treat this text as the authoritative source and answer the question directly "
        f"from it — summarize, compare, and explain the sections as needed. "
        f"Use any CROSS-REFERENCE lines for section correspondences. "
        f"Treat LEGISLATIVE CONTEXT as non-binding background. "
        f"Cite sources by their [n] marker. Only say information is unavailable if the "
        f"CONTEXT genuinely does not contain it.\n\n"
        f"CONTEXT:\n{context}"
        + (f"\n\n{ctx_block}" if ctx_block else "")
        + f"\n\nQUESTION: {query}\n\nANSWER:"
    )


@dataclass
class AnswerResult:
    """Everything a caller needs to render a response or a RAGAS row."""
    answer: str
    context: str                        # full assembled context string sent to the LLM
    docs_with_scores: list = field(default_factory=list)   # semantic hits, post-floor
    resolved_docs: list = field(default_factory=list)      # concordance-resolved section (doc, score) pairs
    xref_text: str = ""
    resolved: list = field(default_factory=list)            # [{act, section}]
    rows: list = field(default_factory=list)                 # matched ConcordanceRow objects
    ctx_hits: list = field(default_factory=list)             # interpretive commentary (doc, score) pairs


def generate_answer(
    query: str,
    retriever: "BaseRetriever",
    llm: "BaseChatModel",
    *,
    concordance: Optional["Concordance"] = None,
    corpus: Optional["CorpusConfig"] = None,
    context_retriever: Optional["BaseRetriever"] = None,
    section_index: Optional[dict] = None,
    top_k: int = 5,
    min_score: Optional[float] = 0.55,
    include_context: bool = False,
    context_floor: float = CONTEXT_FLOOR,
    context_top_k: int = 3,
) -> AnswerResult:
    """Retrieve -> concordance-inject -> build_context -> generate.

    Mirrors the API's /answer route exactly: refuses only when raw retrieval
    is empty; a non-empty raw retrieval that scores entirely below `min_score`
    still proceeds (context may come solely from concordance-resolved sections).
    """
    section_index = section_index or {}

    raw_docs = retriever.retrieve(query, top_k=top_k)
    if not raw_docs:
        return AnswerResult(answer=REFUSAL_TEXT, context="")

    docs_with_scores = (
        [(d, s) for d, s in raw_docs if s >= min_score] if min_score is not None else raw_docs
    )
    context = build_context(docs_with_scores)

    xref_text, resolved, rows = concordance_context(query, concordance, corpus)

    resolved_docs = []
    for r in resolved:
        hit = fetch_section_chunk(section_index, r["act"], r["section"])
        if hit:
            resolved_docs.append(hit)
    if resolved_docs:
        context = f"{build_context(resolved_docs)}\n\n{context}"
    if xref_text:
        context = f"{xref_text}\n\n{context}"

    ctx_hits = []
    if include_context and context_retriever is not None:
        hits = context_retriever.retrieve(query, top_k=context_top_k)
        ctx_hits = [(d, s) for d, s in hits if s >= context_floor]
    ctx_block_text = context_block(ctx_hits)

    prompt = build_answer_prompt(query, context, ctx_block_text, corpus)
    llm_resp = llm.invoke(prompt)
    answer_text = getattr(llm_resp, "content", None) or str(llm_resp)

    return AnswerResult(
        answer=answer_text,
        context=context,
        docs_with_scores=docs_with_scores,
        resolved_docs=resolved_docs,
        xref_text=xref_text,
        resolved=resolved,
        rows=rows,
        ctx_hits=ctx_hits,
    )
