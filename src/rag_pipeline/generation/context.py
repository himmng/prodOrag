"""Context block builder + citation rendering.

Turns retrieval results into:
  - a numbered CONTEXT block ([1] ... [2] ...) for inclusion in the LLM prompt
  - a structured citations list for display / programmatic use
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document

def build_context(docs_with_scores: list[tuple]) -> str:
    """Format retrieved (doc, score) pairs as numbered context for the LLM.

    Each block: [n] ACT §SECTION — title\n<full text>
    """
    parts = []
    for i, (doc, _score) in enumerate(docs_with_scores, start=1):
        meta = doc.metadata or {}
        header = f"[{i}]"
        if meta.get("act") and meta.get("section"):
            header += f" {meta['act']} §{meta['section']}"
            if meta.get("section_title"):
                header += f" — {meta['section_title']}"
        elif meta.get("source_path"):
            # Fallback for old RagChunk-style metadata
            header += f" {meta['source_path'].split('/')[-1]}"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)

def pretty_print(response: dict, score_label: str = "score") -> None:
    """Human-readable rendering of an answer dict (question, answer, citations)."""
    print(response["question"])
    print()
    print(response["answer"])
    print()
    print("Sources:")
    for c in response["citations"]:
        path       = c.get("source_path", "?")
        page       = c.get("page_number")
        title      = c.get("section_title")
        score      = c.get("score")
        page_part  = f"  |  p.{page}" if page is not None else ""
        title_part = f" | § {title}" if title else ""
        score_part = f"   ({score_label}={score:.3f})" if score is not None else ""
        print(f"   [{c['n']}] {path}{page_part}{title_part}{score_part}")