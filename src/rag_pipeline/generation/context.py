"""Context block builder + citation rendering.

Turns retrieval results into:
  - a numbered CONTEXT block ([1] ... [2] ...) for inclusion in the LLM prompt
  - a structured citations list for display / programmatic use
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


def build_context(
    results: list[tuple["Document", float]],
) -> tuple[str, list[dict]]:
    """Build a numbered context block + citations list."""
    blocks: list[str] = []
    citations: list[dict] = []
    for i, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        blocks.append(f"[{i}] {doc.page_content}")
        citations.append({
            "n":             i,
            "source_path":   meta.get("source_path", "?"),
            "page_number":   meta.get("page_number"),
            "section_title": meta.get("section_title"),
            "score":         score,
        })
    return "\n\n".join(blocks), citations


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