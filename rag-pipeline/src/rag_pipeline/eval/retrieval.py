"""Retrieval evaluation metrics + refusal detection.

Metrics over a (retriever, eval set) pair:
  - Hit@k:     1 if ANY gold source path appears in top-k, else 0
  - Recall@k:  fraction of gold sources found in top-k
  - MRR:       1/rank of first gold hit; 0 if no hit
  - Snippet@k: 1 if ANY gold snippet appears in any top-k chunk text
               (substring match + 0.6 token-overlap fallback)

Negatives (empty gold_source_paths) are skipped here — they're for
generation-side refusal testing, evaluated with is_refusal().
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING

from rag_pipeline.eval.schema import EvalExample
from rag_pipeline.generation.pipeline import REFUSAL_TEXT
from rag_pipeline.eval.schema import load_refusal_markers

if TYPE_CHECKING:
    from rag_pipeline.retrievers.base import BaseRetriever


_WORD_RE = re.compile(r"\w+")
_NORM_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _NORM_RE.sub(" ", text.lower()).strip()


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text)}


def _snippet_matches(snippet: str, retrieved_text: str, token_overlap: float = 0.6) -> bool:
    """Substring match first; fall back to fractional token overlap."""
    s_norm, r_norm = _normalize(snippet), _normalize(retrieved_text)
    if s_norm in r_norm:
        return True
    s_tok = _tokens(snippet)
    if not s_tok:
        return False
    r_tok = _tokens(retrieved_text)
    return len(s_tok & r_tok) / len(s_tok) >= token_overlap


# ── Per-example metrics ─────────────────────────────────────────────────

def hit_at_k(example: EvalExample, retrieved_paths: list[str]) -> int:
    return int(any(p in retrieved_paths for p in example.gold_source_paths))


def recall_at_k(example: EvalExample, retrieved_paths: list[str]) -> float:
    if not example.gold_source_paths:
        return 0.0
    hits = sum(1 for p in example.gold_source_paths if p in retrieved_paths)
    return hits / len(example.gold_source_paths)


def reciprocal_rank(example: EvalExample, retrieved_paths: list[str]) -> float:
    for rank, path in enumerate(retrieved_paths, start=1):
        if path in example.gold_source_paths:
            return 1.0 / rank
    return 0.0


def snippet_hit_at_k(example: EvalExample, retrieved_texts: list[str]) -> int:
    if not example.gold_snippets:
        return 0
    return int(any(
        _snippet_matches(snip, text)
        for snip in example.gold_snippets
        for text in retrieved_texts
    ))


# ── Top-level evaluator ─────────────────────────────────────────────────

def evaluate_retriever(
    retriever: "BaseRetriever",
    examples: list[EvalExample],
    top_k: int = 5,
) -> dict:
    """Run all retrieval metrics over a (retriever, eval set) pair.

    Returns:
        {
          "overall":       {hit, recall, mrr, snippet},
          "by_difficulty": {easy/medium/hard: {hit, recall, mrr, snippet}},
          "n_positive":    int,
        }
    """
    positives = [e for e in examples if not e.is_negative()]

    sums: dict[str, float] = defaultdict(float)
    per_diff: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_diff_counts: dict[str, int] = defaultdict(int)

    for ex in positives:
        results = retriever.retrieve(ex.question, top_k=top_k)
        paths = [doc.metadata.get("source_path", "") for doc, _ in results]
        texts = [doc.page_content for doc, _ in results]

        scores = {
            "hit":     hit_at_k(ex, paths),
            "recall":  recall_at_k(ex, paths),
            "mrr":     reciprocal_rank(ex, paths),
            "snippet": snippet_hit_at_k(ex, texts),
        }
        for k, v in scores.items():
            sums[k] += v
            per_diff[ex.difficulty][k] += v
        per_diff_counts[ex.difficulty] += 1

    n = max(len(positives), 1)
    overall = {k: sums[k] / n for k in ("hit", "recall", "mrr", "snippet")}

    by_difficulty = {
        diff: {k: per_diff[diff][k] / per_diff_counts[diff]
               for k in ("hit", "recall", "mrr", "snippet")}
        for diff in per_diff_counts
    }

    return {
        "overall":       overall,
        "by_difficulty": by_difficulty,
        "n_positive":    len(positives),
    }


# ── Refusal detection (for negative-example testing) ────────────────────

# All substrings are pre-normalized. Each covers the exact phrase OR a
# common LLM paraphrase. Keep in sync with REFUSAL_TEXT in generation/pipeline.py.

def is_refusal(response_text: str) -> bool:
    """Did the LLM produce a well-formed refusal? Markers live in
    eval/data/refusal_markers.yaml — edit the YAML, not this function."""
    norm = _normalize(response_text)
    return any(marker in norm for marker in load_refusal_markers())