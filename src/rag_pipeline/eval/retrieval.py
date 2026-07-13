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

def _retrieved_sections(results) -> list[tuple]:
    """Extract (act, section) from retrieved docs."""
    out = []
    for doc, _ in results:
        m = doc.metadata or {}
        if m.get("act") and m.get("section"):
            out.append((m["act"], str(m["section"])))
    return out


def section_hit_at_k(example, retrieved_sections) -> int:
    if not example.gold_sections:
        return 0
    gold = {(g["act"], str(g["section"])) for g in example.gold_sections}
    return int(bool(gold & set(retrieved_sections)))


def section_recall_at_k(example, retrieved_sections) -> float:
    if not example.gold_sections:
        return 0.0
    gold = {(g["act"], str(g["section"])) for g in example.gold_sections}
    hits = len(gold & set(retrieved_sections))
    return hits / len(gold)


def section_reciprocal_rank(example, retrieved_sections) -> float:
    if not example.gold_sections:
        return 0.0
    gold = {(g["act"], str(g["section"])) for g in example.gold_sections}
    for i, sec in enumerate(retrieved_sections, 1):
        if sec in gold:
            return 1.0 / i
    return 0.0

# ── Top-level evaluator ─────────────────────────────────────────────────

def evaluate_retriever(
    retriever: "BaseRetriever",
    examples: list[EvalExample],
    top_k: int = 5,
    refusal_floor: float = 0.55,
) -> dict:
    """Run all retrieval metrics over a (retriever, eval set) pair.
    Buckets by difficulty, by category, and by the category×difficulty grid.
    Negatives (out-of-scope) are scored separately as a refusal proxy:
    a negative is 'correct' if nothing retrieved scores >= refusal_floor.
    """
    positives = [e for e in examples if not e.is_negative()]
    negatives = [e for e in examples if e.is_negative()]

    metrics = ("hit", "recall", "mrr", "snippet")
    sums = defaultdict(float)
    per_diff = defaultdict(lambda: defaultdict(float));  diff_n = defaultdict(int)
    per_cat  = defaultdict(lambda: defaultdict(float));  cat_n  = defaultdict(int)
    per_grid = defaultdict(lambda: defaultdict(float));  grid_n = defaultdict(int)
    per_question = []
    for ex in positives:
        results = retriever.retrieve(ex.question, top_k=top_k)
        paths = [doc.metadata.get("source_path", "") for doc, _ in results]
        texts = [doc.page_content for doc, _ in results]
        secs  = _retrieved_sections(results)
        if ex.gold_sections:
            scores = {
                "hit":     section_hit_at_k(ex, secs),
                "recall":  section_recall_at_k(ex, secs),
                "mrr":     section_reciprocal_rank(ex, secs),
                "snippet": snippet_hit_at_k(ex, texts),
            }
        else:
            scores = {
                "hit":     hit_at_k(ex, paths),
                "recall":  recall_at_k(ex, paths),
                "mrr":     reciprocal_rank(ex, paths),
                "snippet": snippet_hit_at_k(ex, texts),
            }

        
        gold = [(g["act"], str(g["section"])) for g in ex.gold_sections] if ex.gold_sections else []
        
        per_question.append({
            "question": ex.question,
            "category": ex.category or "uncategorized",
            "difficulty": ex.difficulty or "unknown",
            "gold_sections": gold,
            "retrieved_sections": secs[:top_k],
            "hit": scores["hit"],
            "recall": scores["recall"],
            "mrr": scores["mrr"],
        })

        cat  = ex.category or "uncategorized"
        diff = ex.difficulty or "unknown"
        grid = f"{cat}|{diff}"
        for k, v in scores.items():
            sums[k] += v
            per_diff[diff][k] += v
            per_cat[cat][k]   += v
            per_grid[grid][k] += v
        diff_n[diff] += 1; cat_n[cat] += 1; grid_n[grid] += 1

    n = max(len(positives), 1)
    def avg_block(store, counts):
        return {key: {k: store[key][k] / counts[key] for k in metrics}
                for key in counts}

    # Negatives: refusal proxy — correct if top retrieved score < floor
    neg_by_diff = defaultdict(lambda: {"correct": 0, "total": 0})
    neg_correct = 0
    neg_per_question = []
    for ex in negatives:
        results = retriever.retrieve(ex.question, top_k=top_k)
        top_score = max((s for _, s in results), default=0.0)
        ok = 1 if top_score < refusal_floor else 0
        neg_correct += ok
        d = ex.difficulty or "unknown"
        neg_by_diff[d]["correct"] += ok
        neg_by_diff[d]["total"]   += 1
        neg_per_question.append({
            "question": ex.question,
            "category": ex.category or "negative",
            "difficulty": ex.difficulty or "unknown",
            "top_score": top_score,
            "correct_empty": bool(ok),
        })

    negatives_block = {
        "refusal_floor": refusal_floor,
        "correct_empty_rate": (neg_correct / len(negatives)) if negatives else None,
        "n_negative": len(negatives),
        "by_difficulty": {
            d: {"correct_empty_rate": v["correct"] / v["total"], "n": v["total"]}
            for d, v in neg_by_diff.items()
        },
        "per_question": neg_per_question,
    }

    return {
        "overall":       {k: sums[k] / n for k in metrics},
        "by_difficulty": avg_block(per_diff, diff_n),
        "by_category":   avg_block(per_cat, cat_n),
        "by_category_difficulty": avg_block(per_grid, grid_n),
        "counts": {
            "by_difficulty": dict(diff_n),
            "by_category":   dict(cat_n),
            "by_category_difficulty": dict(grid_n),
        },
        "negatives":     negatives_block,
        "n_positive":    len(positives),
        "per_question": per_question,
    }

def threshold_sweep(
    retriever_no_filter,
    examples: list[EvalExample],
    thresholds: list[float],
    top_k: int = 5,
) -> list[dict]:
    """Pre-retrieve once with no filter, then score each threshold.

    For each threshold:
      - recall_positive: mean hit@k across positives after filtering
      - refusal_negative: fraction of negatives where all docs were filtered out
      - f1: harmonic mean of the two
    """
    # Cache retrievals once (expensive, otherwise we'd repeat per threshold)
    cached_pos: list[tuple[EvalExample, list]] = []
    cached_neg: list[tuple[EvalExample, list]] = []
    for ex in examples:
        results = retriever_no_filter.retrieve(ex.question, top_k=top_k)
        (cached_neg if ex.is_negative() else cached_pos).append((ex, results))

    rows = []
    for thresh in thresholds:
        # Positives: use hit_at_k on the filtered set
        hits = 0
        for ex, results in cached_pos:
            kept_paths = [
                doc.metadata.get("source_path", "")
                for doc, score in results
                if score >= thresh
            ]
            hits += hit_at_k(ex, kept_paths)

        # Negatives: refusal = no doc survived the filter
        refused = 0
        for ex, results in cached_neg:
            if not any(score >= thresh for _, score in results):
                refused += 1

        n_pos = max(len(cached_pos), 1)
        n_neg = max(len(cached_neg), 1)
        recall = hits / n_pos
        refusal = refused / n_neg
        f1 = (
            2 * recall * refusal / (recall + refusal)
            if (recall + refusal) > 0 else 0.0
        )

        rows.append({
            "threshold":        round(thresh, 4),
            "recall_positive":  round(recall, 4),
            "refusal_negative": round(refusal, 4),
            "f1":               round(f1, 4),
            "n_positives":      len(cached_pos),
            "n_negatives":      len(cached_neg),
        })

    return rows
# ── Refusal detection (for negative-example testing) ────────────────────

# All substrings are pre-normalized. Each covers the exact phrase OR a
# common LLM paraphrase. Keep in sync with REFUSAL_TEXT in generation/pipeline.py.

def is_refusal(response_text: str) -> bool:
    """Did the LLM produce a well-formed refusal? Markers live in
    eval/data/refusal_markers.yaml — edit the YAML, not this function."""
    norm = _normalize(response_text)
    return any(marker in norm for marker in load_refusal_markers())