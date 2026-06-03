"""Evaluation framework: schemas, metrics, and runners."""

from rag_pipeline.eval.qgen import generate_balanced_eval_set
from rag_pipeline.eval.ragas_runner import (
    build_ragas_dataset,
    get_default_metrics,
    get_default_run_config,
    get_ragas_judges,
    load_ragas_dataset,
    run_full_eval,
    score_ragas_dataset,
)
from rag_pipeline.eval.retrieval import (
    evaluate_retriever,
    hit_at_k,
    is_refusal,
    recall_at_k,
    reciprocal_rank,
    snippet_hit_at_k,
)
from rag_pipeline.eval.schema import (
    EvalExample,
    load_eval_set,
    load_negatives,
    load_refusal_markers,
    save_eval_set,
)

__all__ = [
    # Schema + curated data
    "EvalExample",
    "load_eval_set", "save_eval_set",
    "load_negatives", "load_refusal_markers",
    # Retrieval metrics
    "evaluate_retriever", "hit_at_k", "recall_at_k", "reciprocal_rank",
    "snippet_hit_at_k", "is_refusal",
    # Question generation
    "generate_balanced_eval_set",
    # RAGAS
    "get_ragas_judges", "get_default_metrics", "get_default_run_config",
    "build_ragas_dataset", "load_ragas_dataset",
    "score_ragas_dataset", "run_full_eval",
]