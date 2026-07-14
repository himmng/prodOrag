"""Batched, checkpointed RAGAS scoring + dataset builder.

Two-step flow:
  1. build_ragas_dataset()  — for each EvalExample: retrieve → generate →
                              emit a RAGAS-format row. Optional JSON cache.
  2. score_ragas_dataset()  — score in batches with per-batch checkpointing.
                              Re-running resumes from checkpoint, no work lost.

Default metric set drops LLMContextPrecisionWithoutReference — it was ~100%
NaN on local Gemma (its multi-step claim-decomposition prompt is too brittle
for small judges). Re-add when judging with a cloud LLM.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm

from rag_pipeline.config import cfg, log
from rag_pipeline.eval.schema import EvalExample
from rag_pipeline.generation.context import build_context
from rag_pipeline.prompts import get_prompt

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from rag_pipeline.retrievers.base import BaseRetriever


warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")


# ── Judge + metric factories ────────────────────────────────────────────

def get_ragas_judges(llm: "BaseChatModel", embeddings: "Embeddings"):
    """Wrap a LangChain LLM + embeddings as RAGAS-compatible judges."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def get_default_metrics(judge_llm, judge_emb) -> list:
    """The 3-metric set that works reliably with local Gemma judging."""
    from ragas.metrics import Faithfulness, ResponseRelevancy, SemanticSimilarity
    return [
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm, embeddings=judge_emb),
        SemanticSimilarity(embeddings=judge_emb),
    ]


def get_default_run_config():
    from ragas import RunConfig
    return RunConfig(timeout=180, max_retries=3, max_wait=60, max_workers=1)


# ── Dataset builder ─────────────────────────────────────────────────────

def build_ragas_dataset(
    examples: list[EvalExample],
    retriever: "BaseRetriever",
    llm: "BaseChatModel",
    top_k: int | None = None,
    system_prompt_name: str = "system",
    cache_path: Path | None = None,
) -> list[dict]:
    """For each example: retrieve → generate → emit a RAGAS-format row."""
    top_k = top_k or cfg.TOP_K
    system_prompt = get_prompt(system_prompt_name)
    rows: list[dict] = []

    for ex in tqdm(examples, desc="build_ragas_dataset"):
        results = retriever.retrieve(ex.question, top_k=top_k)
        contexts = [d.page_content for d, _ in results]

        if results:
            context_block = build_context(results)
            user_msg = f"CONTEXT:\n{context_block}\n\nQUESTION: {ex.question}\n\nANSWER:"
            resp = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_msg),
            ])
            response_text = resp.content
        else:
            response_text = "I don't have that information in the provided documents."

        rows.append({
            "user_input":         ex.question,
            "retrieved_contexts": contexts,
            "response":           response_text,
            "reference": (
                ex.reference_answer
                or (ex.gold_snippets[0] if ex.gold_snippets else "")
            ),
        })

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info(f"Cached RAGAS dataset ({len(rows)} rows) → {Path(cache_path).name}")

    return rows


def load_ragas_dataset(cache_path: Path) -> list[dict]:
    return json.loads(Path(cache_path).read_text(encoding="utf-8"))


# ── Batched + checkpointed scorer ───────────────────────────────────────

def score_ragas_dataset(
    rows: list[dict],
    metrics: list | None = None,
    judge_llm=None,
    judge_emb=None,
    run_config=None,
    batch_size: int = 5,
    checkpoint_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Score a RAGAS dataset in batches; checkpoint after each batch."""
    from ragas import EvaluationDataset, evaluate
    from rag_pipeline.providers import get_embeddings, get_llm

    # Defaults — wire up judges from providers if not given
    if judge_llm is None or judge_emb is None:
        jl, je = get_ragas_judges(get_llm(), get_embeddings())
        judge_llm = judge_llm or jl
        judge_emb = judge_emb or je
    if metrics is None:
        metrics = get_default_metrics(judge_llm, judge_emb)
    if run_config is None:
        run_config = get_default_run_config()

    if checkpoint_path is None:
        checkpoint_path = cfg.PROJECT_ROOT / "eval" / "results" / "ragas_checkpoint.csv"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    all_dfs: list[pd.DataFrame] = []
    start_idx = 0
    if checkpoint_path.exists():
        existing = pd.read_csv(checkpoint_path)
        all_dfs.append(existing)
        start_idx = len(existing)
        log.info(f"Resuming from {start_idx}/{len(rows)} rows")

    for batch_start in range(start_idx, len(rows), batch_size):
        batch_end = min(batch_start + batch_size, len(rows))
        batch = rows[batch_start:batch_end]
        log.info(f"Scoring rows {batch_start + 1}–{batch_end} of {len(rows)}")

        dataset = EvaluationDataset.from_list(batch)
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_emb,
            run_config=run_config,
            show_progress=False,
        )
        all_dfs.append(result.to_pandas())

        pd.concat(all_dfs, ignore_index=True).to_csv(checkpoint_path, index=False)
        log.info(f"  ✓ checkpoint saved ({sum(len(d) for d in all_dfs)}/{len(rows)})")

    df = pd.concat(all_dfs, ignore_index=True)

    if output_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = cfg.PROJECT_ROOT / "eval" / "results" / f"ragas_{ts}.csv"
    df.to_csv(output_path, index=False)
    log.info(f"Final CSV → {output_path}")

    return df


# ── Convenience: end-to-end ─────────────────────────────────────────────

def run_full_eval(
    examples: list[EvalExample],
    retriever: "BaseRetriever",
    llm: "BaseChatModel",
    top_k: int | None = None,
    dataset_cache: Path | None = None,
    checkpoint_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """End-to-end: build dataset -> score -> return scored DataFrame."""
    if dataset_cache and Path(dataset_cache).exists():
        log.info(f"Loading cached dataset from {dataset_cache}")
        rows = load_ragas_dataset(dataset_cache)
    else:
        rows = build_ragas_dataset(
            examples=examples,
            retriever=retriever,
            llm=llm,
            top_k=top_k,
            cache_path=dataset_cache,
        )

    return score_ragas_dataset(
        rows=rows,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
    )