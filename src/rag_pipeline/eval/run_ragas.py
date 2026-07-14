"""RAGAS run — configurable generator + 1-2 judges. Full output: summary + per-question + plots."""
import sys, json, random
from datetime import datetime, timezone
import pandas as pd
from rag_pipeline.config import cfg, log
from rag_pipeline.eval.schema import load_eval_set
from rag_pipeline.eval.ragas_runner import build_ragas_dataset, score_ragas_dataset
from rag_pipeline.providers import get_llm
from rag_pipeline.schemas import StatuteChunk
from rag_pipeline.retrievers.dense import DenseRetriever
from rag_pipeline.retrievers.bm25 import BM25Retriever
from rag_pipeline.retrievers.ensemble import EnsembleRetriever
from rag_pipeline.retrievers.reranker import Reranker, RerankedRetriever

TOP_K = 5


class MultiRetriever:
    def __init__(self):
        rr = Reranker()
        self.setups = {}
        for act, coll in [("IPC", "IPC_Corpus"), ("BNS", "BNS_Corpus")]:
            path = cfg.DATA_PROCESSED_DIR / f"{act.lower()}_chunks.json"
            chunks = [StatuteChunk(**c) for c in json.load(open(path))]
            ens = EnsembleRetriever([DenseRetriever(collection_name=coll), BM25Retriever(chunks)], fetch_k=20)
            self.setups[act] = RerankedRetriever(ens, rr, fetch_k=20, min_score=None)
    def retrieve(self, query, top_k=TOP_K):
        res = []
        for r in self.setups.values():
            res += r.retrieve(query, top_k=top_k)
        res.sort(key=lambda x: x[1], reverse=True)
        return res[:top_k]


def _subset(subset_n):
    exs = load_eval_set(cfg.EVAL_SET_PATH)
    pos = [e for e in exs if not e.is_negative()]
    random.seed(42)
    by_cat = {}
    for e in pos:
        by_cat.setdefault(e.category, []).append(e)
    per = max(1, subset_n // len(by_cat))
    out = []
    for cat, items in by_cat.items():
        out += random.sample(items, min(per, len(items)))
    return out, list(by_cat)


def main(subset_n=24):
    subset, cats = _subset(subset_n)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")
    evalset = cfg.EVAL_SET_FILE.replace(".json", "")
    base = f"{evalset}__ragas__{stamp}"

    # 1. Build dataset ONCE with the generator model
    gen = get_llm(provider="azure", deployment=cfg.RAGAS_GEN_DEPLOYMENT or None)
    log.info(f"Generator deployment: {cfg.RAGAS_GEN_DEPLOYMENT or 'default'} · {len(subset)} Q · cats={cats}")
    rows = build_ragas_dataset(
        examples=subset, retriever=MultiRetriever(), llm=gen, top_k=TOP_K,
        cache_path=cfg.RAGAS_PERQ_DIR / f"{base}__dataset.json",
    )

    # 2. Score with each configured judge
    judges = {"judge1": cfg.RAGAS_JUDGE1_DEPLOYMENT}
    if cfg.RAGAS_JUDGE2_DEPLOYMENT:
        judges["judge2"] = cfg.RAGAS_JUDGE2_DEPLOYMENT

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_set": cfg.EVAL_SET_FILE,
        "n_questions": len(rows),
        "generator": cfg.RAGAS_GEN_DEPLOYMENT or "default",
        "judges": {},
    }
    for jname, jdep in judges.items():
        log.info(f"Scoring with {jname} (deployment={jdep or 'default'})")
        from rag_pipeline.eval.ragas_runner import get_ragas_judges, get_default_metrics
        from rag_pipeline.providers import get_embeddings
        jl, je = get_ragas_judges(get_llm(provider="azure", deployment=jdep or None), get_embeddings())
        df = score_ragas_dataset(
            rows=rows,
            judge_llm=jl, judge_emb=je,
            metrics=get_default_metrics(jl, je),
            checkpoint_path=cfg.RAGAS_PERQ_DIR / f"{base}__{jname}__checkpoint.csv",
            output_path=cfg.RAGAS_PERQ_DIR / f"{base}__{jname}__perquestion.csv",
        )
        num = df.select_dtypes("number")
        summary["judges"][jname] = {
            "deployment": jdep or "default",
            "metrics_mean": {k: round(float(v), 4) for k, v in num.mean().items()},
        }

    # 3. Save summary
    sp = cfg.RAGAS_SUMMARY_DIR / f"{base}__summary.json"
    sp.write_text(json.dumps(summary, indent=2))
    log.info(f"Summary → {sp.name}")

    # 4. Plot
    try:
        from rag_pipeline.eval.plot_results import plot_ragas
        plot_ragas(sp)
    except Exception as e:
        log.warning(f"Plot skipped: {e}")

    print("\n=== RAGAS ===")
    print(json.dumps(summary["judges"], indent=2))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    main(n)