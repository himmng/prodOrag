"""RAGAS run — configurable generator + 1-2 judges. Full output: summary + per-question + plots."""
import sys, json, random
from datetime import datetime, timezone
import pandas as pd
from rag_pipeline.config import cfg, log
from rag_pipeline.corpus import load_corpus
from rag_pipeline.corpus.concordance import Concordance
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
    """Reranked retrieval spanning IPC + BNS.

    Also keeps the loaded chunks around so callers can build a (act, section)
    -> StatuteChunk index for concordance-resolved section lookups, the same
    way the API's lifespan handler builds `_state["section_index"]`.
    """

    def __init__(self):
        rr = Reranker()
        self.setups = {}
        self.chunks_by_act = {}
        for act, coll in [("IPC", "IPC_Corpus"), ("BNS", "BNS_Corpus")]:
            path = cfg.DATA_PROCESSED_DIR / f"{act.lower()}_chunks.json"
            chunks = [StatuteChunk(**c) for c in json.load(open(path))]
            self.chunks_by_act[act] = chunks
            ens = EnsembleRetriever([DenseRetriever(collection_name=coll), BM25Retriever(chunks)], fetch_k=20)
            self.setups[act] = RerankedRetriever(ens, rr, fetch_k=20, min_score=None)

    def retrieve(self, query, top_k=TOP_K):
        res = []
        for r in self.setups.values():
            res += r.retrieve(query, top_k=top_k)
        res.sort(key=lambda x: x[1], reverse=True)
        return res[:top_k]

    def section_index(self) -> dict:
        return {
            (act, c.section): c
            for act, chunks in self.chunks_by_act.items()
            for c in chunks
        }


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

    # Corpus + concordance — the same source of truth the API's /answer route
    # uses, so cross-reference questions get the same section-text injection
    # instead of being generated from semantic retrieval alone.
    corpus = load_corpus(cfg.RAG_CORPUS)
    concordance = (
        Concordance.from_json(corpus.concordance.output_json)
        if corpus.has_concordance and corpus.concordance.output_json.exists()
        else None
    )

    # 1. Build dataset ONCE with the generator model
    gen = get_llm(provider="azure", deployment=cfg.RAGAS_GEN_DEPLOYMENT or None)
    log.info(f"Generator deployment: {cfg.RAGAS_GEN_DEPLOYMENT or 'default'} · {len(subset)} Q · cats={cats}")
    retriever = MultiRetriever()
    rows = build_ragas_dataset(
        examples=subset, retriever=retriever, llm=gen, top_k=TOP_K,
        concordance=concordance, corpus=corpus, section_index=retriever.section_index(),
        cache_path=cfg.RAGAS_PERQ_DIR / f"{base}__dataset.json",
    )

    # 2. Score with every judge in RAGAS_JUDGE_DEPLOYMENTS (comma-separated)
    judge_deps = [d.strip().strip('"').strip("'") for d in cfg.RAGAS_JUDGE_DEPLOYMENTS.split(",") if d.strip().strip('"').strip("'")]
    if not judge_deps:
        judge_deps = [cfg.RAGAS_GEN_DEPLOYMENT or "default"]   # fall back to self-judge
    log.info(f"Judges ({len(judge_deps)}): {judge_deps}")

    from rag_pipeline.eval.ragas_runner import get_ragas_judges, get_default_metrics
    from rag_pipeline.providers import get_embeddings

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_set": cfg.EVAL_SET_FILE,
        "n_questions": len(rows),
        "generator": cfg.RAGAS_GEN_DEPLOYMENT or "default",
        "judges": {},
    }

    for jdep in judge_deps:
        jname = jdep  # deployment name is the judge key
        log.info(f"Scoring with judge={jname}")
        jl, je = get_ragas_judges(
            get_llm(provider="azure", deployment=jdep or None),
            get_embeddings(),
        )
        df = score_ragas_dataset(
            rows=rows,
            judge_llm=jl, judge_emb=je,
            metrics=get_default_metrics(jl, je),
            checkpoint_path=cfg.RAGAS_PERQ_DIR / f"{base}__{jname}__checkpoint.csv",
            output_path=cfg.RAGAS_PERQ_DIR / f"{base}__{jname}__perquestion.csv",
        )
        num = df.select_dtypes("number")
        metric_cols = [c for c in num.columns if c in ("faithfulness", "answer_relevancy", "semantic_similarity")]
        by_cat = {}
        if "category" in df.columns:
            for cat, g in df.groupby("category"):
                by_cat[cat] = {m: round(float(g[m].mean()), 4) for m in metric_cols}
        summary["judges"][jname] = {
            "deployment": jdep,
            "metrics_mean": {k: round(float(v), 4) for k, v in num[metric_cols].mean().items()},
            "by_category": by_cat,
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