# PROGRESS.md — protoRAG development log

> Append-only running log of what changed and why, newest first. Complements the
> two standing docs: **CLAUDE.md** (how to work here) and **docs/HANDOFF.md**
> (current phase status + next steps). Update this file whenever a unit of work
> lands — a feature, a fix, a data rebuild, a decision. Keep entries short:
> what changed, why, and any follow-up. Use absolute dates.

---

## Phase status (mirror of HANDOFF §1)

| Phase | Scope | Status |
|---|---|---|
| 0 | Docling parse → Chroma → naive retrieval | ✅ done |
| 1 | Hybrid retrieval: BM25 + dense + RRF + BGE reranker | ✅ done |
| 2 | Query expansion (multi-query) | 🟡 built, PARKED (refusal 5%→25%) |
| 3 | Agentic router (LangGraph) | ⏳ planned |
| 4 | Eval framework (hit@k, recall, mrr, refusal) | ✅ done; **eval set v2/v3 in progress** |
| 5 | Prod hardening: FastAPI, SSE, auth, rate limit, CI, Docker | ✅ done |
| 5.5 | Streamlit dashboard | ✅ done |
| Corpus-agnostic refactor | Config-driven serving, `/meta`, gated concordance | ✅ done |
| 6 | Cloud deploy (Azure models, managed vector store) | ⏳ future |
| 7.5 | Contextual layer (committee report + SOR) | ✅ done |

---

## Log

### 2026-07-09 — Eval-set v3 generator (section-based, production-grade)

Branch `feature/eval-set-v2`. Built the second-generation eval-set generator on
top of v2, addressing quality gaps found while reviewing the v2 output.

**v2 recap** (`eval/qgen_v2.py`, `eval/eval_set_v2.json`, 200 Qs): retired the old
file-path-matching `qgen.py`; ground-truth is now **section-based** and stamped by
construction (never invented by the model). Four categories (ipc/bns substantive,
cross_reference, negative) × three difficulties; distribution 200 @ 60/60/80.
Prompts under `prompts/templates/qgen_v2_*`. Atomic incremental save + auto-resume
so an interrupted run wastes no API calls. Azure content-filter errors are caught
and skipped, not fatal.

**v3 upgrades** (`eval/qgen_v3.py`, `eval/eval_set_v3.json`):
1. `gold_snippets` stamped deterministically from source chunk text → enables
   answer-faithfulness eval, not just retrieval.
2. Hard substantive questions pair **numerically adjacent** sections (topical
   neighbourhood) instead of two random sections — realistic compounds.
   (`chapter_*` metadata is null in the chunks, so adjacency is the relatedness
   proxy; window ±5.)
3. **Verification pass** (LLM-as-judge, prompts `qgen_v3_verify*`) drops questions
   not answerable from their gold source; negatives judged genuinely out-of-domain.
   Toggle with `--no-verify`.
4. Curated **must-cover** marquee sections seeded first (murder, theft, cheating,
   rape, forgery, defamation, conspiracy, …).
5. Hard negatives are **near-domain** (CrPC/BNSS, evidence, CPC, constitutional) —
   the real refusal failure mode — via `qgen_v3_negative_near`.

Run: `python -m rag_pipeline.eval.qgen_v3 --out eval/eval_set_v3.json`.
Follow-up (separate task): point `/eval/retrieval` load at the new set and run
hit@k / recall / mrr; consider adding a faithfulness metric that uses gold_snippets.

### 2026-07-09 — Eval-set v2 generator committed

`b3dc495` Add evaluation set generation for legal RAG system.

### 2026-07-07 — Section-based eval metrics

`eb869fe` section-based evaluation metrics + new smoke test data. `evaluate_retriever`
and `/eval/retrieval` match on `gold_sections` via `_MultiCollectionRetriever`.
