# HANDOFF.md — protoRAG session handoff

> Snapshot of project state at the point of moving from chat to Claude Code.
> Pair with `CLAUDE.md` (standing instructions). This file = current state,
> decisions with rationale, and ordered next steps.

---

## 1. Where we are — phase status

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundation: Docling parse → Chroma → naive retrieval | ✅ done |
| 1 | Hybrid retrieval: BM25 + dense + RRF + BGE reranker | ✅ done |
| 2 | Query intelligence (multi-query expansion) | 🟡 built, PARKED (raised refusal 5%→25%) |
| 3 | Agentic router (LangGraph, per-query intent) | ⏳ planned, not started |
| 4 | Eval framework (hit@k, recall, mrr, refusal) | ✅ done (but eval set NOT rebuilt for IPC+BNS — see §4) |
| 5 | Production hardening: FastAPI, SSE, auth, rate limit, CI, Docker | ✅ done |
| 5.5 | Streamlit dashboard | ✅ done |
| Corpus rebuild (A–E) | Dual IPC/BNS collections, structural parse, concordance | ✅ done |
| Concordance cross-reference feature | Query-time lookup, separate cross_reference block, page previews | ✅ done (this session) |
| Case-file upload + Q&A | Upload real cases, ask IPC/BNS interpretation, ISOLATED | 🚧 IN PROGRESS — see §3 |
| Corpus-agnostic refactor | Remove IPC/BNS hardcoding from API/serving layer | ⏳ planned, not started — see §5 |
| 6 | Cloud deploy (Azure models, managed vector store) | ⏳ future — fixes latency ceiling |
| 7.5 | Contextual layer (committee report + SOR) | ⏳ deferred — PDFs banked in data/raw/_deferred/ |

---

## 2. What got done THIS session (all committed or to-be-committed)

1. **Concordance query-time wiring.** Section mentions in a query (e.g. "IPC 420")
   trigger a deterministic concordance lookup; the mapping is injected into the LLM
   context so it can answer "IPC 420 corresponds to BNS 318". Files:
   `api/main.py` (`_concordance_context`, `_fetch_section_chunk`, `_concordance_citation`).

2. **Separate `cross_reference` response field/event.** Concordance results are no
   longer mixed into `citations`. New `CrossReference` schema
   (`api/schemas.py`) carries concordance_row, ipc/bns_section, status, page_number,
   and the real IPC + BNS section citations. Both `/answer` and `/answer/stream`
   populate it; streaming emits it as its own SSE event before `citations`.

3. **MAIN_CITATION_FLOOR = 0.55.** Filters reranker-noise (~0.50) from main
   citations so meta-queries show an empty main-citation list + the authoritative
   cross_reference block, while substantive queries keep their real ~0.72 citations.
   Applied in BOTH routes in `api/main.py`.

4. **Concordance page tracking.** `ConcordanceRow` gained `row_index` and
   `page_number` (`corpus/concordance.py`); parser stamps them; JSON regenerated.
   `/documents/page-image` endpoint extended to accept `CONCORDANCE` and render the
   concordance PDF page. Row 465 = IPC 420 ↔ BNS 318, concordance page 19.

5. **Dashboard rendering** (`apps/dashboard/app.py`): `render_cross_reference` panel,
   `_view_page` helper (inline PDF page images), Streamlit key-collision fix
   (`key_prefix="live"` for streaming, `f"hist{i}"` for history re-render).

6. **Health check fixed.** Was reading a non-existent `cfg.OLLAMA_PORT` and old
   `_state["chunks"]/["hybrid_r"]` keys → now uses `cfg.OLLAMA_HOST` and
   `_state["by_act"]`. Status shows healthy.

7. **Contract tests** updated for the `by_act` state structure and mock metadata.

**IMPORTANT — commit before opening Claude Code:**
```bash
git add -A
git commit -m "feat: concordance cross-reference block + citation floor filter + page previews"
```
Verify no `.env`, `chroma_db/`, `.venv/`, `data/processed/` are staged.

---

## 3. IN PROGRESS — case-file upload + Q&A (the active thread)

**Goal.** User uploads a real-life case file (judgment/FIR/charge sheet). They ask
which IPC/BNS sections apply, why, and how to interpret the case. The uploaded case
must be **strictly isolated**: never mixed into IPC_Corpus/BNS_Corpus, and never
leaked across sessions/users.

**Decision (production-grade, deliberate).** Default to **text-in-context**, NOT
per-session vector collections:
- Case file → parse → hold full text in a **session-scoped store with TTL**.
- On query: retrieve relevant IPC/BNS sections (existing pipeline) + inject the case
  text + concordance into the LLM prompt → interpretation with citations.
- Case text is CONTEXT; IPC/BNS vectors are never touched. Isolation is structural.
- Rationale: per-session ChromaDB collections are an isolation *liability* at scale
  (orphaned collections, cross-session leak risk, no reliable Streamlit session-end
  hook, unbounded storage). Escalate to an ephemeral isolated collection ONLY for
  case files too large for the context window, and only with strict TTL + GC +
  collision-proof `session_{uuid}` naming. Build that overflow path later, not now.

**Current state of the code:**
- `POST /documents/upload` EXISTS (`api/main.py` ~line 667): parses pdf/txt/md,
  stores full text via `_state["doc_store"].add(...)`. Uses `_state["uploaded_parser"]`.
- `DocumentStore` is in `src/rag_pipeline/api/documents.py` (class at line 44).
  **NOT YET REVIEWED for isolation** — the immediate next step is to read this class
  and determine if it's a flat global dict (leaks across users) or session-scoped.

**NEXT CONCRETE STEP (resume here):**
1. Read `src/rag_pipeline/api/documents.py` (the full `DocumentStore` class).
2. Add **session-scoping + TTL** to `DocumentStore` — uploads keyed by a per-client
   session token so user A's file is invisible to user B. THIS IS THE FIRST FIX,
   before building anything on top (production isolation must be correct first).
3. Update `/documents/upload` to take a session token.
4. Add `POST /answer/case`: takes session_id + case_doc_id + question → retrieves
   relevant IPC/BNS sections for the question → prompt = [CASE TEXT] + [retrieved
   IPC/BNS sections] + [concordance if sections named] + question → LLM interpretation
   → returns answer + IPC/BNS citations (case text stays context, never embedded).
5. Streamlit: file uploader, session-token generation, case-scoped Q&A panel.
6. TTL/cleanup; interface designed to swap in Redis for cloud.

**Open production parameters (get answers before finalizing design):**
- Production LLM context budget? (Assume Azure GPT-4o-class 128k in prod, gemma ~8k
  now → text-in-context is the primary path; vector-overflow rare.)
- Multi-user concurrent from day one? (Assume yes → DocumentStore MUST be
  session-scoped now; in-memory now, Redis-backed later.)

Do this on branch `feature/case-upload`.

---

## 4. Eval rebuild — REQUIRED, not yet done

The eval set was built for the OLD IPC-only corpus. The IPC+BNS rebuild is
**unmeasured**. Before trusting the new corpus / tuning thresholds:
- Build 30–50 mixed questions: ~15 IPC-only, ~15 BNS-only, ~10 cross-reference
  ("what is IPC 420 called in BNS?"), ~10 negatives (out-of-domain).
- Run `/eval/retrieval` and `/eval/threshold-sweep` against the new corpus.
- Re-calibrate `min_score` AND `MAIN_CITATION_FLOOR` (currently 0.55, a heuristic
  guess) against real data. Add precision@k / nDCG (currently missing).
- Acceptance target: hit@k ≥ 0.92, mrr ≥ 0.70, refusal ≥ 0.9.
- Only 6 negatives existed in the old eval → refusal F1 was coarse; expand.

---

## 5. Corpus-agnostic refactor — planned

The INGEST layer is already generic (YAML corpus registry, `load_corpus`,
`StatuteParser`/retrievers/ingest take config). The API/SERVING layer still hardcodes
IPC/BNS. To remove (on branch `feature/corpus-agnostic`):

| Hardcoded | Where | Fix |
|---|---|---|
| `[("IPC","IPC_Corpus"),("BNS","BNS_Corpus")]` | lifespan | iterate `corpus.sources` |
| `collections: list[Literal["IPC","BNS"]]` | AnswerRequest schema | `list[str]`, validate at runtime |
| `_concordance_context` IPC/BNS lookup | routes | generic optional "cross-reference map" per corpus |
| Prompt "Indian Penal Code / BNS" | routes | from `corpus.display_name` / config |
| `pdf_map={"IPC":..,"BNS":..,"CONCORDANCE":..}` | `/documents/page-image` | build from `corpus.sources` |
| `CrossReference` ipc_section/bns_section fields | schema | generic source/target with act labels as data |

Open design questions to resolve first:
- Is cross-reference/concordance a GENERAL feature (some corpora have mapping tables,
  some don't → make it an optional per-corpus module) or IPC/BNS-only?
- How many acts can a corpus have? (1? 2? N federal+state codes?) → design N-collection.
- Sequence: developer leans "finish concordance + case-upload, THEN generalize."
  Reasonable, but note we'll be generalizing code we just wrote. Do eval first if you
  want to generalize measured code.

---

## 6. Known non-blocking issues

- `chapter_number = None` on all chunks (chapter regex doesn't match the PDF format).
- 40 IPC chunks didn't enrich with concordance (subsection ID format mismatch, e.g.
  "302" vs "302(1)"). Main section chunk still carries the mapping.
- IPC chunk count 808 > ~511 main sections (subsections double-counted by design).
- `bns_title='(4)'` artifact on some concordance rows (title extraction imperfect;
  mapping itself is correct).
- Latency ceiling ~30–140s from local Ollama (fixed only by cloud LLM, Phase 6).
- swig DeprecationWarning + token>512 truncation warning — both harmless.
- Reranker on 8GB GPU can OOM if orphaned processes hold VRAM → `nvidia-smi`, kill,
  restart; or move reranker to CPU.

---

## 7. GitHub / workflow

- Company repo: `Advantev/adv_policy_engine`, remote name `advantev`. Personal:
  `himmng/protoRAG`, remote `origin`.
- Code was pushed to `advantev` `main` (merged the company's initial commit with
  `--allow-unrelated-histories`, then force-pushed the developer's version).
- **Going forward: feature branches → PR → merge to main. No direct pushes to main.**
  Conventional commits. Consider a branch-protection rule requiring CI + PR on `main`.
- CI (`.github/workflows/ci.yml`) runs pytest on push/PR. Keep it green before PRs.

---

## 8. Recommended next-step order (from here)

1. **Commit this session's work** (§2) so Claude Code sees current state.
2. **Finish case-upload** (§3) on `feature/case-upload` — start by reading
   `documents.py` and adding session isolation + TTL to `DocumentStore`.
3. **Eval rebuild** (§4) — measure the IPC+BNS corpus; recalibrate min_score +
   MAIN_CITATION_FLOOR on real data.
4. **Corpus-agnostic refactor** (§5) on `feature/corpus-agnostic`.
5. **Cloud deploy** (Phase 6) — `MODEL_PROVIDER=azure`, managed vector store; fixes
   the latency ceiling (p95 140s → ~5s).
6. Agentic router (Phase 3) + contextual layer (7.5) once the above are solid.

---

## 9. First prompt to use in Claude Code

> Read `CLAUDE.md` and `docs/HANDOFF.md`. We're mid-way through the case-upload
> feature (§3). The next step is adding session-scoping and TTL to `DocumentStore`
> in `src/rag_pipeline/api/documents.py` for strict per-user isolation. Show me the
> current `DocumentStore` class, then propose the isolation fix with exact edits.