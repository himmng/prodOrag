"""Generate a section-based eval set (v2) for the legal RAG system.

Ground-truth principle: never let the model invent the gold answer. Every
question is generated FROM a real source (a StatuteChunk or a concordance row),
so gold_sections is known by construction and stamped by us — the model only
writes the question + reference_answer.

Four categories:
  - ipc_substantive / bns_substantive: feed one section's text, gold = that section.
    (hard: feed TWO sections from the same act, gold = both.)
  - cross_reference: feed a concordance row's two provisions, gold = both sections.
  - negative: no source, out-of-domain question, gold = [] (system must refuse).

Bad LLM outputs are logged + skipped (final count may be slightly below target).
Sources are sampled WITHOUT replacement within a run so questions don't duplicate.
Saves incrementally to disk for crash safety.

CLI:
    python -m rag_pipeline.eval.qgen_v2 --out eval/eval_set_v2.json
    python -m rag_pipeline.eval.qgen_v2 --out eval/eval_set_v2.json --limit 10
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage
from tqdm import tqdm

from rag_pipeline.config import cfg, log
from rag_pipeline.eval.schema import EvalExample, load_eval_set, save_eval_set
from rag_pipeline.prompts import render_prompt
from rag_pipeline.providers import get_llm

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

# ── Source loading + filtering ─────────────────────────────────────────

_PROCESSED = cfg.PROJECT_ROOT / "data" / "processed"
_MIN_CHARS = 120

# Editorial-noise section titles (footnote/amendment fragments the statute
# parser mis-tagged as sections). Mirrors api/main.py::_NOISE_TITLE.
_NOISE_TITLE = re.compile(
    r"^\s*(subs\b|ins\b|rep\b|added by|omitted|certain words|clause\b|"
    r"the words?\b|the brackets|the proviso|section\s+\d+\s+re-?numbered|"
    r"explanation numbered|illustrations rep|the indian penal code has been extended)",
    re.IGNORECASE,
)

# Out-of-domain topic hints to steer negative-question variety.
_NEGATIVE_DOMAINS = [
    "Indian company / corporate law",
    "income tax and GST",
    "contract and commercial disputes",
    "labour and employment law",
    "intellectual property (patents, trademarks)",
    "property and real-estate registration",
    "family law (marriage, inheritance)",
    "environmental regulation",
    "general trivia unrelated to law",
    "software engineering and programming",
]

# Per-difficulty instruction injected into substantive / cross-reference prompts.
_DIFFICULTY_HINT = {
    "substantive": {
        "easy": "Ask for a SINGLE explicit fact stated in the section "
                "(a number, name, punishment, or exact phrase).",
        "medium": "Require combining 2-3 facts or understanding a condition "
                  "within the section to answer.",
    },
    "cross_reference": {
        "easy": "Ask simply what the corresponding provision in the other act is.",
        "medium": "Ask about the correspondence and one notable aspect of how the "
                  "provision is treated across the two acts.",
        "hard": "Ask for a substantive comparison of the actual text of the two "
                "provisions — what changed or stayed the same, in detail.",
    },
    "negative": {
        "easy": "Make it clearly unrelated to criminal law.",
        "medium": "Make it a plausible legal-sounding question in a DIFFERENT legal "
                  "domain, so it still must be refused.",
        "hard": "Make it a sophisticated, legal-sounding question in a different "
                "legal domain that could tempt an over-eager system to answer.",
    },
}


def _is_noise(chunk: dict) -> bool:
    title = (chunk.get("section_title") or "").strip()
    return bool(_NOISE_TITLE.match(title))


def _load_chunks(name: str) -> list[dict]:
    """Load + filter StatuteChunk dicts: drop short and editorial-noise chunks."""
    path = _PROCESSED / name
    chunks = json.loads(path.read_text(encoding="utf-8"))
    good = [
        c for c in chunks
        if len((c.get("text") or "")) >= _MIN_CHARS and not _is_noise(c)
    ]
    log.info(f"Loaded {len(good)}/{len(chunks)} usable chunks ← {name}")
    return good


def _load_concordance() -> list[dict]:
    """Load concordance rows where BOTH ipc_section and bns_section are present."""
    path = _PROCESSED / "concordance.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    good = [r for r in rows if r.get("ipc_section") and r.get("bns_section")]
    log.info(f"Loaded {len(good)}/{len(rows)} mapped concordance rows")
    return good


# ── Parse helpers (mirrors old qgen.py) ────────────────────────────────

def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers even though the prompt forbids them."""
    return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()


def _invoke_json(llm: "BaseChatModel", prompt: str, tag: str) -> dict | None:
    """Invoke the LLM and parse strict JSON; log + skip on any failure.

    Catches broadly (not just parse errors): Azure raises ValueError when a
    content filter trips, and providers can raise transient API errors. A single
    failed item must never abort the whole run — we log and skip it. Interrupts
    (KeyboardInterrupt/SystemExit are BaseException) still propagate.
    """
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        data = json.loads(_strip_json_fences(resp.content))
        q = (data.get("question") or "").strip()
        if not q:
            raise KeyError("question")
        return {
            "question": q,
            "reference_answer": (data.get("reference_answer") or "").strip() or None,
        }
    except Exception as e:  # noqa: BLE001 — deliberate: skip any bad item, keep going
        log.warning(f"[qgen_v2-{tag}] generation/parse failed, skipping: {e}")
        return None


# ── Per-category builders ──────────────────────────────────────────────

def _build_substantive(
    act: str, category: str, difficulty: str,
    chunks: list[dict], rng: random.Random,
) -> EvalExample | None:
    if difficulty == "hard":
        if len(chunks) < 2:
            return None
        a, b = chunks.pop(), chunks.pop()
        prompt = render_prompt(
            "qgen_v2_substantive_hard", act=act, passage_a=a["text"], passage_b=b["text"],
        )
        parsed = _invoke_json(llm=_LLM, prompt=prompt, tag=category)
        if not parsed:
            return None
        gold = [{"act": act, "section": a["section"]},
                {"act": act, "section": b["section"]}]
        notes = f"auto-gen from chunks {a['chunk_id']}, {b['chunk_id']}"
    else:
        c = chunks.pop()
        prompt = render_prompt(
            "qgen_v2_substantive", act=act, difficulty=difficulty,
            difficulty_hint=_DIFFICULTY_HINT["substantive"][difficulty],
            passage=c["text"],
        )
        parsed = _invoke_json(llm=_LLM, prompt=prompt, tag=category)
        if not parsed:
            return None
        gold = [{"act": act, "section": c["section"]}]
        notes = f"auto-gen from chunk {c['chunk_id']}"
    return EvalExample(
        question=parsed["question"], category=category, difficulty=difficulty,
        gold_sections=gold, reference_answer=parsed["reference_answer"], notes=notes,
    )


def _build_cross_reference(
    difficulty: str, ipc_by_section: dict, bns_by_section: dict,
    rows: list[dict], rng: random.Random,
) -> EvalExample | None:
    row = rows.pop()
    ipc_sec, bns_sec = row["ipc_section"], row["bns_section"]
    ipc_chunk = ipc_by_section.get(ipc_sec)
    bns_chunk = bns_by_section.get(bns_sec)
    if not ipc_chunk or not bns_chunk:
        return None
    prompt = render_prompt(
        "qgen_v2_cross_reference", act_a="IPC", act_b="BNS",
        difficulty=difficulty, difficulty_hint=_DIFFICULTY_HINT["cross_reference"][difficulty],
        title_a=ipc_chunk.get("section_title") or "", title_b=bns_chunk.get("section_title") or "",
        passage_a=ipc_chunk["text"], passage_b=bns_chunk["text"],
    )
    parsed = _invoke_json(llm=_LLM, prompt=prompt, tag="cross_reference")
    if not parsed:
        return None
    return EvalExample(
        question=parsed["question"], category="cross_reference", difficulty=difficulty,
        gold_sections=[{"act": "IPC", "section": ipc_sec},
                       {"act": "BNS", "section": bns_sec}],
        reference_answer=parsed["reference_answer"],
        notes=f"auto-gen from concordance row {row.get('row_index')}",
    )


def _build_negative(difficulty: str, rng: random.Random) -> EvalExample | None:
    domain = rng.choice(_NEGATIVE_DOMAINS)
    prompt = render_prompt(
        "qgen_v2_negative", difficulty=difficulty,
        difficulty_hint=_DIFFICULTY_HINT["negative"][difficulty], domain=domain,
    )
    parsed = _invoke_json(llm=_LLM, prompt=prompt, tag="negative")
    if not parsed:
        return None
    return EvalExample(
        question=parsed["question"], category="negative", difficulty=difficulty,
        gold_sections=[], reference_answer=parsed["reference_answer"],
        notes=f"auto-gen negative ({domain})",
    )


# ── Distribution plan ──────────────────────────────────────────────────

# (category, difficulty) -> count. Total 200, 60/60/80 easy/medium/hard.
PLAN: dict[tuple[str, str], int] = {
    ("ipc_substantive", "easy"): 18, ("ipc_substantive", "medium"): 16, ("ipc_substantive", "hard"): 20,
    ("bns_substantive", "easy"): 18, ("bns_substantive", "medium"): 16, ("bns_substantive", "hard"): 20,
    ("cross_reference", "easy"): 12, ("cross_reference", "medium"): 16, ("cross_reference", "hard"): 20,
    ("negative", "easy"): 12, ("negative", "medium"): 12, ("negative", "hard"): 20,
}

# Module-level LLM handle (set in generate()).
_LLM: "BaseChatModel" = None  # type: ignore


def _load_resume_state(out_path: Path):
    """Load an existing partial eval set to resume from.

    Returns (examples, done_counts, used_chunk_ids, used_rows). Sources already
    consumed are parsed back out of each example's `notes` so we neither reuse a
    source nor re-spend the API call that produced it.
    """
    from collections import Counter

    existing = load_eval_set(out_path)
    done: Counter = Counter((e.category, e.difficulty) for e in existing)
    used_chunk_ids: set[str] = set()
    used_rows: set[int] = set()
    for e in existing:
        note = e.notes or ""
        used_chunk_ids.update(re.findall(r"\b[0-9a-f]{16}\b", note))
        m = re.search(r"concordance row (\d+)", note)
        if m:
            used_rows.add(int(m.group(1)))
    return existing, done, used_chunk_ids, used_rows


def generate(out_path: Path, limit: int | None = None, seed: int = 42,
             save_every: int = 10, resume: bool = True) -> list[EvalExample]:
    """Run the PLAN, generating section-based eval examples.

    If `resume` and out_path already exists, continue from that partial set:
    completed (category, difficulty) counts are subtracted from the plan and
    already-used sources are excluded, so no prior API call is wasted on restart.
    """
    global _LLM
    _LLM = get_llm()
    rng = random.Random(seed)

    ipc = _load_chunks("ipc_chunks.json")
    bns = _load_chunks("bns_chunks.json")
    conc = _load_concordance()
    rng.shuffle(ipc); rng.shuffle(bns); rng.shuffle(conc)

    ipc_by_section = {c["section"]: c for c in ipc}
    bns_by_section = {c["section"]: c for c in bns}

    # Optionally scale the plan down for a quick test batch.
    plan = dict(PLAN)
    if limit is not None:
        plan = _scaled_plan(limit)

    # Resume: load partial set, drop completed counts + already-used sources.
    examples: list[EvalExample] = []
    used_chunk_ids: set[str] = set()
    used_rows: set[int] = set()
    if resume and out_path.exists():
        examples, done, used_chunk_ids, used_rows = _load_resume_state(out_path)
        for key in list(plan):
            plan[key] = max(0, plan[key] - done.get(key, 0))
        log.info(f"Resuming from {len(examples)} existing examples; "
                 f"{sum(plan.values())} remaining to generate.")

    # Per-category source pools consumed via .pop() (sampling w/o replacement),
    # with already-used sources excluded so resume never reuses a source.
    pool = {
        "ipc_substantive": [c for c in ipc if c["chunk_id"] not in used_chunk_ids],
        "bns_substantive": [c for c in bns if c["chunk_id"] not in used_chunk_ids],
        "cross_reference": [r for r in conc if r.get("row_index") not in used_rows],
    }

    target_new = sum(plan.values())
    total = target_new + len(examples)

    def maybe_save():
        if len(examples) and len(examples) % save_every == 0:
            save_eval_set(examples, out_path)

    with tqdm(total=target_new, desc="qgen_v2") as bar:
        for (category, difficulty), count in plan.items():
            for _ in range(count):
                bar.update(1)
                bar.set_postfix_str(f"{category}/{difficulty}")
                ex = _dispatch(category, difficulty, pool, ipc_by_section,
                               bns_by_section, rng)
                if ex:
                    examples.append(ex)
                    maybe_save()

    save_eval_set(examples, out_path)
    log.info(f"Generated {len(examples)}/{total} eval examples "
             f"(target {sum(PLAN.values())}) → {out_path}")
    return examples


def _dispatch(category, difficulty, pool, ipc_by_section, bns_by_section, rng):
    if category == "ipc_substantive":
        return _build_substantive("IPC", category, difficulty, pool["ipc_substantive"], rng)
    if category == "bns_substantive":
        return _build_substantive("BNS", category, difficulty, pool["bns_substantive"], rng)
    if category == "cross_reference":
        return _build_cross_reference(difficulty, ipc_by_section, bns_by_section,
                                      pool["cross_reference"], rng)
    if category == "negative":
        return _build_negative(difficulty, rng)
    return None


def _scaled_plan(limit: int) -> dict[tuple[str, str], int]:
    """A small mixed plan for --limit test batches: spread across all categories."""
    cats = ["ipc_substantive", "bns_substantive", "cross_reference", "negative"]
    diffs = ["easy", "medium", "hard"]
    plan: dict[tuple[str, str], int] = {}
    i = 0
    while sum(plan.values()) < limit:
        key = (cats[i % len(cats)], diffs[(i // len(cats)) % len(diffs)])
        plan[key] = plan.get(key, 0) + 1
        i += 1
    return plan


# ═══════════════════════════════════════════════════════════════════════
# ADVERSARIAL categories — conflict, hard_negative, distractor.
# These test how the system FAILS in production, not just the happy path.
# Ground truth (gold_sections / expected_behavior / failure_modes) is stamped
# by construction from the concordance + curated offence families, never by
# the model. Output goes to eval/eval_set_adversarial.json (separate set).
# ═══════════════════════════════════════════════════════════════════════

# Adjacent legal domains for hard negatives (legal register, NOT IPC/BNS).
_HARD_NEG_AREAS = [
    "Indian company / corporate law (Companies Act, SEBI regulations)",
    "income tax and GST law",
    "the Code of Civil Procedure — civil suits, decrees, execution",
    "contract and commercial law (Indian Contract Act)",
    "cybercrime / data offences under the IT Act not mirrored in the IPC/BNS",
    "labour and industrial-dispute law",
]

# Distractor families: semantically-adjacent but legally-DISTINCT offences.
# (section, offence-label); target and neighbour are drawn from DIFFERENT labels
# so the pair always contrasts two real offences, never two chunks of one.
_DISTRACTOR_FAMILIES: list[tuple[str, list[tuple[str, str]]]] = [
    ("IPC", [("378", "theft"), ("383", "extortion"),
             ("390", "robbery"), ("391", "dacoity")]),
    ("IPC", [("420", "cheating"), ("406", "criminal breach of trust")]),
    ("IPC", [("323", "hurt"), ("325", "grievous hurt")]),
    ("BNS", [("303", "theft"), ("308", "extortion"), ("309", "robbery")]),
]

# Full-run adversarial plan (mode -> count). Total 100.
ADV_PLAN: list[tuple[str, int]] = [
    ("conflict_changed", 26),
    ("conflict_deleted", 14),
    ("hard_negative", 30),
    ("distractor", 30),
]


def _build_conflict_changed(row, ipc_by, bns_by):
    ic, bc = ipc_by[row["ipc_section"]], bns_by[row["bns_section"]]
    prompt = render_prompt(
        "qgen_adv_conflict_changed",
        ipc_title=ic.get("section_title") or "", bns_title=bc.get("section_title") or "",
        ipc_text=ic["text"], bns_text=bc["text"])
    parsed = _invoke_json(llm=_LLM, prompt=prompt, tag="conflict_changed")
    if not parsed:
        return None
    return EvalExample(
        question=parsed["question"], category="conflict", difficulty="medium",
        gold_sections=[{"act": "IPC", "section": row["ipc_section"]},
                       {"act": "BNS", "section": row["bns_section"]}],
        reference_answer=parsed["reference_answer"],
        expected_behavior=("must present BOTH acts and state the difference; "
                           "must not silently pick one"),
        failure_modes=["silent_act_selection"],
        notes=f"auto-gen conflict/changed from concordance row {row.get('row_index')}")


def _build_conflict_deleted(row, ipc_by):
    ic = ipc_by[row["ipc_section"]]
    prompt = render_prompt(
        "qgen_adv_conflict_deleted", ipc_section=row["ipc_section"],
        ipc_title=ic.get("section_title") or "", ipc_text=ic["text"])
    parsed = _invoke_json(llm=_LLM, prompt=prompt, tag="conflict_deleted")
    if not parsed:
        return None
    return EvalExample(
        question=parsed["question"], category="conflict", difficulty="hard",
        gold_sections=[{"act": "IPC", "section": row["ipc_section"]}],
        reference_answer=parsed["reference_answer"],
        expected_behavior=("must state the section was repealed/omitted under the "
                           "BNS; must not recite it as current law"),
        failure_modes=["stale_law"],
        notes=f"auto-gen conflict/deleted from concordance row {row.get('row_index')}")


def _build_hard_negative_adv(rng):
    area = rng.choice(_HARD_NEG_AREAS)
    prompt = render_prompt("qgen_adv_hard_negative", area=area)
    parsed = _invoke_json(llm=_LLM, prompt=prompt, tag="hard_negative")
    if not parsed:
        return None
    return EvalExample(
        question=parsed["question"], category="hard_negative", difficulty="hard",
        gold_sections=[], reference_answer=parsed["reference_answer"],
        expected_behavior=("must refuse / state not covered; must not stretch an "
                           "unrelated section to answer"),
        failure_modes=["failed_to_refuse", "overreach"],
        notes=f"auto-gen hard_negative ({area})")


def _build_distractor(rng, ipc_by, bns_by):
    act, members = rng.choice(_DISTRACTOR_FAMILIES)
    by = ipc_by if act == "IPC" else bns_by
    avail = [(s, lbl) for s, lbl in members if s in by]
    if len(avail) < 2:
        return None
    (t_sec, t_lbl), (n_sec, n_lbl) = rng.sample(avail, 2)
    tc, nc = by[t_sec], by[n_sec]
    prompt = render_prompt(
        "qgen_adv_distractor", act=act,
        target_title=tc.get("section_title") or "", target_text=tc["text"],
        neighbour_title=nc.get("section_title") or "", neighbour_text=nc["text"])
    parsed = _invoke_json(llm=_LLM, prompt=prompt, tag="distractor")
    if not parsed:
        return None
    return EvalExample(
        question=parsed["question"], category="distractor", difficulty="hard",
        gold_sections=[{"act": act, "section": t_sec}],
        reference_answer=parsed["reference_answer"],
        expected_behavior=("retrieval must select the target offence, not the "
                           f"semantically-adjacent neighbour ({n_lbl})"),
        failure_modes=["distractor_confusion"],
        notes=f"auto-gen distractor target {act} {t_sec} ({t_lbl}) vs "
              f"neighbour {n_sec} ({n_lbl}); chunk {tc['chunk_id']}")


def _adv_scaled_plan(limit: int) -> list[tuple[str, int]]:
    """Small mixed adversarial batch: round-robin across all four modes."""
    modes = [m for m, _ in ADV_PLAN]
    counts = {m: 0 for m in modes}
    i = 0
    while sum(counts.values()) < limit:
        counts[modes[i % len(modes)]] += 1
        i += 1
    return [(m, counts[m]) for m in modes]


def generate_adversarial(out_path: Path, limit: int | None = None, seed: int = 42,
                         save_every: int = 10, resume: bool = True):
    """Generate the adversarial eval set (conflict / hard_negative / distractor)."""
    global _LLM
    _LLM = get_llm()
    rng = random.Random(seed)

    ipc = _load_chunks("ipc_chunks.json")
    bns = _load_chunks("bns_chunks.json")
    ipc_by = {c["section"]: c for c in ipc}
    bns_by = {c["section"]: c for c in bns}

    # Concordance rows carrying status (raw load — deleted rows have null BNS).
    raw = json.loads((_PROCESSED / "concordance.json").read_text(encoding="utf-8"))
    changed = [r for r in raw if r.get("status") == "changed"
               and r.get("ipc_section") in ipc_by and r.get("bns_section") in bns_by]
    deleted = [r for r in raw if r.get("status") == "deleted"
               and r.get("ipc_section") in ipc_by]
    rng.shuffle(changed); rng.shuffle(deleted)

    plan = _adv_scaled_plan(limit) if limit is not None else list(ADV_PLAN)

    # Resume: subtract completed per-mode counts + exclude used concordance rows.
    examples: list[EvalExample] = []
    used_rows: set[int] = set()
    if resume and out_path.exists():
        from collections import Counter
        examples = load_eval_set(out_path)
        done: Counter = Counter()
        for e in examples:
            note = e.notes or ""
            for mode in ("conflict/changed", "conflict/deleted", "hard_negative", "distractor"):
                if mode in note:
                    done[mode.replace("/", "_")] += 1
            m = re.search(r"concordance row (\d+)", note)
            if m:
                used_rows.add(int(m.group(1)))
        plan = [(mode, max(0, cnt - done.get(mode, 0))) for mode, cnt in plan]
        log.info(f"Resuming adversarial from {len(examples)} existing; "
                 f"{sum(c for _, c in plan)} to go.")
    changed = [r for r in changed if r.get("row_index") not in used_rows]
    deleted = [r for r in deleted if r.get("row_index") not in used_rows]

    target_new = sum(c for _, c in plan)

    def maybe_save():
        if len(examples) and len(examples) % save_every == 0:
            save_eval_set(examples, out_path)

    with tqdm(total=target_new, desc="qgen_adv") as bar:
        for mode, count in plan:
            for _ in range(count):
                bar.update(1); bar.set_postfix_str(mode)
                if mode == "conflict_changed":
                    ex = _build_conflict_changed(changed.pop(), ipc_by, bns_by) if changed else None
                elif mode == "conflict_deleted":
                    ex = _build_conflict_deleted(deleted.pop(), ipc_by) if deleted else None
                elif mode == "hard_negative":
                    ex = _build_hard_negative_adv(rng)
                elif mode == "distractor":
                    ex = _build_distractor(rng, ipc_by, bns_by)
                else:
                    ex = None
                if ex:
                    examples.append(ex); maybe_save()

    save_eval_set(examples, out_path)
    log.info(f"Generated {len(examples)} adversarial eval examples "
             f"(target {sum(c for _, c in ADV_PLAN)}) → {out_path}")
    return examples


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate section-based eval set (v2).")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Generate a small mixed test batch of N questions.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore any existing output file and start over "
                         "(default: resume from it).")
    ap.add_argument("--adversarial", action="store_true",
                    help="Generate the adversarial set (conflict / hard_negative / "
                         "distractor) instead of the standard set.")
    args = ap.parse_args()
    if args.adversarial:
        out = args.out or Path("eval/eval_set_adversarial.json")
        generate_adversarial(out, limit=args.limit, seed=args.seed, resume=not args.fresh)
    else:
        out = args.out or Path("eval/eval_set_v2.json")
        generate(out, limit=args.limit, seed=args.seed, resume=not args.fresh)


if __name__ == "__main__":
    main()
