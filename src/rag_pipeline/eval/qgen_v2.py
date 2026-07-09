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


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate section-based eval set (v2).")
    ap.add_argument("--out", type=Path, default=Path("eval/eval_set_v2.json"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Generate a small mixed test batch of N questions.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore any existing output file and start over "
                         "(default: resume from it).")
    args = ap.parse_args()
    generate(args.out, limit=args.limit, seed=args.seed, resume=not args.fresh)


if __name__ == "__main__":
    main()
