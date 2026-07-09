"""Generate a section-based eval set (v3) — production-grade upgrade of v2.

Same ground-truth-by-construction principle as v2 (gold is stamped from the real
source, never invented by the model), plus five quality upgrades:

  1. gold_snippets stamped DETERMINISTICALLY from the source chunk text — enables
     answer-faithfulness/grounding eval, not just retrieval eval. Ground-truth,
     zero extra API cost.
  2. Hard substantive questions pair NUMERICALLY ADJACENT sections (same topical
     neighbourhood in codified law) instead of two random sections, so compound
     questions are realistic rather than Frankenstein stitches.
  3. A VERIFICATION pass (LLM-as-judge) drops questions not genuinely answerable
     from their gold source; negatives are judged genuinely out-of-domain.
  4. A curated MUST-COVER list guarantees marquee offences (murder, theft,
     cheating, rape, forgery, defamation, …) appear in the set.
  5. Hard negatives are NEAR-DOMAIN (procedure/evidence/other codes) — the real
     failure mode — rather than only far-domain topics.

Reuses v2 primitives (fence-strip, JSON-parse guard, chunk/concordance loaders,
atomic incremental save, resume-on-restart). Bad LLM outputs are logged + skipped.

CLI:
    python -m rag_pipeline.eval.qgen_v3 --out eval/eval_set_v3.json
    python -m rag_pipeline.eval.qgen_v3 --out eval/eval_set_v3.json --limit 12
    python -m rag_pipeline.eval.qgen_v3 --out eval/eval_set_v3.json --no-verify
    python -m rag_pipeline.eval.qgen_v3 --out eval/eval_set_v3.json --fresh
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

from rag_pipeline.config import log
from rag_pipeline.eval.schema import EvalExample, load_eval_set, save_eval_set
from rag_pipeline.prompts import render_prompt
from rag_pipeline.providers import get_llm

# Reuse v2 primitives (single source of truth for these).
from rag_pipeline.eval.qgen_v2 import (
    PLAN,
    _DIFFICULTY_HINT,
    _NEGATIVE_DOMAINS,
    _invoke_json,
    _load_chunks,
    _load_concordance,
    _scaled_plan,
    _strip_json_fences,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

# ── Config knobs ───────────────────────────────────────────────────────

_SNIPPET_CHARS = 500       # gold_snippet trim length
_ADJ_WINDOW = 5            # max |section_num_a - section_num_b| for hard pairing

# Marquee sections we insist the set covers (best-effort: missing ones skipped).
_MUST_COVER = {
    "IPC": ["302", "304B", "307", "323", "324", "326", "354", "375", "376",
            "378", "379", "380", "390", "392", "395", "415", "420", "463",
            "468", "499", "500", "120B", "406", "409", "498A", "34", "149"],
    "BNS": ["103", "63", "64", "69", "115", "117", "118", "303", "304", "309",
            "310", "318", "319", "336", "338", "356", "61", "85", "3", "111"],
}

# Near-domain areas for HARD negatives — sound like criminal law, actually aren't.
_NEAR_DOMAIN_AREAS = [
    "Code of Criminal Procedure / BNSS — arrest procedure, bail, FIR, charge framing",
    "Indian Evidence Act / Bharatiya Sakshya Adhiniyam — admissibility, burden of proof",
    "Civil Procedure Code — suits, decrees, civil jurisdiction",
    "Constitutional law — fundamental rights, writ jurisdiction, habeas corpus",
    "limitation periods and the appeals/revision procedure",
    "juvenile justice and probation administration",
]

# Module-level singletons (set in generate()).
_LLM: "BaseChatModel" = None  # type: ignore
_VERIFY: bool = True


# ── Small helpers ──────────────────────────────────────────────────────

def _snippet(text: str, limit: int = _SNIPPET_CHARS) -> str:
    """Whitespace-normalise and trim a chunk's text to a gold snippet."""
    s = " ".join((text or "").split())
    if len(s) <= limit:
        return s
    return s[:limit].rsplit(" ", 1)[0] + " …"


def _sec_num(section: str) -> int | None:
    """Leading integer of a section id ('302A' -> 302); None if unparseable."""
    m = re.match(r"\d+", str(section or ""))
    return int(m.group()) if m else None


def _invoke_raw(prompt: str, tag: str) -> dict | None:
    """Invoke the LLM and parse an arbitrary JSON object (for verify prompts)."""
    try:
        resp = _LLM.invoke([HumanMessage(content=prompt)])
        return json.loads(_strip_json_fences(resp.content))
    except Exception as e:  # noqa: BLE001 — a failed judge must not abort the run
        log.warning(f"[qgen_v3-{tag}] verify failed, treating as reject: {e}")
        return None


def _verify_positive(question: str, source: str) -> bool:
    if not _VERIFY:
        return True
    data = _invoke_raw(
        render_prompt("qgen_v3_verify", question=question, source=source), "verify")
    return bool(data and data.get("answerable") is True)


def _verify_xref(question: str, source: str) -> bool:
    """Verify a cross-reference question. The correspondence is authoritative
    context (concordance), so — unlike substantive verify — we do NOT require the
    mapping to be derivable from the two provision texts."""
    if not _VERIFY:
        return True
    data = _invoke_raw(render_prompt(
        "qgen_v3_verify_xref", question=question, act_a="IPC", act_b="BNS",
        source=source), "verify-xref")
    return bool(data and data.get("answerable") is True)


def _verify_negative(question: str) -> bool:
    if not _VERIFY:
        return True
    data = _invoke_raw(
        render_prompt("qgen_v3_verify_negative", question=question), "verify-neg")
    return bool(data and data.get("out_of_domain") is True)


# ── Adjacency pairing for hard substantive ─────────────────────────────

def _pop_adjacent_pair(pool: list[dict]) -> tuple[dict, dict] | None:
    """Pop chunk A, then the nearest-numbered chunk B from a DIFFERENT section
    within _ADJ_WINDOW. Falls back to a random second chunk if no neighbour."""
    if len(pool) < 2:
        return None
    a = pool.pop()
    a_num = _sec_num(a["section"])
    best_i, best_d = None, None
    if a_num is not None:
        for i, c in enumerate(pool):
            if c["section"] == a["section"]:
                continue
            n = _sec_num(c["section"])
            if n is None:
                continue
            d = abs(n - a_num)
            if d <= _ADJ_WINDOW and (best_d is None or d < best_d):
                best_i, best_d = i, d
    b = pool.pop(best_i) if best_i is not None else pool.pop()
    return a, b


# ── Per-category builders (each returns EvalExample or None) ────────────

def _build_substantive(act, category, difficulty, chunks, rng):
    if difficulty == "hard":
        pair = _pop_adjacent_pair(chunks)
        if not pair:
            return None
        a, b = pair
        prompt = render_prompt(
            "qgen_v2_substantive_hard", act=act, passage_a=a["text"], passage_b=b["text"])
        parsed = _invoke_json(llm=_LLM, prompt=prompt, tag=category)
        if not parsed or not _verify_positive(parsed["question"], f"{a['text']}\n\n{b['text']}"):
            return None
        return EvalExample(
            question=parsed["question"], category=category, difficulty=difficulty,
            gold_sections=[{"act": act, "section": a["section"]},
                           {"act": act, "section": b["section"]}],
            gold_snippets=[_snippet(a["text"]), _snippet(b["text"])],
            reference_answer=parsed["reference_answer"],
            notes=f"auto-gen from adjacent chunks {a['chunk_id']}, {b['chunk_id']}")
    c = chunks.pop()
    prompt = render_prompt(
        "qgen_v2_substantive", act=act, difficulty=difficulty,
        difficulty_hint=_DIFFICULTY_HINT["substantive"][difficulty], passage=c["text"])
    parsed = _invoke_json(llm=_LLM, prompt=prompt, tag=category)
    if not parsed or not _verify_positive(parsed["question"], c["text"]):
        return None
    return EvalExample(
        question=parsed["question"], category=category, difficulty=difficulty,
        gold_sections=[{"act": act, "section": c["section"]}],
        gold_snippets=[_snippet(c["text"])],
        reference_answer=parsed["reference_answer"],
        notes=f"auto-gen from chunk {c['chunk_id']}")


def _build_cross_reference(difficulty, ipc_by_section, bns_by_section, rows, rng):
    row = rows.pop()
    ipc_sec, bns_sec = row["ipc_section"], row["bns_section"]
    ipc_chunk = ipc_by_section.get(ipc_sec)
    bns_chunk = bns_by_section.get(bns_sec)
    if not ipc_chunk or not bns_chunk:
        return None
    prompt = render_prompt(
        "qgen_v2_cross_reference", act_a="IPC", act_b="BNS", difficulty=difficulty,
        difficulty_hint=_DIFFICULTY_HINT["cross_reference"][difficulty],
        title_a=ipc_chunk.get("section_title") or "", title_b=bns_chunk.get("section_title") or "",
        passage_a=ipc_chunk["text"], passage_b=bns_chunk["text"])
    parsed = _invoke_json(llm=_LLM, prompt=prompt, tag="cross_reference")
    src = f"{ipc_chunk['text']}\n\n{bns_chunk['text']}"
    if not parsed or not _verify_xref(parsed["question"], src):
        return None
    return EvalExample(
        question=parsed["question"], category="cross_reference", difficulty=difficulty,
        gold_sections=[{"act": "IPC", "section": ipc_sec},
                       {"act": "BNS", "section": bns_sec}],
        gold_snippets=[_snippet(ipc_chunk["text"]), _snippet(bns_chunk["text"])],
        reference_answer=parsed["reference_answer"],
        notes=f"auto-gen from concordance row {row.get('row_index')}")


def _build_negative(difficulty, rng):
    if difficulty == "hard":
        area = rng.choice(_NEAR_DOMAIN_AREAS)
        prompt = render_prompt("qgen_v3_negative_near", area=area)
        note = f"auto-gen near-domain negative ({area.split(' —')[0]})"
    else:
        domain = rng.choice(_NEGATIVE_DOMAINS)
        prompt = render_prompt(
            "qgen_v2_negative", difficulty=difficulty,
            difficulty_hint=_DIFFICULTY_HINT["negative"][difficulty], domain=domain)
        note = f"auto-gen negative ({domain})"
    parsed = _invoke_json(llm=_LLM, prompt=prompt, tag="negative")
    if not parsed or not _verify_negative(parsed["question"]):
        return None
    return EvalExample(
        question=parsed["question"], category="negative", difficulty=difficulty,
        gold_sections=[], gold_snippets=[], reference_answer=parsed["reference_answer"],
        notes=note)


# ── Pool construction (must-cover first) ───────────────────────────────

def _ordered_pool(chunks: list[dict], act: str) -> list[dict]:
    """Pool consumed via .pop() (from the end); place must-cover chunks LAST so
    they are drawn first, guaranteeing marquee-section coverage."""
    cover_secs = set(_MUST_COVER.get(act, []))
    cover = [c for c in chunks if c["section"] in cover_secs]
    rest = [c for c in chunks if c["section"] not in cover_secs]
    return rest + cover


# ── Resume (mirrors v2) ────────────────────────────────────────────────

def _load_resume_state(out_path: Path):
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


# ── Driver ─────────────────────────────────────────────────────────────

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


def generate(out_path: Path, limit: int | None = None, seed: int = 42,
             save_every: int = 10, resume: bool = True, verify: bool = True):
    global _LLM, _VERIFY
    _LLM = get_llm()
    _VERIFY = verify
    rng = random.Random(seed)

    ipc = _load_chunks("ipc_chunks.json")
    bns = _load_chunks("bns_chunks.json")
    conc = _load_concordance()
    rng.shuffle(ipc); rng.shuffle(bns); rng.shuffle(conc)

    ipc_by_section = {c["section"]: c for c in ipc}
    bns_by_section = {c["section"]: c for c in bns}

    plan = dict(PLAN)
    if limit is not None:
        plan = _scaled_plan(limit)

    examples: list[EvalExample] = []
    used_chunk_ids: set[str] = set()
    used_rows: set[int] = set()
    if resume and out_path.exists():
        examples, done, used_chunk_ids, used_rows = _load_resume_state(out_path)
        for key in list(plan):
            plan[key] = max(0, plan[key] - done.get(key, 0))
        log.info(f"Resuming from {len(examples)} existing; {sum(plan.values())} to go.")

    pool = {
        "ipc_substantive": _ordered_pool(
            [c for c in ipc if c["chunk_id"] not in used_chunk_ids], "IPC"),
        "bns_substantive": _ordered_pool(
            [c for c in bns if c["chunk_id"] not in used_chunk_ids], "BNS"),
        # Only rows whose BOTH sections have a usable chunk — otherwise the build
        # returns None before verify, silently starving the cross_reference bucket.
        "cross_reference": [
            r for r in conc
            if r.get("row_index") not in used_rows
            and r["ipc_section"] in ipc_by_section
            and r["bns_section"] in bns_by_section
        ],
    }

    target_new = sum(plan.values())
    total = target_new + len(examples)

    def maybe_save():
        if len(examples) and len(examples) % save_every == 0:
            save_eval_set(examples, out_path)

    with tqdm(total=target_new, desc="qgen_v3") as bar:
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
             f"(target {sum(PLAN.values())}, verify={verify}) → {out_path}")
    return examples


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate section-based eval set (v3).")
    ap.add_argument("--out", type=Path, default=Path("eval/eval_set_v3.json"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Small mixed test batch of N questions.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore existing output and start over (default: resume).")
    ap.add_argument("--no-verify", dest="verify", action="store_false",
                    help="Skip the LLM verification pass (faster, lower quality).")
    args = ap.parse_args()
    generate(args.out, limit=args.limit, seed=args.seed,
             resume=not args.fresh, verify=args.verify)


if __name__ == "__main__":
    main()
