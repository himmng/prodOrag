"""Generate a stratified eval set from a chunk corpus.

Three difficulty bands, each with its own YAML prompt:
  - easy:   single fact from one passage
  - medium: 2-3 fact synthesis from one passage
  - hard:   synthesis across TWO passages from different source files

Hard-difficulty cross-passage sampling is what makes the eval set useful
for measuring real RAG quality on compound questions.

Bad LLM outputs are logged + skipped (final count may be slightly below target).
Saves incrementally to disk for crash safety.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage
from tqdm import tqdm

from rag_pipeline.config import log
from rag_pipeline.eval.schema import EvalExample, save_eval_set
from rag_pipeline.prompts import render_prompt

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from rag_pipeline.schemas import RagChunk


def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers even though the prompt forbids them."""
    return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()


def _try_single_passage(
    chunk: "RagChunk",
    llm: "BaseChatModel",
    prompt_name: str,
    difficulty: str,
) -> EvalExample | None:
    prompt = render_prompt(prompt_name, passage=chunk.text)
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        data = json.loads(_strip_json_fences(resp.content))
        return EvalExample(
            question=data["question"].strip(),
            gold_source_paths=[chunk.source_path],
            gold_snippets=[data["gold_snippet"].strip()] if data.get("gold_snippet") else [],
            reference_answer=(data.get("reference_answer") or "").strip() or None,
            difficulty=difficulty,
        )
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        log.warning(f"[qgen-{difficulty}] parse failed: {e}")
        return None


def _try_two_passage(
    chunk_a: "RagChunk",
    chunk_b: "RagChunk",
    llm: "BaseChatModel",
) -> EvalExample | None:
    prompt = render_prompt("qgen_hard", passage_a=chunk_a.text, passage_b=chunk_b.text)
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        data = json.loads(_strip_json_fences(resp.content))
        snippets = data.get("gold_snippets", [])
        if not isinstance(snippets, list):
            snippets = [str(snippets)]
        return EvalExample(
            question=data["question"].strip(),
            gold_source_paths=[chunk_a.source_path, chunk_b.source_path],
            gold_snippets=[s.strip() for s in snippets if s and isinstance(s, str)],
            reference_answer=(data.get("reference_answer") or "").strip() or None,
            difficulty="hard",
        )
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        log.warning(f"[qgen-hard] parse failed: {e}")
        return None


def generate_balanced_eval_set(
    chunks: list["RagChunk"],
    llm: "BaseChatModel",
    n_easy:   int = 40,
    n_medium: int = 60,
    n_hard:   int = 100,
    save_path: Path | None = None,
    save_every: int = 20,
    seed: int = 42,
) -> list[EvalExample]:
    """Generate a stratified eval set: easy + medium + hard."""
    rng = random.Random(seed)
    examples: list[EvalExample] = []

    # Group chunks by source for cross-file hard sampling
    by_source: dict[str, list["RagChunk"]] = {}
    for c in chunks:
        by_source.setdefault(c.source_path, []).append(c)
    if len(by_source) < 2 and n_hard > 0:
        log.warning("Hard questions require ≥2 source files; reducing n_hard → 0")
        n_hard = 0

    def maybe_save():
        if save_path and len(examples) and len(examples) % save_every == 0:
            save_eval_set(examples, save_path)

    # EASY
    for _ in tqdm(range(n_easy), desc="qgen-easy"):
        ex = _try_single_passage(rng.choice(chunks), llm, "qgen_easy", "easy")
        if ex:
            examples.append(ex)
            maybe_save()

    # MEDIUM
    for _ in tqdm(range(n_medium), desc="qgen-medium"):
        ex = _try_single_passage(rng.choice(chunks), llm, "qgen_medium", "medium")
        if ex:
            examples.append(ex)
            maybe_save()

    # HARD — two chunks from different source files
    sources = list(by_source.keys())
    for _ in tqdm(range(n_hard), desc="qgen-hard"):
        src_a, src_b = rng.sample(sources, 2)
        chunk_a = rng.choice(by_source[src_a])
        chunk_b = rng.choice(by_source[src_b])
        ex = _try_two_passage(chunk_a, chunk_b, llm)
        if ex:
            examples.append(ex)
            maybe_save()

    if save_path:
        save_eval_set(examples, save_path)
    log.info(
        f"Generated {len(examples)} eval examples (target: {n_easy + n_medium + n_hard})"
    )
    return examples