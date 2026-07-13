"""Evaluation example schema + YAML-backed loaders.

EvalExample is the unit of evaluation. POSITIVE examples have non-empty
gold_source_paths — we expect the retriever to find those files and the
generator to answer faithfully. NEGATIVE examples have empty
gold_source_paths — we expect the system to refuse politely.

Negative examples and refusal markers live in YAML under eval/data/, not
in code, so domain reviewers can curate them without touching Python.
Override via the NEGATIVES_DIR env var (mounted ConfigMap, S3, etc.).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field

from rag_pipeline.config import log

Difficulty = Literal["easy", "medium", "hard"]


class EvalExample(BaseModel):
    """A single eval row. Positive if gold_source_paths is non-empty."""

    question:          str
    gold_source_paths: list[str] = Field(default_factory=list)
    gold_snippets:     list[str] = Field(default_factory=list)
    gold_sections:     list[dict] = Field(default_factory=list)
    reference_answer:  Optional[str] = None
    difficulty:        Difficulty   = "medium"
    notes:             Optional[str] = None
    category:          Optional[str] = None

    # Behavioural ground truth (adversarial sets). When a question is scored on
    # HOW the system responds rather than which sections it retrieves,
    # `expected_behavior` states the required behaviour in plain language and
    # `failure_modes` tags the production failure(s) the question is built to catch
    # (e.g. "stale_law", "silent_act_selection", "failed_to_refuse", "overreach").
    expected_behavior: Optional[str]  = None
    failure_modes:     list[str]      = Field(default_factory=list)

    def is_negative(self) -> bool:
        return not self.gold_source_paths and not self.gold_sections


# ── YAML-backed loaders ────────────────────────────────────────────────

_DEFAULT_DATA_DIR = Path(__file__).parent / "data"


def _resolve_data_dir() -> Path:
    """Use NEGATIVES_DIR env var if set, otherwise the package default."""
    env = os.environ.get("NEGATIVES_DIR")
    return Path(env) if env else _DEFAULT_DATA_DIR


@lru_cache(maxsize=8)
def load_negatives(name: str = "ipc_negatives") -> list[EvalExample]:
    """Load a curated negative-example set from YAML."""
    path = _resolve_data_dir() / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Negative set {name!r} not found at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    examples = [EvalExample(**e) for e in data.get("examples", [])]
    log.info(f"Loaded {len(examples)} negative examples ← {path.name}")
    return examples


@lru_cache(maxsize=8)
def load_refusal_markers(name: str = "refusal_markers") -> list[str]:
    """Load refusal-detection substrings from YAML."""
    path = _resolve_data_dir() / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Refusal-markers set {name!r} not found at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    markers = [m.strip().lower() for m in data.get("markers", []) if m]
    log.info(f"Loaded {len(markers)} refusal markers ← {path.name}")
    return markers


# ── Eval-set I/O ────────────────────────────────────────────────────────

def save_eval_set(examples: list[EvalExample], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [e.model_dump(mode="json") for e in examples]
    # Atomic write: dump to a temp file in the same dir, then os.replace so a
    # crash/interruption mid-write can never leave a truncated checkpoint.
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    log.info(f"Saved {len(examples)} eval examples → {path.name}")


def load_eval_set(path: Path) -> list[EvalExample]:
    path = Path(path)
    if not path.exists():
        log.warning(f"Eval set not found: {path.name}")
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    examples = [EvalExample(**d) for d in data]
    log.info(f"Loaded {len(examples)} eval examples ← {path.name}")
    return examples