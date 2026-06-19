"""Prompt loading + rendering.

Prompts live in YAML files under prompts/templates/. Each file has:
  - template: the prompt body (Python str.format placeholders)
  - description, variables, version: metadata

Production override: set PROMPTS_DIR env var (e.g., /mnt/configmap/prompts
in K8s, or a mounted S3/Azure Blob path) to load from a different location
without changing code.

Usage:
    from rag_pipeline.prompts import render_prompt
    prompt = render_prompt("query_expansion", n=3, query="...")
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

from rag_pipeline.config import log


_DEFAULT_DIR = Path(__file__).parent / "templates"


class PromptManager:
    """Loads and renders prompts from YAML files."""

    def __init__(self, prompts_dir: Path | None = None):
        self.dir = Path(prompts_dir) if prompts_dir else self._resolve_dir()
        if not self.dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.dir}")
        log.info(f"PromptManager: loading from {self.dir}")

    @staticmethod
    def _resolve_dir() -> Path:
        """Use PROMPTS_DIR env var if set, otherwise the package default."""
        env_path = os.environ.get("PROMPTS_DIR")
        return Path(env_path) if env_path else _DEFAULT_DIR

    @lru_cache(maxsize=64)
    def _load(self, name: str) -> dict:
        path = self.dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Prompt {name!r} not found at {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or "template" not in data:
            raise ValueError(
                f"Prompt {name!r} must be a YAML dict with a 'template' field"
            )
        return data

    def get(self, name: str) -> str:
        """Return the raw template string."""
        return self._load(name)["template"]

    def render(self, name: str, **kwargs) -> str:
        """Render a prompt with format-style variables."""
        return self.get(name).format(**kwargs)

    def metadata(self, name: str) -> dict:
        """Return prompt metadata (description, version, variables, ...) without the template."""
        return {k: v for k, v in self._load(name).items() if k != "template"}


# ── Module-level singleton ────────────────────────────────────────────
_manager: PromptManager | None = None


def _get_manager() -> PromptManager:
    global _manager
    if _manager is None:
        _manager = PromptManager()
    return _manager


def get_prompt(name: str) -> str:
    """Return the raw template string for a prompt."""
    return _get_manager().get(name)


def render_prompt(name: str, **kwargs) -> str:
    """Render a prompt with format-style variables."""
    return _get_manager().render(name, **kwargs)