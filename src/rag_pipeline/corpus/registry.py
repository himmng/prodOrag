"""Corpus registry — loads YAML config into typed Pydantic models.

Used by the ingest CLI and (later) by the API to discover which corpus
is active. Lets us swap entire knowledge bases by changing one env var.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from rag_pipeline.config import cfg


class SourceConfig(BaseModel):
    act:              Literal["IPC", "BNS"]
    pdf_path:         Path
    collection:       str
    chunks_output:    Path
    body_start_page:  int = 1   # ← new field, default 1 = no skip

class ContextSourceConfig(BaseModel):
    """A non-statute, interpretive document (committee report, SOR, etc.).

    Parsed with the PROSE parser (not the statute section parser) and stored
    in its own collection, retrieved as labeled commentary — never as statute.
    """
    doc_type:      str            # "committee_report" | "sor" | ...
    pdf_path:      Path
    collection:    str
    chunks_output: Path
    display_name:  str = ""


class ConcordanceConfig(BaseModel):
    pdf_path:    Path
    output_json: Path


class ParserConfig(BaseModel):
    section_regex:   str
    preserve_blocks: list[str] = Field(default_factory=list)
    max_chunk_chars: int       = 1500


class CorpusConfig(BaseModel):
    """Top-level corpus description. One YAML → one CorpusConfig."""
    name:            str
    display_name:    str
    sources:         list[SourceConfig]
    context_sources: list[ContextSourceConfig] = Field(default_factory=list)
    concordance:     ConcordanceConfig | None = None
    parser:          ParserConfig
    eval_set_path:   Path | None = None
    negatives_path:  Path | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "CorpusConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def resolve_paths(self, project_root: Path) -> None:
        """Convert all relative paths to absolute (relative to project root)."""
        for src in self.sources:
            if not src.pdf_path.is_absolute():
                src.pdf_path = project_root / src.pdf_path
            if not src.chunks_output.is_absolute():
                src.chunks_output = project_root / src.chunks_output
        for ctx in self.context_sources:
            if not ctx.pdf_path.is_absolute():
                ctx.pdf_path = project_root / ctx.pdf_path
            if not ctx.chunks_output.is_absolute():
                ctx.chunks_output = project_root / ctx.chunks_output
        if self.concordance:
            if not self.concordance.pdf_path.is_absolute():
                self.concordance.pdf_path = project_root / self.concordance.pdf_path
            if not self.concordance.output_json.is_absolute():
                self.concordance.output_json = project_root / self.concordance.output_json
        for attr in ("eval_set_path", "negatives_path"):
            v = getattr(self, attr)
            if v and not v.is_absolute():
                setattr(self, attr, project_root / v)


def load_corpus(name: str) -> CorpusConfig:
    """Load a corpus config by name (e.g. 'ipc_bns').

    Looks for configs/corpora/{name}.yaml under the project root.
    """
    config_path = cfg.PROJECT_ROOT / "configs" / "corpora" / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Corpus config not found: {config_path}")

    corpus = CorpusConfig.from_yaml(config_path)
    corpus.resolve_paths(cfg.PROJECT_ROOT)
    return corpus