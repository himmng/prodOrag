"""Parse the IPC↔BNS concordance PDF into structured JSON.

The PDF is a two-column table with status color-coding (new/changed/deleted).
We extract via Docling → reconstruct rows → infer status from text markers.

Output schema (one row):
{
  "bns_section":   "1(1)" | null,
  "ipc_section":   "1"    | null,
  "bns_title":     "Short title..." | null,
  "ipc_title":     "Title and extent..." | null,
  "status":        "new" | "changed" | "deleted" | "unchanged"
}
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from rag_pipeline.config import log
from rag_pipeline.parsers.docling import DoclingHybridParser


@dataclass
class ConcordanceRow:
    bns_section: Optional[str] = None
    ipc_section: Optional[str] = None
    bns_title:   Optional[str] = None
    ipc_title:   Optional[str] = None
    status:      str          = "unchanged"


# Status keywords (case-insensitive) → enum
_STATUS_TOKENS = {
    "new sub-section": "new",
    "new section":     "new",
    "new chapter":     "new",
    "deleted":         "deleted",
    "omission":        "deleted",
    "(change)":        "changed",
    "change":          "changed",
}


def _infer_status(bns_text: str, ipc_text: str) -> str:
    haystack = f"{bns_text} {ipc_text}".lower()
    # Order matters — "new" and "deleted" override "changed"
    for token, label in _STATUS_TOKENS.items():
        if token in haystack:
            return label
    return "unchanged"


# Section ID at start of cell: "1(1)", "2", "302A", "29A"
_SECTION_HEAD = re.compile(r"^\s*(\d+[A-Z]?(?:\([^)]+\))?)\b")


def _split_section_and_title(cell: str) -> tuple[Optional[str], Optional[str]]:
    """Pull the section identifier off the front of a cell, leaving the title."""
    cell = cell.strip()
    if not cell:
        return None, None
    m = _SECTION_HEAD.match(cell)
    if not m:
        return None, cell or None
    sec = m.group(1)
    rest = cell[m.end():].strip().lstrip(".").strip()
    # Strip surrounding quotes from titles like "'document'"
    rest = rest.strip("'\"").strip()
    return sec, (rest or None)


# ── Main extractor ────────────────────────────────────────────────────

class ConcordanceParser:
    """Extracts rows from the concordance PDF.

    Docling returns tables as text-rich chunks. We scan for two-column
    patterns where each row has a BNS cell and an IPC cell.
    """

    def __init__(self):
        self._docling = DoclingHybridParser()

    def parse(self, pdf_path: Path) -> list[ConcordanceRow]:
        log.info(f"Parsing concordance: {pdf_path.name}")

        # Strategy: use Docling's table-aware extraction. Each "doc chunk"
        # often corresponds to one table row when the parser detects tables.
        doc_chunks = self._docling.parse(pdf_path)

        rows: list[ConcordanceRow] = []

        for dc in doc_chunks:
            text = (dc.text if hasattr(dc, "text") else str(dc)).strip()
            if not text:
                continue

            # Skip header rows / banners
            low = text.lower()
            if "corresponding section table" in low \
               or "bharatiya nyaya sanhita" in low and "indian penal code" in low:
                continue

            # Two-column rows usually have a tab, multiple spaces, or newline as separator
            cells = self._split_two_columns(text)
            if not cells:
                continue
            left, right = cells

            bns_section, bns_title = _split_section_and_title(left)
            ipc_section, ipc_title = _split_section_and_title(right)

            # Drop rows where neither side has a section ID
            if not (bns_section or ipc_section):
                continue

            row = ConcordanceRow(
                bns_section=bns_section,
                ipc_section=ipc_section,
                bns_title=bns_title,
                ipc_title=ipc_title,
                status=_infer_status(left, right),
            )
            rows.append(row)

        log.info(f"  Extracted {len(rows)} concordance rows")
        log.info(f"  Status distribution: {self._status_counts(rows)}")
        return rows

    @staticmethod
    def _split_two_columns(text: str) -> Optional[tuple[str, str]]:
        """Heuristically split a row into (left, right) cells."""
        # Prefer tab separator
        if "\t" in text:
            parts = [p.strip() for p in text.split("\t") if p.strip()]
            if len(parts) >= 2:
                return parts[0], parts[1]

        # Try 2+ spaces as separator (common in PDF table extraction)
        parts = re.split(r"\s{2,}", text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) == 2:
            return parts[0], parts[1]
        if len(parts) > 2:
            # Re-join into halves — last item is usually IPC side
            return " ".join(parts[:-1]), parts[-1]

        # Fallback: split on newline if there's exactly one
        nl_parts = [p.strip() for p in text.split("\n") if p.strip()]
        if len(nl_parts) == 2:
            return nl_parts[0], nl_parts[1]

        return None

    @staticmethod
    def _status_counts(rows: list[ConcordanceRow]) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            out[r.status] = out.get(r.status, 0) + 1
        return out


# ── Lookup helper (used at retrieval time) ────────────────────────────

class Concordance:
    """In-memory bidirectional lookup over the concordance rows."""

    def __init__(self, rows: list[ConcordanceRow]):
        self._ipc_to_bns: dict[str, ConcordanceRow] = {}
        self._bns_to_ipc: dict[str, ConcordanceRow] = {}
        for r in rows:
            if r.ipc_section:
                self._ipc_to_bns[r.ipc_section] = r
            if r.bns_section:
                self._bns_to_ipc[r.bns_section] = r

    def lookup_ipc(self, ipc_section: str) -> Optional[ConcordanceRow]:
        return self._ipc_to_bns.get(ipc_section)

    def lookup_bns(self, bns_section: str) -> Optional[ConcordanceRow]:
        return self._bns_to_ipc.get(bns_section)

    @classmethod
    def from_json(cls, path: Path) -> "Concordance":
        with open(path) as f:
            data = json.load(f)
        rows = [ConcordanceRow(**r) for r in data]
        return cls(rows)


def save_rows(rows: list[ConcordanceRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in rows], f, indent=2, ensure_ascii=False)
    log.info(f"Wrote {len(rows)} concordance rows → {path}")