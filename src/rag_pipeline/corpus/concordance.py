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
    row_index:   Optional[int] = None 
    page_number: Optional[int] = None 


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
_SECTION_HEAD = re.compile(
    r"^\s*(\d+[A-Z]?(?:\(\d+[A-Za-z]?\))?)\s*[.\s]?"
)


def _split_section_and_title(cell: str) -> tuple[Optional[str], Optional[str]]:
    """Pull the section identifier off the front of a cell, leaving the title."""
    cell = cell.strip()
    if not cell:
        return None, None

    lines = [ln.strip() for ln in cell.split("\n") if ln.strip()]

    for i, line in enumerate(lines):
        m = _SECTION_HEAD.match(line)
        if not m:
            continue

        sec = m.group(1)
        # Title candidate: text in this line after the section ID
        remainder = line[m.end():].strip().lstrip(".").strip()

        # If this line has nothing after the section ID, use the next line(s)
        if not remainder and i + 1 < len(lines):
            remainder = " ".join(lines[i + 1:]).strip()

        # Clean trailing/leading punctuation and quotes
        title = remainder.strip().rstrip(".").strip().strip("'\"").strip()
        return sec, (title or None)

    # No section header at all — whole cell is the title
    title = " ".join(lines).strip("'\"").strip()
    return None, title or None


# ── Main extractor ────────────────────────────────────────────────────

class ConcordanceParser:
    """Extracts table rows from the concordance PDF using pdfplumber.

    pdfplumber preserves cell boundaries, unlike Docling which flattens
    tables to prose. Each row gives us (bns_cell, ipc_cell) directly.
    """

    def parse(self, pdf_path: Path) -> list[ConcordanceRow]:
        log.info(f"Parsing concordance with pdfplumber: {pdf_path.name}")

        import pdfplumber

        rows: list[ConcordanceRow] = []
        seen: set[tuple[Optional[str], Optional[str]]] = set()

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for table in tables:
                    for raw_row in table:
                        # Skip None-only or header rows
                        cells = [(c or "").strip() for c in raw_row]
                        if all(not c for c in cells):
                            continue

                        # Header detection
                        joined = " ".join(cells).lower()
                        if "bharatiya nyaya sanhita" in joined and \
                           "indian penal code" in joined and \
                           "chapter" not in joined:
                            continue

                        # Expecting 2 cells per row: [BNS, IPC]
                        if len(cells) < 2:
                            continue
                        bns_cell, ipc_cell = cells[0], cells[1]

                        row = self._build_row(bns_cell, ipc_cell)
                        if row is None:
                            continue

                        key = (row.bns_section, row.ipc_section)
                        if key in seen:
                            continue
                        seen.add(key)
                        row.row_index = len(rows) + 1
                        row.page_number = page_num 
                        rows.append(row)

        log.info(f"  Extracted {len(rows)} unique rows")
        log.info(f"  Status distribution: {self._status_counts(rows)}")
        return rows

    @staticmethod
    def _build_row(bns_cell: str, ipc_cell: str) -> Optional[ConcordanceRow]:
        bns_cell = bns_cell.strip().rstrip(",").strip()
        ipc_cell = ipc_cell.strip().rstrip(".").strip()
        if not bns_cell and not ipc_cell:
            return None

        # Skip the act-name banner row
        # ("Bharatiya Nyaya Sanhita, 2023 (BNS)" / "Indian Penal Code, 1860 (IPC)")
        joined = (bns_cell + " " + ipc_cell).lower()
        if "(bns)" in joined or "(ipc)" in joined or \
           "sanhita" in joined or "penal code" in joined:
            return None
        # Skip chapter divider rows ("CHAPTER I – PRELIMINARY")
        if bns_cell.lower().startswith("chapter ") or \
           ipc_cell.lower().startswith("chapter "):
            return None

        bns_section, bns_title = _split_section_and_title(bns_cell)
        ipc_section, ipc_title = _split_section_and_title(ipc_cell)

        return ConcordanceRow(
            bns_section=bns_section,
            ipc_section=ipc_section,
            bns_title=bns_title,
            ipc_title=ipc_title,
            status=_infer_status(bns_cell, ipc_cell),
        )

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