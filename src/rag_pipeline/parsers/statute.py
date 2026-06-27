"""Structural parser for statute PDFs (IPC, BNS).

Pipeline:
  1. Docling extracts text + page-number metadata from the PDF
  2. We concatenate per-page text, normalize whitespace
  3. Split by section regex (e.g., "302.", "498A.", "2(1)")
  4. Within each section, identify sub-blocks (illustration, exception, etc.)
  5. Emit one StatuteChunk per section (and per sub-block if section is large)

Section ID is the canonical key: "302", "302A", "2(1)". No "Section" prefix.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Iterator, Optional

from rag_pipeline.config import log
from rag_pipeline.corpus.registry import ParserConfig
from rag_pipeline.parsers.docling import DoclingHybridParser
from rag_pipeline.schemas import StatuteChunk


# ── Block-type detectors ──────────────────────────────────────────────
# Order matters: more specific first. Each returns chunk_type or None.

_BLOCK_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("illustration", re.compile(r"^\s*Illustration[s]?\b", re.IGNORECASE)),
    ("explanation",  re.compile(r"^\s*Explanation\s*\d*\.?", re.IGNORECASE)),
    ("exception",    re.compile(r"^\s*Exception[s]?\b",     re.IGNORECASE)),
    ("proviso",      re.compile(r"^\s*Provided\s+that\b",   re.IGNORECASE)),
]


def _classify_block(text: str) -> str:
    """Identify the chunk_type of a text block."""
    head = text.lstrip()[:80]   # only check the start
    for label, pat in _BLOCK_PATTERNS:
        if pat.match(head):
            return label
    # Heuristic: "shall be punished" → punishment
    if re.search(r"\bshall\s+be\s+punished\b", text, re.IGNORECASE):
        return "punishment"
    return "other"


# ── Section splitting ─────────────────────────────────────────────────

def _split_by_section(
    full_text: str,
    section_regex: str,
) -> list[tuple[str, Optional[str], int, str]]:
    """Split the full document text into (section, subsection, start_pos, body).

    Section header pattern matches e.g., '302.', '302A.', '2(1).'.
    Returns list ordered by document position.
    """
    pat = re.compile(section_regex)
    matches = list(pat.finditer(full_text))
    if not matches:
        log.warning("No section headers matched; check section_regex")
        return []

    sections: list[tuple[str, Optional[str], int, str]] = []
    for i, m in enumerate(matches):
        section_id = m.group(1)
        subsection = m.group(2) if m.lastindex and m.lastindex >= 2 else None
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()
        sections.append((section_id, subsection, m.start(), body))
    return sections


# ── Title + chapter heuristics ────────────────────────────────────────

# Pull the first line of section body as title (legal acts usually format this way)
_TITLE_LINE = re.compile(r"^([^\n.]{3,160})")
_CHAPTER_RE = re.compile(
    r"^\s*CHAPTER\s+([IVXLCDM]+|\d+)\s*[—:.-]?\s*(.{3,160})?",
    re.IGNORECASE | re.MULTILINE,
)
# Marker substring near the start of the actual law body.
# Both acts have "ARRANGEMENT OF SECTIONS" in the TOC; the LAW BODY starts
# with the long form act name (or its first numbered section after).
_TOC_END_MARKERS = {
    "IPC": [
        "An Act to provide a general Penal Code for India",  # IPC preamble
        "Preamble.",                                          # alternative
    ],
    "BNS": [
        "An Act to consolidate and amend the provisions",     # BNS preamble
        "Preamble.",
    ],
}


def _extract_title(body: str) -> str:
    m = _TITLE_LINE.match(body)
    return m.group(1).strip().rstrip(".") if m else ""


def _build_chapter_index(full_text: str) -> list[tuple[int, str, str]]:
    """Return [(char_offset, chapter_number, chapter_title), ...] sorted by offset."""
    out = []
    for m in _CHAPTER_RE.finditer(full_text):
        num   = m.group(1).strip()
        title = (m.group(2) or "").strip().rstrip(".")
        out.append((m.start(), num, title))
    return out


def _chapter_for_offset(
    offset: int,
    chapters: list[tuple[int, str, str]],
) -> tuple[Optional[str], Optional[str]]:
    """Find which chapter a section belongs to (last chapter header before it)."""
    last = (None, None)
    for off, num, title in chapters:
        if off <= offset:
            last = (num, title)
        else:
            break
    return last


# ── Sub-block splitting within a section ──────────────────────────────

def _split_subblocks(body: str) -> list[tuple[str, str]]:
    """Split a section body into (chunk_type, text) sub-blocks.

    Splits at the START of recognized block markers (Illustration, Explanation, etc.).
    The pre-marker text is the "main" section body (offence + punishment).
    """
    # Find all block-marker positions
    cuts: list[tuple[int, str]] = []
    for label, pat in _BLOCK_PATTERNS:
        for m in pat.finditer(body):
            # Only count if at start of a line
            line_start = body.rfind("\n", 0, m.start()) + 1
            if m.start() - line_start <= 3:  # allow small indent
                cuts.append((line_start, label))

    if not cuts:
        return [(_classify_block(body), body.strip())]

    cuts.sort()
    blocks: list[tuple[str, str]] = []

    # Main body = everything before the first cut
    first_cut_pos = cuts[0][0]
    main = body[:first_cut_pos].strip()
    if main:
        blocks.append((_classify_block(main), main))

    # Each cut → its text until next cut (or end)
    for i, (pos, label) in enumerate(cuts):
        end = cuts[i + 1][0] if i + 1 < len(cuts) else len(body)
        text = body[pos:end].strip()
        if text:
            blocks.append((label, text))

    return blocks


# ── Page-number tracking ──────────────────────────────────────────────

def _page_for_offset(
    offset: int,
    page_offsets: list[tuple[int, int]],
) -> int:
    """Given char offset in concatenated text, return page number."""
    for char_off, page_num in page_offsets:
        if offset >= char_off:
            current = page_num
        else:
            return current
    return current if page_offsets else 1

# skip ToC
def _skip_toc(text: str, act: str) -> str:
    """Find where the real body starts; discard the TOC."""
    markers = _TOC_END_MARKERS.get(act, [])
    best = -1
    for marker in markers:
        idx = text.find(marker)
        if idx != -1 and (best == -1 or idx < best):
            best = idx

    if best == -1:
        log.warning(f"  No TOC end marker found for {act}; keeping full text")
        return text

    log.info(f"  TOC ends at char {best:,}; using body only")
    return text[best:]

# ── Public entry point ───────────────────────────────────────────────

class StatuteParser:
    """Parses a statute PDF into StatuteChunks.

    Configured by a ParserConfig from the corpus registry. Knows nothing
    about IPC vs BNS — the caller passes `act` for tagging.
    """

    def __init__(self, parser_cfg: ParserConfig):
        self.cfg = parser_cfg
        self._docling = DoclingHybridParser()
    

    def parse(
        self,
        pdf_path: Path,
        act: str,
        body_start_page: int = 1,
    ) -> list[StatuteChunk]:
        """Parse one PDF → list of StatuteChunks tagged with `act`.

        body_start_page: skip pages before this (1-indexed).
        TOC pages should be excluded by setting this past the TOC.
        """
        log.info(f"Parsing {pdf_path.name} as {act} statute (body from page {body_start_page})")

        # Step 1: extract via Docling
        doc_chunks = self._docling.parse(pdf_path)

        # Step 2: KEEP ONLY chunks at/after body_start_page
        parts: list[str] = []
        page_offsets: list[tuple[int, int]] = []
        running = 0
        kept_pages = 0
        for dc in doc_chunks:
            page = getattr(dc, "page_number", None) or \
                   (dc.metadata or {}).get("page_number", 1)
            if page < body_start_page:
                continue
            text = dc.text if hasattr(dc, "text") else str(dc)
            page_offsets.append((running, page))
            parts.append(text)
            running += len(text) + 1
            kept_pages += 1

        full_text = "\n".join(parts)
        log.info(f"  Kept {kept_pages} chunks; body text = {len(full_text):,} chars")

        # Step 3: build chapter index from body text only
        chapters = _build_chapter_index(full_text)
        log.info(f"  Found {len(chapters)} chapter headers")

        # Step 4: split by section regex
        sections = _split_by_section(full_text, self.cfg.section_regex)
        log.info(f"  Found {len(sections)} sections in body")

        # Step 5: build chunks (no more TOC filtering — we already skipped it)
        out: list[StatuteChunk] = []
        for section_id, subsection, doc_offset, body in sections:
            body_clean = body.strip()
            if not body_clean:
                continue

            chapter_num, chapter_title = _chapter_for_offset(doc_offset, chapters)
            page = _page_for_offset(doc_offset, page_offsets)
            section_title = _extract_title(body_clean)
            parent = section_id if subsection else None

            subblocks = _split_subblocks(body_clean)

            if len(subblocks) == 1 and len(body_clean) <= self.cfg.max_chunk_chars:
                ctype, text = subblocks[0]
                disambiguator = f"{section_id}|{subsection or ''}|{doc_offset}"
                out.append(StatuteChunk(
                    chunk_id=StatuteChunk.make_id(act, disambiguator, text),
                    source_path=str(pdf_path),
                    page_number=page,
                    act=act,
                    section=section_id,
                    subsection=subsection,
                    parent_section=parent,
                    section_title=section_title,
                    chapter_number=chapter_num,
                    chapter_title=chapter_title,
                    chunk_type=ctype,
                    text=text,
                ))
            else:
                for idx, (ctype, text) in enumerate(subblocks):
                    if not text.strip():
                        continue
                    disambiguator = f"{section_id}|{subsection or ''}|{doc_offset}|{idx}|{ctype}"
                    out.append(StatuteChunk(
                        chunk_id=StatuteChunk.make_id(act, disambiguator, text),
                        source_path=str(pdf_path),
                        page_number=page,
                        act=act,
                        section=section_id,
                        subsection=subsection,
                        parent_section=parent,
                        section_title=section_title,
                        chapter_number=chapter_num,
                        chapter_title=chapter_title,
                        chunk_type=ctype,
                        text=text,
                    ))

        log.info(f"  Produced {len(out)} chunks (types: {dict(_count_types(out))})")
        return out
    
def _count_types(chunks: list[StatuteChunk]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.chunk_type] = counts.get(c.chunk_type, 0) + 1
    return counts