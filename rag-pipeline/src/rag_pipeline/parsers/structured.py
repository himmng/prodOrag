"""
Parser for tabular + structured + free-text formats which Docling doesn't cover.
Routing by extension:
  .csv / .xlsx / .xls  → N rows per chunk (default 10)
  .json (array)        → one chunk per record
  .json (object)       → one chunk per file
  .jsonl / .ndjson     → one chunk per line
  .txt                 → recursive character split, cfg.CHUNK_SIZE / cfg.CHUNK_OVERLAP

"""

from __future__ import annotations
import json
from pathlib import Path

from rag_pipeline.config import cfg, log 
from rag_pipeline.parsers.base import BaseParser
from rag_pipeline.schemas import RagChunk


class StructuredDataParser(BaseParser):
    """Tabular + structured Json + free text."""
    supported_extensions = {
        ".csv", ".xlsx", ".xls",
        ".json", ".jsonl", ".ndjson",
        ".txt",
    }

    def __init__(self, rows_per_chunk: int = 10):
        self.rows_per_chunk = rows_per_chunk

    def parse(self, path: Path) -> list[RagChunk]:
        ext = path.suffix.lower()
        if ext in {".csv", ".xlsx", ".xls"}:
            return self._parse_tabular(path)
        if ext in {".jsonl", ".ndjson"}:
            return self._parse_jsonl(path)
        if ext == ".json":
            return self._parse_json(path)
        if ext == ".txt":
            return self._parse_text(path)
        return []

    def _parse_tabular(self, path: Path) -> list[RagChunk]:
        import pandas as pd
        ext = path.suffix.lower()
        try:
            if ext == ".csv":
                sheets = {None: pd.read_csv(path)}
            else:
                sheets = pd.read_excel(path, sheet_name=None)
        except Exception as e:
            log.error(f"Failed to read {path.name}: {e}")
            return []

        out: list[RagChunk] = []
        for sheet_name, df in sheets.items():
            for start in range(0, len(df), self.rows_per_chunk):
                window = df.iloc[start:start + self.rows_per_chunk]
                out.append(RagChunk(
                    text=window.to_csv(index=False),
                    source_path=str(path.resolve()),
                    source_format=ext.lstrip("."),
                    sheet_name=sheet_name,
                    row_range=f"{start}-{start + len(window) - 1}",
                    element_type="rows",
                ))
        return out

    def _parse_jsonl(self, path: Path) -> list[RagChunk]:
        out: list[RagChunk] = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning(f"{path.name}:{i + 1} bad JSON — {e}")
                    continue
                out.append(RagChunk(
                    text=json.dumps(obj, ensure_ascii=False, indent=2),
                    source_path=str(path.resolve()),
                    source_format=path.suffix.lower().lstrip("."),
                    row_range=f"{i}-{i}",
                    element_type="record",
                ))
        return out

    def _parse_json(self, path: Path) -> list[RagChunk]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            log.error(f"{path.name} bad JSON — {e}")
            return []

        if isinstance(data, list):
            return [
                RagChunk(
                    text=json.dumps(obj, ensure_ascii=False, indent=2),
                    source_path=str(path.resolve()),
                    source_format="json",
                    row_range=f"{i}-{i}",
                    element_type="record",
                )
                for i, obj in enumerate(data)
            ]
        return [RagChunk(
            text=json.dumps(data, ensure_ascii=False, indent=2),
            source_path=str(path.resolve()),
            source_format="json",
            element_type="document",
        )]

    def _parse_text(self, path: Path) -> list[RagChunk]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text = path.read_text(encoding="utf-8")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.CHUNK_SIZE,
            chunk_overlap=cfg.CHUNK_OVERLAP,
        )
        return [
            RagChunk(
                text=t,
                source_path=str(path.resolve()),
                source_format="txt",
                element_type="text",
            )
            for t in splitter.split_text(text)
        ]