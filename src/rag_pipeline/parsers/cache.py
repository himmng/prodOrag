""" On-disk cache for parsed chunks.

Stores RagChunks as JSON via Pydantic's model_dump. 
Loading reconstructs full RagChunk objects (with re-validated identity fields)."""

from __future__ import annotations

import json
from pathlib import Path

from rag_pipeline.config import log
from rag_pipeline.schemas import RagChunk


def save_chunks_cache(chunks: list[RagChunk], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.model_dump(mode="json") for c in chunks]
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"Saved {len(chunks)} chunks -> {path.name}")


def load_chunks_cache(path: Path) -> list[RagChunk]:
    path = Path(path)
    if not path.exists():
        log.warning(f"Cache miss: {path.name}")
        return []
    
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = [RagChunk(**d) for d in data]
    log.info(f"Loaded {len(chunks)} chunks <- {path.name}")

    return chunks