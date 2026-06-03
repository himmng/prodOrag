""" Parser contract + dipatcher.

A parser turns files into Ragchunks. The dispatcher routes files to the first parser that supports them,
 walk directiories, dedupes by content_hash."""

from __future__ import annotations # for forward references in type hints (e.g. in dataclasses)
from pathlib import Path # for filesystem paths
from abc import ABC, abstractmethod # for defining the Parser interface
from collections import Counter # for counting supported formats in the dispatcher
from typing import Iterable, List, Optional, Set, Type # for type hints

from rag_pipeline.config import log # project-wide logger
from rag_pipeline.schemas import RagChunk # the standard output of all parsers


class BaseParser(ABC):
    """All parsers implement this interface.
    
    Parsers MUST be deterministic: parse(file) called twice -> same chunks (modulo any random-seed config).
    This is what makes content-hash-based chunk IDs stable across re-parses."""
    
    supported_extensions: Set[str] = set() # subclasses override

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions
    
    @abstractmethod
    def parse(self, path: Path) -> list[RagChunk]:
        """ Convert one file to a list of RagChunks."""

class ParserDispatcher:
    """Routes files to parsers - first-supports-wins.
    
    Parser order matters: if two parsers both claim `.html`, the earlier one wins. Put more specialized parsers first."""

    def __init__(self, parsers: Iterable[BaseParser]):
        self.parsers = list(parsers)
    
    def parser_for(self, path: Path) -> BaseParser | None:
        """Return the first parser that supports this file, or None if no parser supports it."""
        for parser in self.parsers:
            if parser.supports(path):
                return parser
        return None
    
    def parse_file(self, path: Path) -> List[RagChunk]:
        """Parse one file to chunks, or raise if no parser supports it."""
        parser = self.parser_for(path)
        if parser is None:
            log.warning(f"No parser for {path.suffix} - skipping {path.name}")
            return []
        try:
            return parser.parse(path)
        except Exception as e:
            log.error(f"Error parsing {path} with {parser.__class__.__name__}: {e}")
            return []
        
    def parse_directory(self, root: Path) -> List[RagChunk]:
        """Recurse, parse, dedupe by content_hash. Returns the union."""
        from tqdm import tqdm # for progress bars

        root = Path(root)
        if not root.exists():
            log.error(f"Directory  not found {root}")
            return []

        files = [p for p in root.rglob("*") if p.is_file() and self.parser_for(p)]
        log.info(f"Found {len(files)} supported files under {root}")

        all_chunks: list[RagChunk] = []
        seen: set[str] = set() # content_hash dedupe
        per_format: Counter = Counter()

        for path in tqdm(files, desc="Parsing", unit="file"):
            chunks = self.parse_file(path)
            new_chunks = [c for c in chunks if c.content_hash not in seen]
            seen.update(c.content_hash for c in new_chunks)
            for c in new_chunks:
                per_format[c.source_format] += 1
            all_chunks.extend(new_chunks)
        
        log.info(f"Produced {len(all_chunks)} unique chunks "
                 f"({sum(len(self.parse_file(p)) for p in [] or 'dedup applied')})")
        
        for fmt, n in sorted(per_format.items()):
            log.info(f"  {fmt}: {n} chunks")
        return all_chunks
    
