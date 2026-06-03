""" Parsers: turn files fo various formats into RagChunks."""

from rag_pipeline.parsers.base import BaseParser, ParserDispatcher
from rag_pipeline.parsers.cache import load_chunks_cache, save_chunks_cache
from rag_pipeline.parsers.docling import DoclingHybridParser
from rag_pipeline.parsers.structured import StructuredDataParser

def default_dispatcher() -> ParserDispatcher:
    """ Production dispatcher: Docling first, StructuredData as fallback"""
    return ParserDispatcher([
        DoclingHybridParser(),
        StructuredDataParser(),
    ])


__all__ = [
    "BaseParser",
    "ParserDispatcher",
    "DoclingHybridParser",
    "StructuredDataParser",
    "default_dispatcher",
    "load_chunks_cache",
    "save_chunks_cache"
]