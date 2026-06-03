""" Docling-based parser with structure-aware chunking.

Uses docling's Hybridchunker which respects document structure (sections, tables, lists) 
and merges small adjacent chunks up to a token budget.

Preserves heading hierarchy and page numbers for citation provenance.

Supports: pdfs, docx, pptx, html, htm, md"""

from __future__ import annotations

import logging
from pathlib import Path

from rag_pipeline.parsers.base import BaseParser
from rag_pipeline.schemas import RagChunk

# suppress docling's cuda kernel JIT-compile noise.
# (transforers/kernels/deformable_detr CUDA kernel fails on pyTorch 2.x)
# silently falls back to CPU at ~13-17 sec/PDF -noisy but works

logging.getLogger("transformers").setLevel(logging.CRITICAL)

class DoclingHybridParser(BaseParser):
    """Structure-aware parser for PDF, Docx, PPTx, html, markdown """

    supported_extensions = {".pdf", ".docx", ".pptx", ".html", ".htm", ".md"}

    def __init__(
            self,
            tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
            max_tokens: int = 512,
            merge_peers: bool = True,
    ):
        # heavy imports done in __init__ so unused parsers don't pull in touch
        from docling.document_converter import DocumentConverter
        from docling.chunking import HybridChunker
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
        from transformers import AutoTokenizer

        self._converter = DocumentConverter()
        self._chunker = HybridChunker(
            tokenizer=HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(tokenizer_model),
                max_tokens=max_tokens,
            ),
            merge_peers=merge_peers,
        )
    
    def parse(self, path: Path) -> list[RagChunk]:
        doc = self._converter.convert(path).document
        suffix = path.suffix.lower().lstrip(".")
        source_format = "html" if suffix == "htm" else suffix # normalize .htm -> .html

        out: list[RagChunk] = []
        for dc in self._chunker.chunk(doc):
            text = self._chunker.contextualize(chunk=dc)

            # page number from first prov of first doc item

            page = None
            if dc.meta.doc_items:
                first = dc.meta.doc_items[0]
                if first.prov:
                    page = first.prov[0].page_no

            # deepest heading in the hierarchy
            section_title = dc.meta.headings[-1] if dc.meta.headings else None

            out.append(RagChunk(
                text=text,
                source_path=str(path.resolve()),
                source_format=source_format,
                page_number=page,
                section_title=section_title,
                element_type="docling-chunk",
            ))

        return out
