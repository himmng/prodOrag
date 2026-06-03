"""Generation: turn retrieved chunks into grounded, cited answers."""

from rag_pipeline.generation.context import build_context, pretty_print
from rag_pipeline.generation.pipeline import REFUSAL_TEXT, answer

__all__ = ["answer", "build_context", "pretty_print", "REFUSAL_TEXT"]