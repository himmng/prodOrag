"""Generation: turn retrieved chunks into grounded, cited answers."""

from rag_pipeline.generation.context import build_context, pretty_print
from rag_pipeline.generation.pipeline import REFUSAL_TEXT, answer, answer_stream

__all__ = ["answer", "answer_stream", "build_context", "pretty_print", "REFUSAL_TEXT"]