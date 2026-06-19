"""Shared pytest fixtures.

Add reusable test data, mock objects, etc. here as the suite grows.
"""

from __future__ import annotations

import pytest

from rag_pipeline.eval.schema import EvalExample

@pytest.fixture
def positive_example() -> EvalExample:
    """A typical positive eval example."""
    return EvalExample(
        question="What is the punishment for theft?",
        gold_source_paths=["/corpus/ipc/section_379.pdf"],
        gold_snippets=["punishment for theft", "imprisonment"],
        difficulty="easy",
    )


@pytest.fixture
def negative_example() -> EvalExample:
    """A negative (OOD) example — empty gold lists imply is_negative()."""
    return EvalExample(
        question="What is the capital of France?",
        gold_source_paths=[],
        gold_snippets=[],
        difficulty="negative",
    )