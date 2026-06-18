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
        expected_source_path="/corpus/ipc/section_379.pdf",
        expected_snippets=["punishment for theft", "imprisonment"],
        difficulty="easy",
        is_negative_flag=False,   # adjust if your EvalExample field name differs
    )


@pytest.fixture
def negative_example() -> EvalExample:
    """A negative (OOD) eval example."""
    return EvalExample(
        question="What is the capital of France?",
        expected_source_path=None,
        expected_snippets=[],
        difficulty="negative",
        is_negative_flag=True,
    )