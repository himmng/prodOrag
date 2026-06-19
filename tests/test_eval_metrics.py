"""Unit tests for retrieval metric functions.

These run against synthetic data — no LLM, no vectorstore, no network.
"""

from rag_pipeline.eval.retrieval import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    snippet_hit_at_k,
)


def test_hit_at_k_when_expected_in_results(positive_example):
    retrieved = [
        "/corpus/ipc/section_378.pdf",
        "/corpus/ipc/section_379.pdf",   # match
        "/corpus/ipc/section_380.pdf",
    ]
    assert hit_at_k(positive_example, retrieved) == 1


def test_hit_at_k_when_expected_missing(positive_example):
    retrieved = [
        "/corpus/ipc/section_300.pdf",
        "/corpus/ipc/section_301.pdf",
    ]
    assert hit_at_k(positive_example, retrieved) == 0


def test_reciprocal_rank_at_position_2(positive_example):
    retrieved = [
        "/corpus/ipc/section_378.pdf",
        "/corpus/ipc/section_379.pdf",   # rank 2 (1-indexed)
        "/corpus/ipc/section_380.pdf",
    ]
    assert reciprocal_rank(positive_example, retrieved) == 0.5


def test_reciprocal_rank_when_missing(positive_example):
    retrieved = ["/corpus/ipc/section_100.pdf"]
    assert reciprocal_rank(positive_example, retrieved) == 0.0


def test_recall_at_k_full_recall(positive_example):
    retrieved = ["/corpus/ipc/section_379.pdf"]
    assert recall_at_k(positive_example, retrieved) == 1.0


def test_snippet_hit_finds_match(positive_example):
    retrieved_texts = [
        "Whoever commits theft shall be punished with imprisonment...",
    ]
    assert snippet_hit_at_k(positive_example, retrieved_texts) == 1


def test_snippet_hit_no_match(positive_example):
    retrieved_texts = ["The weather is sunny today."]
    assert snippet_hit_at_k(positive_example, retrieved_texts) == 0