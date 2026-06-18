"""Unit tests for the SSE formatter."""

import json
from rag_pipeline.api.sse import sse_event


def test_sse_event_basic_shape():
    out = sse_event("token", {"text": "hello"})
    lines = out.strip().split("\n")
    assert lines[0] == "event: token"
    assert lines[1].startswith("data: ")
    assert json.loads(lines[1][6:]) == {"text": "hello"}


def test_sse_event_terminator_is_double_newline():
    out = sse_event("done", {"latency_ms": 1234})
    assert out.endswith("\n\n")


def test_sse_event_handles_unicode():
    out = sse_event("token", {"text": "Sec. § 302"})
    payload = json.loads(out.strip().split("\n", 1)[1][6:])
    assert payload["text"] == "Sec. § 302"