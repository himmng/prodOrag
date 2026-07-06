"""End-to-end API contract tests using FastAPI's TestClient.

No Ollama, no Chroma, no reranker — we bypass the lifespan startup and
inject mock retriever + LLM into the app's state dict.
"""


# Configure auth BEFORE importing anything from rag_pipeline
# (module-level _VALID_KEYS in middleware/auth.py is set at import time)

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk

# Override auth keys for tests — bypasses .env entirely
from rag_pipeline.api.middleware import auth as _auth_mod
_auth_mod._VALID_KEYS = {"test-key-1", "test-key-2"}

from rag_pipeline.api.main import app, _state


# ── Mocks ──────────────────────────────────────────────────────────────
class _MockRetriever:
    name = "mock"

    def retrieve(self, query: str, top_k: int = 5):
        return [
            (
                Document(
                    page_content=f"Mock content for query: {query}",
                    metadata={
                        "source_path":    "/test/mock_section.pdf",
                        "page_number":    1,
                        "section_title":  "Mock Section",
                        "chunk_id":       "mock-001",
                        "act":            "IPC",          # ← add
                        "section":        "302",          # ← add
                        "corresponds_to": "103",          # ← add
                        "change_status":  "unchanged",    # ← add
                    },
                ),
                0.5,
            ),
        ]
class _MockLLM:
    """Behaves like ChatOllama for both .invoke() and .stream()."""

    def invoke(self, prompt):
        return AIMessage(content="Mock answer with citation [1].")

    def stream(self, prompt):
        for tok in ["Mock ", "streaming ", "answer ", "[1]."]:
            yield AIMessageChunk(content=tok)


class _FakeCorpus:
    """Minimal corpus stand-in for the serving layer (no YAML/PDFs needed)."""
    name = "test"
    display_name = "Test Corpus"
    acts = ["IPC", "BNS"]
    has_concordance = False
    concordance = None
    context_collections: list[str] = []

    def pdf_map(self):
        return {}


# ── Fixtures ───────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    """TestClient with mocked state — lifespan is NOT triggered."""
    mock = _MockRetriever()
    llm  = _MockLLM()

    # Build the per-act structure the routes now expect
    act_setup = {
        "dense":             mock,
        "bm25":              mock,
        "ensemble":          mock,
        "hybrid_r":          mock,
        "hybrid_r_nofilter": mock,
        "n_chunks":          0,
    }

    _state.clear()
    _state.update({
        "corpus":      _FakeCorpus(),
        "by_act": {
            "IPC": dict(act_setup),
            "BNS": dict(act_setup),
        },
        "reranker":    None,
        "llm":         llm,
        "concordance": None,
        "eval_set":    [],
    })
    return TestClient(app)

# ── Health / meta ──────────────────────────────────────────────────────

def test_health_is_open(client):
    """/health requires no auth — important for orchestration probes."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()


def test_meta_exposes_corpus(client):
    """/meta drives corpus-agnostic clients — must report acts + display name."""
    body = client.get("/meta").json()
    assert body["acts"] == ["IPC", "BNS"]
    assert body["display_name"] == "Test Corpus"
    assert body["cross_reference"] is None  # fake corpus has no concordance


def test_answer_rejects_unknown_act(client):
    """collections is validated against the active corpus at runtime."""
    resp = client.post("/answer",
                       json={"query": "x", "collections": ["ZZZ"]},
                       headers={"X-API-Key": "test-key-1"})
    assert resp.status_code == 422


def test_answer_empty_collections_searches_all(client):
    """Empty collections → all corpus acts (no 422)."""
    resp = client.post("/answer",
                       json={"query": "theft", "collections": []},
                       headers={"X-API-Key": "test-key-1"})
    assert resp.status_code == 200


# ── Auth ───────────────────────────────────────────────────────────────

def test_answer_without_key_returns_401(client):
    resp = client.post("/answer", json={"query": "test"})
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate", "").lower() == "apikey"


def test_answer_with_invalid_key_returns_403(client):
    resp = client.post(
        "/answer",
        json={"query": "test"},
        headers={"X-API-Key": "bogus"},
    )
    assert resp.status_code == 403


def test_answer_with_valid_key_returns_200(client):
    resp = client.post(
        "/answer",
        json={"query": "test", "top_k": 1},
        headers={"X-API-Key": "test-key-1"},
    )
    assert resp.status_code == 200


# ── Response schema ────────────────────────────────────────────────────

def test_answer_response_has_required_fields(client):
    resp = client.post(
        "/answer",
        json={"query": "test", "top_k": 1},
        headers={"X-API-Key": "test-key-1"},
    )
    body = resp.json()
    for field in ("question", "answer", "citations", "retriever", "latency_ms"):
        assert field in body, f"missing field: {field}"


def test_citation_shape(client):
    resp = client.post(
        "/answer",
        json={"query": "test", "top_k": 1},
        headers={"X-API-Key": "test-key-1"},
    )
    citations = resp.json()["citations"]
    assert len(citations) >= 1
    cite = citations[0]
    for field in ("n", "source_path", "score"):
        assert field in cite, f"missing citation field: {field}"


# ── Rate-limit headers ─────────────────────────────────────────────────

def test_rate_limit_headers_present(client):
    resp = client.post(
        "/answer",
        json={"query": "test"},
        headers={"X-API-Key": "test-key-1"},
    )
    assert "x-ratelimit-limit" in resp.headers
    assert "x-ratelimit-remaining" in resp.headers


# ── Request-ID middleware ──────────────────────────────────────────────

def test_request_id_header_round_trip(client):
    resp = client.get("/health")
    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) >= 8


def test_request_id_is_echoed_when_provided(client):
    resp = client.get("/health", headers={"X-Request-ID": "abc123def456"})
    assert resp.headers["x-request-id"] == "abc123def456"


# ── Validation ─────────────────────────────────────────────────────────

def test_answer_rejects_empty_query(client):
    resp = client.post(
        "/answer",
        json={"query": ""},
        headers={"X-API-Key": "test-key-1"},
    )
    assert resp.status_code == 422  # Pydantic min_length=1 violation


def test_answer_rejects_unknown_retriever(client):
    resp = client.post(
        "/answer",
        json={"query": "x", "retriever": "garbage"},
        headers={"X-API-Key": "test-key-1"},
    )
    assert resp.status_code == 422  # Literal[...] enforcement