"""Contract tests for session-isolated case upload + /answer/case.

No Ollama / Chroma / embeddings — we inject a fake, in-memory doc store and a
fake parser into the app state, so these tests exercise ROUTE wiring, session
scoping, and response schemas (not the real vector isolation, which lives in
DocumentStore + vectorstore and is covered by construction).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag_pipeline.api.middleware import auth as _auth_mod
_auth_mod._VALID_KEYS = {"test-key-1"}

from rag_pipeline.api.main import app, _state
from rag_pipeline.api.documents import UploadedDoc, CaseExcerpt
from datetime import datetime, timedelta, timezone


class _MockRetriever:
    def retrieve(self, query, top_k=5):
        return [(
            Document(
                page_content=f"statute for {query}",
                metadata={"source_path": "/x.pdf", "page_number": 1,
                          "section_title": "Theft", "chunk_id": "c1",
                          "act": "IPC", "section": "378"},
            ),
            0.9,   # above MAIN_CITATION_FLOOR
        )]


class _MockLLM:
    def invoke(self, prompt):
        return AIMessage(content="The case attracts IPC 378 [1].")


class _MockContextRetriever:
    def retrieve(self, query, top_k=3):
        return [(
            Document(
                page_content="The Committee recommended clearer definitions.",
                metadata={"doc_type": "committee_report",
                          "display_name": "BNS Standing Committee Report",
                          "page_number": 42},
            ),
            0.4,   # above CONTEXT_FLOOR (0.15)
        )]


class _FakeDocStore:
    """In-memory, session-scoped — mirrors DocumentStore's interface sans vectors."""

    def __init__(self):
        self._s: dict[str, dict[str, UploadedDoc]] = {}

    def _mk(self, session_id, filename):
        now = datetime.now(timezone.utc)
        return UploadedDoc(
            doc_id=f"doc{len(self._s.get(session_id, {}))}",
            session_id=session_id, filename=filename,
            collection_name=f"case_{session_id}_x", char_count=100, n_chunks=2,
            uploaded_at=now.isoformat(), expires_at=now + timedelta(hours=1),
            section_refs=["302"],
        )

    def add(self, session_id, filename, chunks):
        doc = self._mk(session_id, filename)
        self._s.setdefault(session_id, {})[doc.doc_id] = doc
        return doc

    def get(self, session_id, doc_id):
        return self._s.get(session_id, {}).get(doc_id)

    def list_all(self, session_id):
        return list(self._s.get(session_id, {}).values())

    def delete(self, session_id, doc_id):
        return self._s.get(session_id, {}).pop(doc_id, None) is not None

    def search_case(self, session_id, doc_id, query, top_k=5):
        if not self.get(session_id, doc_id):
            return []
        return [CaseExcerpt(text="Accused stole a phone.", score=0.8,
                            page_number=1, section_title="Facts")]


class _FakeParser:
    def parse(self, path):
        return []  # unused (we upload txt in tests)


@pytest.fixture(scope="module")
def client():
    mock = _MockRetriever()
    act = {"dense": mock, "bm25": mock, "ensemble": mock,
           "hybrid_r": mock, "hybrid_r_nofilter": mock, "n_chunks": 0}
    _state.clear()
    _state.update({
        "by_act": {"IPC": dict(act), "BNS": dict(act)},
        "reranker": None, "llm": _MockLLM(), "concordance": None,
        "doc_store": _FakeDocStore(), "uploaded_parser": _FakeParser(),
        "context_retriever": _MockContextRetriever(),
    })
    return TestClient(app)


H = {"X-API-Key": "test-key-1"}


def _upload(client, session, name="case.txt", body=b"Accused stole a phone under section 302."):
    return client.post(
        "/documents/upload",
        files={"file": (name, body, "text/plain")},
        headers={**H, "X-Session-Id": session},
    )


def test_upload_requires_session_header(client):
    r = client.post("/documents/upload",
                    files={"file": ("c.txt", b"hello world", "text/plain")}, headers=H)
    assert r.status_code == 422  # missing X-Session-Id


def test_upload_returns_docinfo(client):
    r = _upload(client, "sessA")
    assert r.status_code == 200
    body = r.json()
    assert body["doc_id"] and body["filename"] == "case.txt"


def test_isolation_list_scoped_to_session(client):
    _upload(client, "sessB", name="b.txt")
    a = client.get("/documents", headers={**H, "X-Session-Id": "sessA"}).json()
    b = client.get("/documents", headers={**H, "X-Session-Id": "sessB"}).json()
    a_files = {d["filename"] for d in a["documents"]}
    b_files = {d["filename"] for d in b["documents"]}
    assert "b.txt" in b_files and "b.txt" not in a_files  # B's file invisible to A


def test_answer_case_shape(client):
    doc_id = _upload(client, "sessC").json()["doc_id"]
    r = client.post("/answer/case",
                    json={"doc_id": doc_id, "question": "Which sections apply?"},
                    headers={**H, "X-Session-Id": "sessC"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert len(body["case_excerpts"]) == 1
    assert body["case_excerpts"][0]["section_title"] == "Facts"
    assert len(body["citations"]) >= 1  # IPC statute citation


def test_answer_context_opt_in(client):
    # Default: general /answer excludes commentary
    off = client.post("/answer", json={"query": "theft"}, headers=H).json()
    assert off.get("context") == []
    # Opt-in: commentary surfaces in its own field, labeled non-authoritative
    on = client.post("/answer",
                     json={"query": "theft", "include_context": True}, headers=H).json()
    assert len(on["context"]) == 1
    assert on["context"][0]["doc_type"] == "committee_report"


def test_answer_case_includes_context_by_default(client):
    doc_id = _upload(client, "sessCtx").json()["doc_id"]
    r = client.post("/answer/case",
                    json={"doc_id": doc_id, "question": "Which sections apply?"},
                    headers={**H, "X-Session-Id": "sessCtx"}).json()
    assert len(r["context"]) == 1  # case Q&A includes commentary without opt-in


def test_answer_case_cross_session_404(client):
    doc_id = _upload(client, "sessD").json()["doc_id"]
    # sessE tries to read sessD's case → must not resolve
    r = client.post("/answer/case",
                    json={"doc_id": doc_id, "question": "?"},
                    headers={**H, "X-Session-Id": "sessE"})
    assert r.status_code == 404
