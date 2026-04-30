import asyncio
from fastapi.testclient import TestClient

from backend.main import app
from backend.core.config import AppConfig, save_config


client = TestClient(app)


class DummyLLM:
    async def chat(self, messages):
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"API ANSWER: {last_user}"


def setup_test_config(tmp_path):
    cfg = AppConfig()
    cfg.storage.vector_dir = str(tmp_path / "vec")
    cfg.storage.docs_dir = str(tmp_path / "docs")
    save_config(cfg, path=str(tmp_path / "config.yaml"))
    return cfg


def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_config_get_and_post(tmp_path, monkeypatch):
    # redirect default config path to temp
    from backend import core

    cfg = setup_test_config(tmp_path)

    resp = client.get("/api/config")
    assert resp.status_code == 200

    update = {"rag": {"top_k": 3}}
    resp2 = client.post("/api/config", json=update)
    assert resp2.status_code == 200
    assert resp2.json()["rag"]["top_k"] == 3


def test_chat_endpoint_no_docs(tmp_path, monkeypatch):
    # Ensure config uses temp dirs
    setup_test_config(tmp_path)

    resp = client.post(
        "/api/chat",
        json={"message": "hello", "history": [], "conversation_id": None},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data


def test_upload_document_and_chat(tmp_path, monkeypatch):
    setup_test_config(tmp_path)

    # Upload a small text document
    files = {"file": ("doc.txt", b"This is a test document about protoRAG.")}
    resp = client.post("/api/documents", files=files)
    assert resp.status_code == 200

    resp2 = client.post(
        "/api/chat",
        json={"message": "protoRAG", "history": [], "conversation_id": None},
    )
    assert resp2.status_code == 200
    data = resp2.json()
    assert "answer" in data
    assert isinstance(data["sources"], list)
