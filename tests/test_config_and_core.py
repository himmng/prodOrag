import asyncio
from pathlib import Path

from backend.core.config import AppConfig, load_config, save_config
from backend.core.chunking import chunk_text
from backend.core.vector_store import VectorStore
from backend.core.orchestrator import RAGOrchestrator
from backend.core.llm_client import LLMClient
from backend.core.embedding_client import EmbeddingClient


def test_load_and_save_config_roundtrip(tmp_path, monkeypatch):
    cfg = AppConfig()
    cfg.llm.model = "test-model"

    cfg_path = tmp_path / "config.yaml"

    def fake_default_path():
        return cfg_path

    # Use explicit path to avoid touching real config
    save_config(cfg, path=str(cfg_path))
    loaded = load_config(path=str(cfg_path))

    assert loaded.llm.model == "test-model"


def test_chunk_text_basic():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, overlap=2)

    # First chunk
    assert chunks[0] == text[0:10]
    # Overlap respected: second chunk starts 2 chars before previous end
    assert chunks[1].startswith(text[8:18])
    # All text covered
    assert "".join(chunks).startswith(text[0])


class DummyEmbeddingClient(EmbeddingClient):
    async def embed(self, texts):
        # simple deterministic embeddings (vector length 3)
        return [[float(len(t)), 0.0, 1.0] for t in texts]


class DummyLLMClient(LLMClient):
    async def chat(self, messages):
        # echo last user content plus marker
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"ANSWER: {last_user}"


def test_vector_store_add_and_query(tmp_path):
    cfg = AppConfig()
    cfg.storage.vector_dir = str(tmp_path)

    store = VectorStore(cfg)
    embeddings = [[1.0, 0.0, 0.0]]
    metadatas = [{"doc_id": "doc1", "chunk_index": 0, "text": "hello world"}]
    ids = ["doc1:0"]

    store.add_documents("doc1", embeddings, metadatas, ids)

    results = store.query(embeddings[0], top_k=1)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc1"
    assert "hello world" in results[0]["text"]


def test_orchestrator_chat_uses_context(tmp_path, monkeypatch):
    cfg = AppConfig()
    cfg.storage.vector_dir = str(tmp_path)

    # Prepare vector store with one chunk
    store = VectorStore(cfg)
    embeddings = [[5.0, 0.0, 1.0]]
    metadatas = [{"doc_id": "doc1", "chunk_index": 0, "text": "answer is 42"}]
    ids = ["doc1:0"]
    store.add_documents("doc1", embeddings, metadatas, ids)

    # Dummy embedding client returns same embedding for any text length 5
    class FixedEmbeddingClient(DummyEmbeddingClient):
        async def embed(self, texts):
            return [embeddings[0] for _ in texts]

    orch = RAGOrchestrator(
        cfg,
        llm_client=DummyLLMClient(),
        embedding_client=FixedEmbeddingClient(),
        vector_store=store,
    )

    async def run_chat():
        result = await orch.chat("hello", [])
        assert "ANSWER:" in result["answer"]
        assert result["sources"]
        assert result["sources"][0]["doc_id"] == "doc1"

    asyncio.run(run_chat())
