from fastapi.testclient import TestClient
from rag_pipeline.api.main import app, _state
from rag_pipeline.api.middleware import auth as _auth_mod
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk

_auth_mod._VALID_KEYS = {"test-key-1", "test-key-2"}


class _MockRetriever:
    name = "mock"

    def retrieve(self, query: str, top_k: int = 5):
        return [
            (
                Document(
                    page_content=f"Mock content for query: {query}",
                    metadata={
                        "source_path": "/test/mock_section.pdf",
                        "page_number": 1,
                        "section_title": "Mock Section",
                        "chunk_id": "mock-001",
                    },
                ),
                0.5,
            ),
        ]


class _MockLLM:
    def invoke(self, prompt):
        return AIMessage(content="Mock answer with citation [1].")

    def stream(self, prompt):
        for tok in ["Mock ", "streaming ", "answer ", "[1]."]:
            yield AIMessageChunk(content=tok)


def main():
    mock = _MockRetriever()
    llm = _MockLLM()

    _state.clear()
    _state.update({
        "chunks": [],
        "dense": mock,
        "bm25": mock,
        "ensemble": mock,
        "reranker": None,
        "hybrid_r": mock,
        "hybrid_r_nofilter": mock,
        "llm": llm,
        "eval_set": [],
    })

    client = TestClient(app)

    try:
        resp = client.post(
            "/answer",
            json={"query": "test", "top_k": 1},
            headers={"X-API-Key": "test-key-1"},
        )
        print("STATUS:", resp.status_code)
        print("HEADERS:", resp.headers)
        print("BODY:\n", resp.text)
    except Exception as e:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
