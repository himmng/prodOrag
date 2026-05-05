# test_rag.py  (run with: python test_rag.py)
import asyncio
from backend.core.config import get_config
from backend.core.orchestrator import get_orchestrator

async def main():
    config = get_config()
    orch   = get_orchestrator(config)

    # ── Test 1: plain chat before any upload ────────────────────────
    print("\n=== Test 1: Plain CHAT mode ===")
    r = await orch.chat("What is the capital of France?")
    print(f"Mode   : {r['mode']}")
    print(f"Answer : {r['answer']}")

    # ── Test 2: upload a document ────────────────────────────────────
    print("\n=== Test 2: Upload document ===")
    from backend.core.document_store import save_document_and_index

    # Use a real file on your machine
    FILE_PATH = "/Users/himanshutiwari/Downloads/CV_Himanshu.pdf"

    # Simulate UploadFile with a simple mock
    class MockUploadFile:
        def __init__(self, path):
            self.filename = path.split("/")[-1]
            self._path    = path
        async def read(self):
            with open(self._path, "rb") as f:
                return f.read()

    result = await save_document_and_index(
        MockUploadFile(FILE_PATH), config, conversation_id="test_conv"
    )
    print(result["message"])

    # Notify orchestrator so it flips to RAG mode
    orch.notify_document_added(conversation_id="test_conv")

    # ── Test 3: RAG query ────────────────────────────────────────────
    print("\n=== Test 3: RAG mode query ===")
    r = await orch.chat(
        "Give a brief summary of this document.",
        conversation_id="test_conv",
    )
    print(f"Mode    : {r['mode']}")
    print(f"Answer  : {r['answer']}")
    print(f"Sources : {len(r['sources'])} chunks retrieved")

    # ── Test 4: opt out → plain chat ────────────────────────────────
    print("\n=== Test 4: disable_rag() ===")
    print(orch.disable_rag(conversation_id="test_conv"))
    r = await orch.chat("Tell me a fun fact.", conversation_id="test_conv")
    print(f"Mode   : {r['mode']}")
    print(f"Answer : {r['answer']}")

    # ── Test 5: opt back in ──────────────────────────────────────────
    print("\n=== Test 5: enable_rag() ===")
    print(orch.enable_rag(conversation_id="test_conv"))
    r = await orch.chat(
        "What skills are listed?",
        conversation_id="test_conv",
    )
    print(f"Mode   : {r['mode']}")
    print(f"Answer : {r['answer']}")

    # ── Test 6: status ───────────────────────────────────────────────
    print("\n=== Test 6: status() ===")
    print(orch.status(conversation_id="test_conv"))

asyncio.run(main())