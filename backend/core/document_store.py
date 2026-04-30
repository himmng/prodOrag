import json
from pathlib import Path
from typing import Any, Dict

from fastapi import UploadFile

from backend.core.config import AppConfig
from backend.core.chunking import chunk_text
from backend.core.embedding_client import create_embedding_client
from backend.core.vector_store import get_vector_store


async def save_document_and_index(file: UploadFile, config: AppConfig) -> None:
    storage_dir = Path(config.storage.docs_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    file_path = storage_dir / file.filename
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    file_path.write_text(text, encoding="utf-8")

    doc_id = file.filename

    chunks = chunk_text(text, config.rag.chunk_size, config.rag.chunk_overlap)

    embedding_client = create_embedding_client(config)
    embeddings = []
    metadatas = []
    ids = []

    for idx, chunk in enumerate(chunks):
        vecs = await embedding_client.embed([chunk])
        if not vecs:
            continue
        embeddings.append(vecs[0])
        metadatas.append(
            {
                "doc_id": doc_id,
                "chunk_index": idx,
                "file_name": file.filename,
                "path": str(file_path),
                "text": chunk,
            }
        )
        ids.append(f"{doc_id}:{idx}")

    if embeddings:
        vector_store = get_vector_store(config)
        vector_store.add_documents(doc_id, embeddings, metadatas, ids)

    index_path = storage_dir / "index.json"
    index: Dict[str, Any] = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            index = {}

    index[doc_id] = {
        "file_name": file.filename,
        "path": str(file_path),
    }

    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
