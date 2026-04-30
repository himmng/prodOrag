from pathlib import Path
import asyncio

from starlette.datastructures import UploadFile as StarletteUploadFile

from backend.core.config import get_config
from backend.core.document_store import save_document_and_index


def make_upload_file(file_path: Path) -> StarletteUploadFile:
    """
    Wrap a local file as a Starlette/FastAPI UploadFile.
    """
    f = file_path.open("rb")
    return StarletteUploadFile(filename=file_path.name, file=f)


async def main() -> None:
    # 1. Load AppConfig from config/config.yaml via existing helper
    config = get_config()

    print("Loaded config:")
    print(f"  embeddings provider: {config.embeddings.provider}")
    print(f"  embeddings base_url: {config.embeddings.base_url}")
    print(f"  embeddings model:    {config.embeddings.model}")
    print(f"  docs_dir:            {config.storage.docs_dir}")
    print(f"  vector_dir:          {config.storage.vector_dir}")
    print()

    docs_dir = Path(config.storage.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # 2. Create a small text document in docs_dir
    test_doc_path = docs_dir / "test_vector_store.txt"
    test_content = "This is a small test document for verifying local embeddings and vector store.\n"
    test_doc_path.write_text(test_content, encoding="utf-8")
    print(f"Created test document at: {test_doc_path}")

    # 3. Wrap it as UploadFile and call the existing pipeline
    upload_file = make_upload_file(test_doc_path)

    # conversation_id=None ensures it uses the base vector_dir from config.storage.vector_dir
    await save_document_and_index(
        file=upload_file,
        config=config,
        conversation_id=None,
    )

    print("save_document_and_index completed.")
    print()

    # 4. Verify that the vector directory exists and contains files (Chroma DB)
    vector_dir = Path(config.storage.vector_dir)
    if not vector_dir.exists():
        raise RuntimeError(f"Vector dir {vector_dir} does not exist")

    files = [p for p in vector_dir.glob("**/*") if p.is_file()]
    if not files:
        raise RuntimeError(f"No files found in vector dir {vector_dir}")

    print(f"Vector store directory {vector_dir} populated with {len(files)} files:")
    for f in files:
        print(" -", f)


if __name__ == "__main__":
    asyncio.run(main())