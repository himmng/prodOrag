from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File

from ..config.store import load_config


router = APIRouter()


@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)) -> dict:
    config = load_config()
    base = Path(config.file_store_path)
    base.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        dest = base / f.filename
        with dest.open("wb") as out:
            out.write(await f.read())
        saved.append(str(dest))

    return {"saved": saved}
