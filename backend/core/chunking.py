from typing import List


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        return [text]

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= length:
            break
        start = max(0, end - overlap)

    return chunks
