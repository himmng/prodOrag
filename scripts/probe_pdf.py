"""Look at what Docling extracts from each PDF — first 3000 chars."""

from rag_pipeline.corpus import load_corpus
from rag_pipeline.parsers.docling import DoclingHybridParser


def main():
    corpus = load_corpus("ipc_bns")
    parser = DoclingHybridParser()

    for src in corpus.sources:
        print(f"\n{'='*60}")
        print(f"FILE: {src.pdf_path.name}")
        print(f"{'='*60}")
        doc_chunks = parser.parse(src.pdf_path)
        full = "\n".join(
            (dc.text if hasattr(dc, "text") else str(dc))
            for dc in doc_chunks
        )
        print(f"Total chars: {len(full):,}")
        print(f"Doc chunks: {len(doc_chunks)}")
        print(f"\n--- First 3000 chars ---\n")
        print(full[:3000])
        print(f"\n--- Around char 30000 (deep in doc) ---\n")
        print(full[30000:33000] if len(full) > 33000 else "(too short)")

    # Concordance separately
    print(f"\n{'='*60}")
    print(f"FILE: {corpus.concordance.pdf_path.name}")
    print(f"{'='*60}")
    doc_chunks = parser.parse(corpus.concordance.pdf_path)
    print(f"Doc chunks: {len(doc_chunks)}")
    print(f"\n--- First 5 doc-chunks (verbatim) ---\n")
    for i, dc in enumerate(doc_chunks[:5]):
        text = dc.text if hasattr(dc, "text") else str(dc)
        print(f"[chunk {i}] ({len(text)} chars):")
        print(repr(text[:500]))
        print()


if __name__ == "__main__":
    main()