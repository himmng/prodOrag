# scripts/find_body_start.py
from rag_pipeline.corpus import load_corpus
from rag_pipeline.parsers.docling import DoclingHybridParser


def main():
    corpus = load_corpus("ipc_bns")
    parser = DoclingHybridParser()

    for src in corpus.sources:
        print(f"\n{'='*60}")
        print(f"FILE: {src.pdf_path.name}  (act={src.act})")
        print(f"{'='*60}\n")
        
        doc_chunks = parser.parse(src.pdf_path)
        full = "\n".join(
            (dc.text if hasattr(dc, "text") else str(dc))
            for dc in doc_chunks
        )
        print(f"Total length: {len(full):,} chars\n")

        # Show 5 evenly-spaced 500-char windows
        n_windows = 5
        for i in range(n_windows):
            offset = (len(full) * i) // n_windows
            print(f"\n--- Window {i+1}/{n_windows} (offset={offset:,}) ---")
            print(full[offset:offset + 500])


if __name__ == "__main__":
    main()