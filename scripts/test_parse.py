# scripts/test_parse.py
"""Smoke test: parse one statute PDF + the concordance PDF."""

from rag_pipeline.corpus import load_corpus
from rag_pipeline.corpus.concordance import ConcordanceParser, save_rows
from rag_pipeline.parsers.statute import StatuteParser


def main():
    corpus = load_corpus("ipc_bns")
    print(f"\n=== Corpus: {corpus.display_name} ===\n")

    # Test 1: IPC parse (faster than BNS for first try; whichever you prefer)
    print("--- Parsing IPC ---")
    ipc_source = next(s for s in corpus.sources if s.act == "IPC")
    statute_parser = StatuteParser(corpus.parser)
    ipc_chunks = statute_parser.parse(ipc_source.pdf_path, act="IPC")

    print(f"\nFirst 3 chunks:")
    for c in ipc_chunks[:3]:
        print(f"  [{c.chunk_type}] §{c.section} ch.{c.chapter_number}")
        print(f"    title: {c.section_title[:60]}")
        print(f"    text:  {c.text[:120]!r}")
        print(f"    page:  {c.page_number}, chars: {c.char_count}")
        print()

    # Test 2: Concordance parse
    print("\n--- Parsing concordance ---")
    conc_parser = ConcordanceParser()
    rows = conc_parser.parse(corpus.concordance.pdf_path)

    print(f"\nFirst 5 rows:")
    for r in rows[:5]:
        print(f"  BNS {r.bns_section!r} ↔ IPC {r.ipc_section!r}  [{r.status}]")
        print(f"    {r.bns_title} / {r.ipc_title}")

    # Save concordance (we'll use it in Stage D)
    save_rows(rows, corpus.concordance.output_json)


if __name__ == "__main__":
    main()