"""Corpus ingest CLI.

Reads a corpus YAML, parses every PDF source, joins concordance metadata
into chunks, persists chunks to JSON, and ingests them into ChromaDB.

Usage:
    rag-ingest --corpus ipc_bns
    rag-ingest --corpus ipc_bns --skip-concordance        # statutes only
    rag-ingest --corpus ipc_bns --only IPC                # single source
    rag-ingest --corpus ipc_bns --dry-run                 # parse, don't write
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rag_pipeline.config import cfg, log
from rag_pipeline.corpus import load_corpus
from rag_pipeline.corpus.concordance import (
    Concordance, ConcordanceParser, save_rows,
)
from rag_pipeline.parsers.statute import StatuteParser
from rag_pipeline.providers import get_embeddings
from rag_pipeline.schemas import StatuteChunk
from rag_pipeline.vectorstore import get_vectorstore


# ── Helpers ───────────────────────────────────────────────────────────

def _enrich_with_concordance(
    chunks: list[StatuteChunk],
    concordance: Concordance,
) -> tuple[int, int]:
    """Mutate each chunk in place, populating `corresponds_to` and `change_status`.

    Returns (n_enriched, n_skipped).
    """
    enriched = skipped = 0
    for c in chunks:
        if c.act == "IPC":
            row = concordance.lookup_ipc(c.section)
            if row:
                c.corresponds_to = row.bns_section
                c.change_status  = row.status
                enriched += 1
            else:
                skipped += 1
        elif c.act == "BNS":
            row = concordance.lookup_bns(c.section)
            if row:
                c.corresponds_to = row.ipc_section
                c.change_status  = row.status
                enriched += 1
            else:
                skipped += 1
    return enriched, skipped


def _save_chunks_json(chunks: list[StatuteChunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.model_dump() for c in chunks]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Wrote {len(chunks)} chunks → {path}")


def _ingest_to_chroma(
    chunks: list[StatuteChunk],
    collection_name: str,
    batch_size: int = 32,
    clean: bool = False,
) -> None:
    if clean:
        import chromadb
        client = chromadb.PersistentClient(path=str(cfg.PROJECT_ROOT / "chroma_db"))
        try:
            client.delete_collection(collection_name)
            log.info(f"  Deleted existing collection: {collection_name}")
        except Exception:
            pass   # didn't exist, no-op
    """Embed chunks and upsert into a ChromaDB collection.

    Uses content-hash chunk_id as the Chroma document ID, so repeated
    ingests are idempotent (existing IDs are upserted, not duplicated).
    """
    log.info(f"Embedding + upserting {len(chunks)} chunks → {collection_name}")
    vs = get_vectorstore(collection_name)
    embeddings = get_embeddings()

    # ChromaDB accepts lists in parallel; we batch for memory
    n = len(chunks)
    for i in range(0, n, batch_size):
        batch = chunks[i:i + batch_size]
        ids       = [c.chunk_id for c in batch]
        documents = [c.text for c in batch]
        # Drop None values from metadata (Chroma rejects them)
        metadatas = [
            {k: v for k, v in c.model_dump().items()
             if k != "text" and v is not None}
            for c in batch
        ]
        vs.add_texts(texts=documents, metadatas=metadatas, ids=ids)
        log.info(f"  Batch {i // batch_size + 1}/{(n + batch_size - 1) // batch_size}: +{len(batch)}")

    final_count = vs._collection.count()  # underscore is intentional — Chroma API
    log.info(f"  Collection '{collection_name}' now has {final_count} vectors")


# ── Main ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-ingest",
        description="Ingest a corpus (parse PDFs → enrich → write JSON + ChromaDB)",
    )
    parser.add_argument(
        "--corpus", required=True,
        help="Corpus name (matches configs/corpora/<name>.yaml)",
    )
    parser.add_argument(
        "--only", choices=["IPC", "BNS"], default=None,
        help="Process only this act (skip the other)",
    )
    parser.add_argument(
        "--skip-concordance", action="store_true",
        help="Skip parsing the concordance PDF (use existing JSON if present)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and enrich, but skip ChromaDB writes",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Embedding/upsert batch size (default 32)",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete target collections before ingest (idempotent rebuild)",
    )
    args = parser.parse_args(argv)

    t0 = time.time()
    log.info("=" * 60)
    log.info(f"Ingest start: corpus={args.corpus}")
    if args.only:
        log.info(f"  Filter: only {args.only}")
    if args.dry_run:
        log.info("  DRY RUN: no ChromaDB writes")
    log.info("=" * 60)

    corpus = load_corpus(args.corpus)
    log.info(f"Loaded corpus: {corpus.display_name}")

    # ── Step 1: parse / load concordance ─────────────────────────────
    concordance: Concordance | None = None
    if corpus.concordance:
        conc_json_path = corpus.concordance.output_json
        if args.skip_concordance and conc_json_path.exists():
            log.info(f"Loading existing concordance: {conc_json_path}")
            concordance = Concordance.from_json(conc_json_path)
        else:
            log.info(f"Parsing concordance PDF: {corpus.concordance.pdf_path}")
            cp = ConcordanceParser()
            rows = cp.parse(corpus.concordance.pdf_path)
            save_rows(rows, conc_json_path)
            concordance = Concordance(rows)

    # ── Step 2: parse + enrich + ingest each source ──────────────────
    statute_parser = StatuteParser(corpus.parser)

    for source in corpus.sources:
        if args.only and source.act != args.only:
            log.info(f"Skipping {source.act} (--only filter)")
            continue

        log.info("-" * 60)
        log.info(f"Source: {source.act}  ({source.pdf_path.name})")
        log.info("-" * 60)

        chunks = statute_parser.parse(
            source.pdf_path,
            act=source.act,
            body_start_page=source.body_start_page,
        )

        if concordance:
            enriched, skipped = _enrich_with_concordance(chunks, concordance)
            log.info(f"  Concordance: enriched {enriched}, no-match {skipped}")

        _save_chunks_json(chunks, source.chunks_output)

        if args.dry_run:
            log.info("  Dry run — skipping ChromaDB write")
        else:
            _ingest_to_chroma(chunks, source.collection, args.batch_size, clean=args.clean)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"Ingest complete in {elapsed:.1f}s")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())