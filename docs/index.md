# protoRAG

A **corpus-agnostic**, production-bound hybrid RAG pipeline for legal corpora. The
reference corpus is Indian criminal law — the Indian Penal Code (IPC 1860) and the
Bharatiya Nyaya Sanhita (BNS 2023) with an authoritative IPC↔BNS concordance — but
nothing IPC/BNS-specific is hardcoded in the core. Swapping in a different corpus is
a matter of dropping a new corpus config + PDFs, setting `RAG_CORPUS`, and
re-ingesting — no code changes.

Hybrid retrieval (BM25 + dense + cross-encoder rerank) → grounded, citation-aware
LLM answers, served over FastAPI with SSE streaming, plus a Streamlit dashboard.

This site has two parts:

- **[Usage Guide](USAGE.md)** — architecture, configuration reference, corpus
  format, and how to use every API endpoint and eval workflow.
- **[API Reference](reference.md)** — auto-generated from docstrings in
  `src/rag_pipeline/`, kept in sync with the code automatically.

For the HTTP API's exact request/response schema, use the running app's own
`/docs` (Swagger) and `/redoc` — those are generated live from the code and are
always accurate.

## Quickstart

```bash
git clone https://github.com/himmng/protoRAG.git
cd protoRAG

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

cp .env.example .env          # then edit for your model provider
rag-ingest --corpus ipc_bns --clean
uvicorn rag_pipeline.api.main:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/docs`. See the [Usage Guide](USAGE.md) for the
dashboard, Docker, and full configuration options.
