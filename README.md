[![CI](https://github.com/himmng/prodOrag/actions/workflows/ci.yml/badge.svg)](https://github.com/himmng/prodOrag/actions/workflows/ci.yml)

# prodOrag

A **corpus-agnostic**, production-bound hybrid RAG pipeline for legal corpora. The
reference corpus is Indian criminal law — the Indian Penal Code (IPC 1860) and the
Bharatiya Nyaya Sanhita (BNS 2023) with an authoritative IPC↔BNS concordance — but
nothing IPC/BNS-specific is hardcoded in the core. Swapping in a different corpus is
a matter of dropping a new corpus config + PDFs, setting `RAG_CORPUS`, and
re-ingesting — no code changes.

Hybrid retrieval (BM25 + dense + cross-encoder rerank) → grounded, citation-aware
LLM answers, served over FastAPI with SSE streaming, plus a Streamlit dashboard.

**Looking for the full guide?** → [docs/USAGE.md](docs/USAGE.md) covers
architecture, every config variable, the corpus format, and how to use each API
endpoint and eval workflow. This README is just the fast path to a running instance.

## Requirements

- Python **>= 3.12**
- An [Ollama](https://ollama.com) instance (default provider) *or* credentials for
  Azure OpenAI / OpenAI / AWS Bedrock / GCP Vertex AI

## Quickstart

```bash
git clone https://github.com/himmng/prodOrag.git
cd prodOrag

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install --upgrade pip

# option A — editable install (recommended for development)
pip install -e .

# option B — flat requirements file (if you just want it running)
pip install -r requirements.txt
pip install -e . --no-deps
```

Copy the example environment file and adjust it for your model provider:

```bash
cp .env.example .env
```

By default the app expects Ollama running locally with `OLLAMA_MODEL` and
`OLLAMA_EMBEDDING_MODEL` pulled. To use a hosted provider instead, set
`LLM_PROVIDER` / `EMBEDDING_PROVIDER` and the matching credentials — see
[docs/USAGE.md § Configuration](docs/USAGE.md#configuration) for the full list.

### Ingest the reference corpus

```bash
rag-ingest --corpus ipc_bns --clean
```

This parses the PDFs in `data/raw/`, builds the Chroma vector collections, and
writes chunk caches to `data/processed/`. See
[docs/USAGE.md § Ingesting a corpus](docs/USAGE.md#ingesting-a-corpus) for options
and how to add your own corpus.

### Run the API

```bash
uvicorn rag_pipeline.api.main:app --host 0.0.0.0 --port 8000
```

Then open:

- `http://localhost:8000/docs` — interactive Swagger UI (try every endpoint)
- `http://localhost:8000/redoc` — ReDoc reference view
- `http://localhost:8000/health` — readiness check
- `http://localhost:8000/meta` — active-corpus metadata

### Run the dashboard

```bash
streamlit run apps/dashboard/app.py
```

The dashboard talks to the API at `API_URL` (default `http://localhost:8000`) and
authenticates with `DASHBOARD_API_KEY` — both read from `.env`.

### Run with Docker

```bash
docker compose up --build     # API on :8000
```

### Ask your first question

```bash
curl -s -X POST http://localhost:8000/answer \
  -H "X-API-Key: dev-key-123" -H "Content-Type: application/json" \
  -d '{"query": "What is the punishment for theft?", "top_k": 5}' | python -m json.tool
```

## Testing

```bash
pytest
```

## Docs site

The usage guide and API reference (auto-generated from docstrings) also build into
a browsable site via [MkDocs](https://www.mkdocs.org):

```bash
pip install -e ".[docs]"
mkdocs serve      # http://127.0.0.1:8000 — live-reloads on edits
mkdocs build      # static site in site/
```

## Project layout

- `src/rag_pipeline/` — core package (API, retrievers, parsers, generation, eval, CLI)
- `apps/dashboard/` — Streamlit dashboard
- `configs/corpora/` — corpus configuration YAMLs
- `data/` — raw and processed corpus artifacts
- `chroma_db/` — persisted Chroma collections
- `tests/` — pytest coverage
- `docs/USAGE.md` — full usage guide, config reference, API workflows

## Contribution

This repository is in alpha. Please open issues for bug reports, feature requests,
or questions about the retrieval pipeline.
