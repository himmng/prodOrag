[![CI](https://github.com/himmng/protorag/actions/workflows/ci.yml/badge.svg)](https://github.com/protorag/actions/workflows/ci.yml)

# protoRAG

Hybrid + agentic retrieval-augmented generation (RAG) pipeline for legal corpora, with a focus on IPC case law.

## What this project does

`protoRAG` combines lexical, dense, and reranked retrieval strategies with LLM-driven answer generation to support:

- multi-strategy retrieval over a legal corpus
- citation-aware answer generation
- streaming LLM responses via Server-Sent Events (SSE)
- retrieval evaluation and threshold calibration
- in-memory ad-hoc document upload for temporary QA

## Key features

- `fastapi` service exposing `/answer`, `/answer/stream`, and evaluation endpoints
- configurable retrievers: `bm25`, `dense`, `ensemble`, and `hybrid_reranked`
- reusable startup singleton pipeline for low latency
- vendor-agnostic model provider support: Ollama, Azure OpenAI, AWS Bedrock, GCP Vertex AI, OpenAI
- auth guard with optional API key mode
- reusable prompt and eval templates shipped with the package

## Getting started

### Install locally

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Install development dependencies if you want to run tests:

```bash
pip install -e '.[dev]'
```

### Run the API

```bash
uvicorn rag_pipeline.api.main:app --host 0.0.0.0 --port 8000
```

Then open:

- `http://localhost:8000/docs` for the interactive OpenAPI docs
- `http://localhost:8000/health` for a quick health check

### Run with Docker

Build and start the service using Docker Compose:

```bash
docker compose up --build
```

The service will be exposed on port `8000`.

## Required data

The API expects a preprocessed chunk cache at:

- `data/processed/phase1_chunks.json`

If this file is missing, the app will raise an error at startup. The repository contains notebooks under `notebooks/` for data preparation and ingestion.

## Configuration

Runtime configuration is controlled by environment variables or a `.env` file in the project root.

Important variables:

- `API_KEYS` — comma-separated keys for API auth; empty value disables auth for development
- `MODEL_PROVIDER` — one of `ollama`, `azure`, `openai`, `gcp`, or `aws`
- `OLLAMA_HOST`, `OLLAMA_PORT`, `OLLAMA_MODEL`, `OLLAMA_EMBEDDING_MODEL`
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`
- `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`
- `AWS_REGION`, `AWS_BEDROCK_MODEL_ID`, `AWS_BEDROCK_EMBEDDING_MODEL_ID`
- `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_VERTEX_MODEL`, `GCP_VERTEX_EMBEDDING_MODEL`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`

### Example `.env`

```text
MODEL_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma-4-e4b:latest
OLLAMA_EMBEDDING_MODEL=embeddinggemma:latest
API_KEYS=dev-key-123
LOG_LEVEL=INFO
```

## API overview

### Health check

`GET /health`

Returns service readiness and component status.

### Generate an answer

`POST /answer`

Request body:

```json
{
  "query": "What is the standard for patent claim interpretation?",
  "top_k": 5,
  "retriever": "hybrid_reranked",
  "fetch_k": 20,
  "min_score": 0.01
}
```

Response includes:

- `question`
- `answer`
- `citations`
- `retriever`
- `latency_ms`

### Streaming answer

`POST /answer/stream`

Returns Server-Sent Events with citations, incremental tokens, and a final completion event.

### Retrieval evaluation

`POST /eval/retrieval`

Run retrieval metrics against the built-in eval set.

### Threshold calibration

`POST /eval/threshold-sweep`

Sweep confidence thresholds and return recommended values for the hybrid reranked retriever.

### Ad-hoc document upload

`POST /documents/upload`

Upload `pdf`, `txt`, or `md` files for temporary query support. Uploaded documents are stored in memory only and are not persisted to the main ChromaDB corpus.

`GET /documents`

List uploaded documents.

`DELETE /documents/{doc_id}`

Remove a previously uploaded document.

## Project layout

- `src/rag_pipeline/` — core package
  - `api/` — FastAPI app, routes, auth, logging, SSE
  - `retrainers/` — retrieval implementations (`bm25`, `dense`, `ensemble`, `reranker`)
  - `generation/` — answer generation and streaming logic
  - `eval/` — retrieval evaluation, threshold sweep, RAGAS support
  - `parsers/` — document parsing and chunking support
  - `prompts/` — prompt templates shipped with package
- `data/` — raw and processed corpus artifacts
- `chroma_db/` — persisted Chroma collection data
- `configs/` — corpus-specific configuration
- `notebooks/` — ingestion and evaluation experimentation
- `tests/` — pytest coverage for API, parsing, and evaluation

## Development notes

- The FastAPI app uses a startup lifespan handler to load heavy objects once and reuse them across requests.
- The default hybrid retriever is a two-stage semantic + reranked pipeline calibrated for legal retrieval.
- The service is designed for local experimentation with switchable model providers.

## Testing

Run tests with:

```bash
pytest
```

## Contribution

This repository is in alpha stage. Please open issues for bug reports, feature requests, or questions about legal retrieval pipelines.
