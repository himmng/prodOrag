# Python-native Local-first RAG App

This project is a production-ready, Python-native full stack RAG application with:

- ChatGPT/NotebookLM-style chat UI
- Local file upload and retrieval-augmented generation (RAG)
- OpenAI-compatible configuration for LLM and embeddings (local or cloud)
- Local disk vector store and file storage
- Docker containerization
- Tailscale-friendly deployment

---

## 1. Prerequisites

- Python 3.10+
- `pip`
- (Optional) Docker
- (Optional) A local OpenAI-compatible server, e.g. Ollama or LM Studio

Example Ollama setup (on your host):

```bash
ollama serve
# pull a chat model
ollama pull llama3
# pull an embedding model
ollama pull nomic-embed-text
```

Configure Ollama to expose an OpenAI-compatible endpoint (for example via an adapter) or point the app to any other OpenAI-compatible base URL.

---

## 2. Install dependencies (local dev)

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -e .[dev]
```

This installs the app in editable mode along with dev tools.

---

## 3. Run the app locally

### 3.1 First-time setup

The app will create a default configuration on first run at `./data/config.json` using a local Ollama-style endpoint:

- LLM base URL: `http://localhost:11434/v1`
- LLM model: `llama3`
- Embedding base URL: `http://localhost:11434/v1`
- Embedding model: `nomic-embed-text`

You can change these later via the config API or by editing the JSON file.

### 3.2 Start the FastAPI server

```bash
python -m backend.server
```

By default the server listens on `http://0.0.0.0:8000`.

### 3.3 Open the UI

Navigate to:

- `http://localhost:8000/` for the chat UI (once static mounting is wired in), or
- `http://localhost:8000/docs` for the FastAPI interactive API docs.

The UI provides:

- Chat area with streaming responses from `/api/chat/stream`
- File upload via `/api/documents/upload`

---

## 4. Configuration

The configuration model is defined in [`backend/config/models.py:AppConfig`](backend/config/models.py:1) and persisted via [`backend/config/store.py`](backend/config/store.py:1).

- Default path: `./data/config.json`
- Structure:
  - `llm`: provider name, base URL, API key, model
  - `embeddings`: provider name, base URL, API key, model
  - `vector_store_path`: local directory for vector DB
  - `file_store_path`: local directory for uploaded files
  - `sqlite_path`: path to SQLite metadata DB
  - `max_chunk_size`, `chunk_overlap`: RAG chunking parameters

You can edit `config.json` directly or expose a settings UI that calls:

- `GET /api/config/` to read current config
- `PUT /api/config/` to update config

---

## 5. Running with Docker

### 5.1 Build the image

From the repo root:

```bash
docker build -t local-rag-app .
```

### 5.2 Run the container

```bash
docker run \
  --name local-rag-app \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  local-rag-app
```

Notes:

- The `data` volume persists configuration, uploaded files, and (later) vector store and SQLite DB.
- The app listens on `0.0.0.0:8000` inside the container.

If your LLM/embeddings server is running on the host (e.g. Ollama), ensure the container can reach it. On macOS/Linux Docker Desktop, `host.docker.internal` is often available; you can set the `base_url` in `config.json` accordingly, e.g. `http://host.docker.internal:11434/v1`.

---

## 6. Using Tailscale

This app is designed to work well with Tailscale in a local-first setup.

### 6.1 Host-level Tailscale (recommended for laptops)

1. Install and log into Tailscale on your host machine.
2. Run the app locally (either directly with Python or via Docker).
3. Use Tailscale to expose `localhost:8000` to your tailnet, for example with `tailscale serve` or `tailscale funnel` (depending on your plan and security requirements).

In this model, the app only needs to bind to `0.0.0.0:8000`; Tailscale handles secure network access.

### 6.2 Tailscale sidecar (Docker Compose)

Alternatively, you can run Tailscale as a sidecar container that forwards traffic to the app container. A typical pattern is:

- App container: runs this FastAPI app on an internal Docker network.
- Tailscale container: joins your tailnet and forwards a tailnet port to the app container.

This keeps the app isolated while still reachable over Tailscale.

---

## 7. Project layout

- [`backend/`](backend/__init__.py:1) – FastAPI app, config, LLM clients, routes
- [`frontend/`](frontend/__init__.py:1) – HTML template, CSS theme, JS chat logic
- [`Dockerfile`](Dockerfile:1) – container image definition
- [`pyproject.toml`](pyproject.toml:1) – project metadata and dependencies
- [`README.md`](README.md:1) – this documentation

This documentation covers running the repo locally with Python, in Docker, and in a Tailscale-secured environment.