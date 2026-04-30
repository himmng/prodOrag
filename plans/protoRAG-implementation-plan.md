# protoRAG Implementation Plan

This plan breaks down protoRAG into concrete implementation steps and files so another mode can implement it directly.

## 1. Project Skeleton

Create the following structure:

- `backend/`
  - `main.py` [`backend/main.py`](backend/main.py:1)
  - `api/`
    - `routes.py` [`backend/api/routes.py`](backend/api/routes.py:1)
  - `core/`
    - `config.py` [`backend/core/config.py`](backend/core/config.py:1)
    - `llm_client.py` [`backend/core/llm_client.py`](backend/core/llm_client.py:1)
    - `embedding_client.py` [`backend/core/embedding_client.py`](backend/core/embedding_client.py:1)
    - `vector_store.py` [`backend/core/vector_store.py`](backend/core/vector_store.py:1)
    - `document_store.py` [`backend/core/document_store.py`](backend/core/document_store.py:1)
    - `chunking.py` [`backend/core/chunking.py`](backend/core/chunking.py:1)
    - `orchestrator.py` [`backend/core/orchestrator.py`](backend/core/orchestrator.py:1)
  - `templates/`
    - `index.html` [`backend/templates/index.html`](backend/templates/index.html:1)
  - `static/`
    - `css/styles.css` [`backend/static/css/styles.css`](backend/static/css/styles.css:1)
    - `js/app.js` [`backend/static/js/app.js`](backend/static/js/app.js:1)
- `config/config.yaml` [`config/config.yaml`](config/config.yaml:1)
- `data/` (empty, gitignored)
- `README.md` [`README.md`](README.md:1)
- `requirements.txt` [`requirements.txt`](requirements.txt:1)

## 2. Dependencies

Populate `requirements.txt` [`requirements.txt`](requirements.txt:1) with:

- `fastapi`
- `uvicorn`
- `jinja2`
- `pydantic`
- `pyyaml`
- `chromadb`
- `python-multipart` (for file uploads)
- `httpx` or `requests` (for calling LLM and embedding endpoints)

## 3. Config Module

Implement `core/config.py` [`backend/core/config.py`](backend/core/config.py:1):

- Define `AppConfig` Pydantic model:
  - `llm` with fields `provider`, `base_url`, `api_key`, `model`
  - `embeddings` with fields `provider`, `base_url`, `api_key`, `model`
  - `storage` with fields `vector_dir`, `docs_dir`
  - `rag` with fields `top_k`, `chunk_size`, `chunk_overlap`
- Functions:
  - `load_config(path: str = "config/config.yaml") -> AppConfig`
  - `save_config(config: AppConfig, path: str = "config/config.yaml")`
- Provide a global accessor or dependency injection function for FastAPI:
  - `get_config() -> AppConfig`

## 4. LLM Client

Implement `core/llm_client.py` [`backend/core/llm_client.py`](backend/core/llm_client.py:1):

- Define abstract base class `LLMClient` with method:
  - `async def chat(self, messages: list[dict]) -> str`
- Implement `OllamaLLMClient`:
  - Use OpenAI-compatible `/v1/chat/completions` endpoint
  - Build headers and JSON body from config
- Implement `LMStudioLLMClient` similarly
- Implement factory function:
  - `def create_llm_client(config: AppConfig) -> LLMClient`

## 5. Embedding Client

Implement `core/embedding_client.py` [`backend/core/embedding_client.py`](backend/core/embedding_client.py:1):

- Define abstract base class `EmbeddingClient` with method:
  - `async def embed(self, texts: list[str]) -> list[list[float]]`
- Implement `OllamaEmbeddingClient` and `LMStudioEmbeddingClient` using `/v1/embeddings`
- Implement factory function:
  - `def create_embedding_client(config: AppConfig) -> EmbeddingClient`

## 6. Vector Store

Implement `core/vector_store.py` [`backend/core/vector_store.py`](backend/core/vector_store.py:1):

- Initialize Chroma client with `persist_directory=config.storage.vector_dir`
- Create or load collection `protoRAG_docs`
- Methods:
  - `add_documents(doc_id: str, embeddings: list[list[float]], metadatas: list[dict], ids: list[str])`
  - `query(embedding: list[float], top_k: int) -> list[dict]` returning documents and metadata

## 7. Document Store and Chunking

Implement `core/document_store.py` [`backend/core/document_store.py`](backend/core/document_store.py:1):

- Save uploaded files to `config.storage.docs_dir`
- Generate `doc_id` and store metadata in a JSON index file

Implement `core/chunking.py` [`backend/core/chunking.py`](backend/core/chunking.py:1):

- Function `chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]`
- Simple character or word-based splitter is sufficient for v1

## 8. Orchestrator

Implement `core/orchestrator.py` [`backend/core/orchestrator.py`](backend/core/orchestrator.py:1):

- Class `RAGOrchestrator` with dependencies:
  - `config: AppConfig`
  - `llm_client: LLMClient`
  - `embedding_client: EmbeddingClient`
  - `vector_store: VectorStore`
- Method `async def chat(self, user_message: str, history: list[dict]) -> dict`:
  - Compute embedding for `user_message`
  - Query vector store for top-k chunks
  - Build prompt with context and history
  - Call LLM client and return response plus source metadata

## 9. FastAPI App and Routes

Implement `backend/api/routes.py` [`backend/api/routes.py`](backend/api/routes.py:1):

- Define Pydantic models for requests and responses:
  - `ChatRequest` with `message`, `history`, `conversation_id`
  - `ChatResponse` with `answer`, `sources`
- Routes:
  - `POST /api/chat`:
    - Use orchestrator to handle chat
  - `POST /api/documents`:
    - Accept file upload
    - Save file, read text, chunk, embed, and add to vector store
  - `GET /api/config` and `POST /api/config`:
    - Use config module to load and save config
  - `GET /health`:
    - Return simple status

Implement `backend/main.py` [`backend/main.py`](backend/main.py:1):

- Create FastAPI app
- Include router from `api/routes.py`
- Mount static files and templates
- Define root route `/` to render `index.html`

## 10. UI Implementation

Implement `backend/templates/index.html` [`backend/templates/index.html`](backend/templates/index.html:1):

- HTML structure mimicking ChatGPT layout:
  - Sidebar with app name and new chat button
  - Main chat area with messages container
  - Input area with textarea and send button
  - Settings panel for config

Implement `backend/static/js/app.js` [`backend/static/js/app.js`](backend/static/js/app.js:1):

- Functions:
  - `sendMessage()` to call `/api/chat`
  - `appendMessage(role, content)` to update UI
  - `uploadDocument()` to call `/api/documents`
  - `loadConfig()` and `saveConfig()` for config endpoints

Implement `backend/static/css/styles.css` [`backend/static/css/styles.css`](backend/static/css/styles.css:1):

- Styles to closely match ChatGPT dark theme and layout

## 11. README and Usage

Implement `README.md` [`README.md`](README.md:1):

- Describe protoRAG
- Installation steps
- How to configure Ollama and LM Studio endpoints
- How to run FastAPI app with `uvicorn`

## 12. Next Mode

Implementation should be done in Code mode using this plan and the architecture document [`plans/protoRAG-architecture-plan.md`](plans/protoRAG-architecture-plan.md:1) as references.