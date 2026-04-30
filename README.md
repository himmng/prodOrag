# protoRAG

protoRAG is a minimal RAG prototype using FastAPI, a simple ChatGPT-like UI, local LLM providers (Ollama/LM Studio), and ChromaDB.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/config.yaml` to point to your Ollama or LM Studio endpoints and desired models.

## Running

Make sure you have a local LLM running:
- For Ollama, install it and pull a model, e.g. `ollama pull llama3`
- For LM Studio, start the server with an OpenAI-compatible endpoint

Update `config/config.yaml` with the correct `llm.base_url`, `llm.model`, and embedding settings.

Start the FastAPI app with uvicorn from the project root:

```bash
uvicorn backend.main:app --reload
```

Then open `http://localhost:8000` in your browser.

## Testing

Run the tests with:

```bash
pytest
```
