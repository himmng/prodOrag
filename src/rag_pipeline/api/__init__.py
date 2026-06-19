"""FastAPI service layer for the RAG pipeline.

Run locally:
    uvicorn rag_pipeline.api.main:app --reload --host 0.0.0.0 --port 8000

Or for production-shaped (no reload, multiple workers):
    uvicorn rag_pipeline.api.main:app --host 0.0.0.0 --port 8000 --workers 1
"""