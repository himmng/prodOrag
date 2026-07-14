# API Reference (code)

Auto-generated from docstrings in `src/rag_pipeline/`. For the *HTTP* API schema
(request/response fields), use the live `/docs` and `/redoc` served by the running
app — see [Usage Guide](USAGE.md#using-the-api).

This page covers the Python-level building blocks: configuration, the FastAPI
routes/schemas, retrievers, parsers, generation, corpus registry, and eval tooling.

## Configuration

::: rag_pipeline.config

## Model providers

::: rag_pipeline.providers

## Vector store

::: rag_pipeline.vectorstore

## API — app & routes

::: rag_pipeline.api.main

::: rag_pipeline.api.schemas

::: rag_pipeline.api.documents

::: rag_pipeline.api.sse

## API — middleware

::: rag_pipeline.api.middleware.auth

::: rag_pipeline.api.middleware.logging

::: rag_pipeline.api.middleware.rate_limit

## Corpus registry

::: rag_pipeline.corpus.registry

::: rag_pipeline.corpus.concordance

## Retrievers

::: rag_pipeline.retrievers.base

::: rag_pipeline.retrievers.dense

::: rag_pipeline.retrievers.bm25

::: rag_pipeline.retrievers.ensemble

::: rag_pipeline.retrievers.reranker

::: rag_pipeline.retrievers.multi_query

## Parsers

::: rag_pipeline.parsers.base

::: rag_pipeline.parsers.statute

::: rag_pipeline.parsers.docling

::: rag_pipeline.parsers.structured

::: rag_pipeline.parsers.cache

## Generation

::: rag_pipeline.generation.pipeline

::: rag_pipeline.generation.context

::: rag_pipeline.generation.llm

## Prompts

::: rag_pipeline.prompts.loader

## Schemas (shared data models)

::: rag_pipeline.schemas

## CLI

::: rag_pipeline.cli.ingest

## Evaluation

::: rag_pipeline.eval.retrieval

::: rag_pipeline.eval.latency

::: rag_pipeline.eval.ragas_runner

::: rag_pipeline.eval.qgen_v3

## Utilities

::: rag_pipeline.utils
