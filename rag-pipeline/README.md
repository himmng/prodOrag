[![CI](https://github.com/himmng/protorag/rag-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/protorag/rag-pipeline/actions/workflows/ci.yml)

## Pre-merge quality check

Before merging changes that touch retrieval, prompts, or eval:

1. Start the local stack: `uvicorn rag_pipeline.api.main:app`
2. Run baseline retrieval eval:
