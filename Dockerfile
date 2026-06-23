# syntax=docker/dockerfile:1.7

# ── Stage 1: builder ──────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only what dep resolution needs (caching)
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --upgrade pip && \
    pip install .

# ── Stage 2: runtime ──────────────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    ANONYMIZED_TELEMETRY=FALSE

# Non-root user
# Non-root user — UID/GID matched to host so bind mounts work
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd -g ${USER_GID} app \
 && useradd -u ${USER_UID} -g ${USER_GID} -m -d /home/app app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src/         ./src/
COPY --chown=app:app pyproject.toml ./

RUN mkdir -p /app/chroma_db /app/data /app/eval /home/app/.cache/huggingface \
 && chown -R app:app /app /home/app

USER app

EXPOSE 8000

# Smoke test via /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)" \
        || exit 1

CMD ["uvicorn", "rag_pipeline.api.main:app", "--host", "0.0.0.0", "--port", "8000"]