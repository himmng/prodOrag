"""Request-scoped logging for the FastAPI app.

- Generates a request_id per request (or uses incoming X-Request-ID)
- Propagates it via contextvar so every log line emitted during the request
  (access log, retrieval, generation, ...) carries it automatically
- Emits one JSON envelope per request (method, path, status, latency_ms)
- Widens the same JSON format (rag_pipeline.logging_utils) to the root logger,
  so uvicorn/fastapi/third-party logs match the app's own log shape
  (call install_json_logging() once at startup).
"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from rag_pipeline.logging_utils import JSONLogFormatter, quiet_noisy_loggers, request_id_var


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Stamps a request_id, times the request, emits one JSON envelope."""

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(req_id)
        start = time.perf_counter()

        access_log = logging.getLogger("rag.api.access")
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            access_log.exception(
                "request.failed",
                extra={"extra_fields": {
                    "method":     request.method,
                    "path":       request.url.path,
                    "status":     500,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 1),
                }},
            )
            request_id_var.reset(token)
            raise

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        access_log.info(
            "request",
            extra={"extra_fields": {
                "method":     request.method,
                "path":       request.url.path,
                "status":     status,
                "latency_ms": latency_ms,
            }},
        )
        response.headers["x-request-id"] = req_id
        request_id_var.reset(token)
        return response


def install_json_logging(level: str = "INFO") -> None:
    """Route every logger (root, so uvicorn/fastapi/etc. included) through the same
    JSON formatter and the same per-session log file config.py opened. Call once at
    startup, before the FastAPI app is constructed.
    """
    from logging.handlers import RotatingFileHandler
    from rag_pipeline.config import SESSION_LOG_PATH

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    fmt = JSONLogFormatter()

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    # Keep writing to the same per-session file config.py opened.
    if SESSION_LOG_PATH is not None:
        file_handler = RotatingFileHandler(
            SESSION_LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    root.setLevel(level.upper())

    # "rag" now propagates to root instead of using its own console+file handlers,
    # so every logger in the process shares one format and one file.
    rag_log = logging.getLogger("rag")
    for h in rag_log.handlers[:]:
        rag_log.removeHandler(h)
    rag_log.propagate = True

    quiet_noisy_loggers()
