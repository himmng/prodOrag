"""Structured JSON logging for the FastAPI app.

- Generates a request_id per request (or uses incoming X-Request-ID)
- Propagates it via contextvar so handler logs auto-carry it
- Emits one JSON envelope per request (method, path, status, latency_ms)
- Replaces the root logger's formatter with JSON (call install_json_logging())
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


# Propagated to every log line emitted during a request handler
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JSONFormatter(logging.Formatter):
    """Single-line JSON per log record."""

    def format(self, record: logging.LogRecord) -> str:
        out: dict = {
            "ts":     datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
            "req_id": request_id_var.get(),
        }
        # Merge any custom extra fields
        if hasattr(record, "extra_fields"):
            out.update(record.extra_fields)  # type: ignore[arg-type]
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


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
    """Replace handlers everywhere with a single JSON one. Call once at startup."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Strip any pre-existing handler from "rag" so it propagates to root
    rag_log = logging.getLogger("rag")
    for h in rag_log.handlers[:]:
        rag_log.removeHandler(h)
    rag_log.propagate = True

    # Quiet noisy libs
    for noisy in ("urllib3", "httpx", "httpcore", "chromadb.telemetry"):
        logging.getLogger(noisy).setLevel("WARNING")