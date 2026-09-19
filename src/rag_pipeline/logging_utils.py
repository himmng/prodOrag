"""Shared JSON log formatter + request-id context, used by config.py (CLI/import-time
setup) and api/middleware/logging.py (server startup) so every log line — from the
first import to the last request — has the same structured shape.
"""

from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from datetime import datetime, timezone

# Set per-request by RequestLoggingMiddleware; "-" outside a request (CLI runs,
# startup, background jobs).
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

ENVIRONMENT = os.environ.get("APP_ENV", "development")


class JSONLogFormatter(logging.Formatter):
    """One JSON object per log line — console and file both use this."""

    def format(self, record: logging.LogRecord) -> str:
        out: dict = {
            "ts":       datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level":    record.levelname,
            "logger":   record.name,
            "msg":      record.getMessage(),
            "module":   record.module,
            "func":     record.funcName,
            "line":     record.lineno,
            "pid":      record.process,
            "thread":   record.threadName,
            "env":      ENVIRONMENT,
            "req_id":   request_id_var.get(),
        }
        if hasattr(record, "extra_fields"):
            out.update(record.extra_fields)  # type: ignore[arg-type]
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


# Third-party libraries that log at INFO/DEBUG far more than anyone wants to see —
# quieted everywhere (CLI and API), not just after the API's install_json_logging().
NOISY_LOGGERS = ("urllib3", "httpx", "httpcore", "chromadb.telemetry")


def quiet_noisy_loggers(level: str = "WARNING") -> None:
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level)
