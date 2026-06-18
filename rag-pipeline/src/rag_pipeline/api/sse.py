"""Server-Sent Events formatting helpers."""

import json
from typing import Any


def sse_event(event: str, data: dict[str, Any]) -> str:
    """Serialize a dict as one SSE frame. Format: 'event: x\\ndata: {...}\\n\\n'."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"