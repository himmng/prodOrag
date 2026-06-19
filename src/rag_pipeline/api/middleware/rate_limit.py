"""Per-key rate limiting using slowapi.

Limits the same identity (API key if present, else client IP). Routes opt
in via the @limiter.limit("N/period") decorator.
"""

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def _identity(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(
    key_func=_identity,
    default_limits=["120/minute"],  # global ceiling
    headers_enabled=True,            # adds X-RateLimit-* response headers
)