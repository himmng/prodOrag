"""API key authentication via X-API-Key header.

Configured via API_KEYS env var (comma-separated). Empty = auth disabled
(dev mode). Used as a FastAPI dependency:

    @app.post("/foo", dependencies=[Depends(verify_api_key)])
"""

from typing import Annotated, Optional
from fastapi import Header, HTTPException, status

from rag_pipeline.config import cfg


_VALID_KEYS = {k.strip() for k in (cfg.API_KEYS or "").split(",") if k.strip()}


def auth_enabled() -> bool:
    return bool(_VALID_KEYS)


def verify_api_key(
    x_api_key: Annotated[Optional[str], Header(alias="X-API-Key")] = None,
) -> str:
    """Returns the validated key (or 'anonymous' in dev mode).

    Raises 401 if missing, 403 if invalid.
    """
    if not _VALID_KEYS:
        return "anonymous"

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if x_api_key not in _VALID_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return x_api_key