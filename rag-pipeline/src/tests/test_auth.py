"""Unit tests for the API key dependency (no FastAPI needed)."""

import pytest
from fastapi import HTTPException

# Import after env is set so module-level _VALID_KEYS picks them up
import os
os.environ["API_KEYS"] = "dev-key-123"

# Force re-import to pick up env
import importlib
from rag_pipeline.api.middleware import auth as auth_mod
importlib.reload(auth_mod)


def test_valid_key_passes():
    assert auth_mod.verify_api_key(x_api_key="dev-key-123") == "dev-key-123"


def test_missing_key_raises_401():
    with pytest.raises(HTTPException) as exc:
        auth_mod.verify_api_key(x_api_key=None)
    assert exc.value.status_code == 401


def test_invalid_key_raises_403():
    with pytest.raises(HTTPException) as exc:
        auth_mod.verify_api_key(x_api_key="bogus")
    assert exc.value.status_code == 403