"""Unit tests for the API key dependency (no FastAPI needed)."""

import pytest
from fastapi import HTTPException

# Import after env is set so module-level _VALID_KEYS picks them up
from rag_pipeline.api.middleware import auth as _auth_mod
_auth_mod._VALID_KEYS = {"test-key-1", "test-key-2"}

# Force re-import to pick up env
import importlib


def test_valid_key_passes():
    assert _auth_mod.verify_api_key(x_api_key="test-key-1") == "test-key-1"


def test_missing_key_raises_401():
    with pytest.raises(HTTPException) as exc:
        _auth_mod.verify_api_key(x_api_key=None)
    assert exc.value.status_code == 401


def test_invalid_key_raises_403():
    with pytest.raises(HTTPException) as exc:
        _auth_mod.verify_api_key(x_api_key="bogus")
    assert exc.value.status_code == 403