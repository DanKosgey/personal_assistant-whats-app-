import os
import time
import pytest

from server.cache import cache_manager
from server.config import get_ai_keys
from server.db import db_manager
from server.server import app


@pytest.mark.asyncio
async def test_cache_expiry_enforced():
    key = "t:exp"
    await cache_manager.set(key, "v", expire=1)
    val = await cache_manager.get(key)
    assert val == "v"
    time.sleep(1.2)
    val2 = await cache_manager.get(key)
    assert val2 is None


def test_config_keys_unification(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "a,b , c")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    keys = get_ai_keys()
    assert keys == ["a", "b", "c"]
    monkeypatch.setenv("GEMINI_API_KEY", "a")
    keys2 = get_ai_keys()
    assert set(keys2) >= {"a", "b", "c"}


def test_db_manager_get_collection_method_present():
    m = db_manager
    assert hasattr(m, "get_collection")


def test_route_reuse_clients_smoke():
    from starlette.testclient import TestClient
    with TestClient(app) as client:
        resp = client.post("/api/messages", json={"from": "+100", "text": "hello"})
        assert resp.status_code in (200, 400, 422)