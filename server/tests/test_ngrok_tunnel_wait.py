import asyncio
from unittest import mock

import httpx

from server.ngrok_client import NgrokClient


async def _fake_tunnels_ok(*a, **k):
    class R:
        def __init__(self):
            self._json = {"tunnels": [{"public_url": "https://unit.test.ngrok.app"}]}

        def raise_for_status(self):
            return None

        def json(self):
            return self._json

    return R()


def test_ngrok_client_get_public_url(monkeypatch):
    async def fake_get(self, url):
        return await _fake_tunnels_ok()

    with mock.patch.object(httpx.AsyncClient, "get", new=fake_get):
        nc = NgrokClient(api_url="http://ngrok:4040")
        url = asyncio.get_event_loop().run_until_complete(nc.get_public_url())
        assert url == "https://unit.test.ngrok.app"
