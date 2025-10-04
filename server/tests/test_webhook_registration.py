import os
import json
from unittest import mock

import requests


def test_register_meta_posts_callback(monkeypatch):
    from server import check_and_register_webhook as reg

    monkeypatch.setenv("WHATSAPP_PROVIDER", "meta")
    monkeypatch.setenv("WEBHOOK_VERIFY_TOKEN", "test-token")
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "x")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "123")
    monkeypatch.setenv("NGROK_DOMAIN", "unit.test.ngrok.app")

    calls = {}

    class Resp:
        def __init__(self, status_code=200, data=None):
            self.status_code = status_code
            self._data = data or {"ok": True}

        def json(self):
            return self._data

        @property
        def ok(self):
            return 200 <= self.status_code < 300

        @property
        def text(self):
            return json.dumps(self._data)

    def fake_get(url, *a, **k):
        calls["get"] = url
        return Resp(200, {"data": []})

    def fake_post(url, *a, **k):
        calls["post"] = url
        return Resp(200, {"success": True})

    with mock.patch.object(requests, "get", side_effect=fake_get), mock.patch.object(requests, "post", side_effect=fake_post):
        rc = reg.main()
        assert rc == 0
        assert "webhooks" in calls["post"]
