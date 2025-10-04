from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx


class NgrokClient:
    def __init__(self, api_url: Optional[str] = None, public_domain: Optional[str] = None):
        # Compose default API URL based on environment
        # In Docker Compose, set NGROK_API_URL=http://ngrok:4040
        # On host/systemd, default to http://127.0.0.1:4040
        self.api_url = api_url or os.getenv("NGROK_API_URL") or "http://127.0.0.1:4040"
        self.public_domain = public_domain or os.getenv("NGROK_DOMAIN")
        self.protocol = (os.getenv("NGROK_PROTOCOL") or "https").lower()

    async def get_tunnels(self) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.api_url.rstrip('/')}/api/tunnels")
                resp.raise_for_status()
                return resp.json()
        except Exception:
            return {"error": "unavailable"}

    async def get_public_url(self) -> Optional[str]:
        # Prefer reserved domain if configured
        if self.public_domain:
            return f"{self.protocol}://{self.public_domain}"
        # Else inspect tunnels for https url
        data = await self.get_tunnels()
        tunnels = data.get("tunnels", []) if isinstance(data, dict) else []
        for t in tunnels:
            pu = t.get("public_url")
            if pu and pu.startswith("https"):
                return pu
        for t in tunnels:
            pu = t.get("public_url")
            if pu:
                return pu
        return None

    async def is_public_health_ok(self) -> bool:
        url = await self.get_public_url()
        if not url:
            return False
        health = f"{url.rstrip('/')}/healthz"
        try:
            async with httpx.AsyncClient(timeout=5.0, verify=True) as client:
                r = await client.get(health)
                return r.status_code == 200
        except Exception:
            return False
