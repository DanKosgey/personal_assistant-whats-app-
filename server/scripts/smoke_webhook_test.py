import importlib.util
import os
import time
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure DEV_SMOKE and DISABLE_WHATSAPP_SENDS for safe local run
os.environ.setdefault("DEV_SMOKE", "1")
os.environ.setdefault("DISABLE_WHATSAPP_SENDS", "1")

# Add the whats-app-agent folder to sys.path so package-relative imports in server.py work
import sys
pkg_root = Path(__file__).parents[2]
sys.path.insert(0, str(pkg_root))

server_mod = importlib.import_module("server.server")
create_app = getattr(server_mod, "create_app")
app = create_app()
client = TestClient(app)

print("GET /health")
r = client.get("/health")
print(r.status_code)
print(r.json())

# Sample minimal WhatsApp webhook payload
payload = {
    "entry": [
        {
            "id": "WHATSAPP_BUSINESS_ACCOUNT_ID",
            "changes": [
                {
                    "value": {
                        "messages": [
                            {
                                "from": "15551234567",
                                "id": "wamid.HBgM",
                                "timestamp": "162022",
                                "type": "text",
                                "text": {"body": "Hello from user via webhook test"}
                            }
                        ]
                    }
                }
            ]
        }
    ]
}

print("POST /api/webhook")
r2 = client.post("/api/webhook", json=payload)
print(r2.status_code)
print(r2.json())

# Wait briefly to allow background tasks to run (TestClient runs background tasks after response)
print("Sleeping to allow background tasks to finish...")
time.sleep(2)
print("Done")
