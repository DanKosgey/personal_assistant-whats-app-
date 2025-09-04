from fastapi.testclient import TestClient
import importlib
import os
import sys

os.environ.setdefault("DEV_SMOKE", "1")
os.environ.setdefault("DISABLE_WHATSAPP_SENDS", "1")
os.environ.setdefault("USE_INMEMORY_CACHE", "1")

APP_IMPORT_PATHS = [
    "agent.whats-app-agent.server.server",
    "agent.whats-app-agent.server",
    "server.server",
    "server"
]

app = None
tried = []
for p in APP_IMPORT_PATHS:
    try:
        mod = importlib.import_module(p)
        app = getattr(mod, "app", None) or (getattr(mod, "create_app", None)() if hasattr(mod, "create_app") else None)
        if app:
            print(f"Loaded app from {p}")
            break
    except Exception as e:
        tried.append((p, str(e)))
        continue

if app is None:
    print("Could not import FastAPI app. Please adjust APP_IMPORT_PATHS.", file=sys.stderr)
    for p, e in tried:
        print(f"Tried {p}: {e}")
    sys.exit(2)

client = TestClient(app)

r = client.get("/health")
print("/health ->", r.status_code, r.text)

webhook_payload = {"messages": [{"from": "254700000000", "text": "hello dev smoke", "id": "smoke-1"}]}
r = client.post("/api/webhook", json=webhook_payload)
print("/api/webhook ->", r.status_code, r.text)

if r.status_code >= 500:
    sys.exit(3)
print("dev_smoke checks passed")
