#!/usr/bin/env bash
set -Eeuo pipefail

# Allow optional wait for dependencies (e.g., DB) if needed
: "${PORT:=8001}"
: "${WORKERS:=4}"

echo "[entrypoint] ENV=${ENV:-unset} PORT=$PORT WORKERS=$WORKERS"

# Run application
exec python -m uvicorn server.server:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --workers "$WORKERS" \
  --proxy-headers \
  --forwarded-allow-ips="*"
