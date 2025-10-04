#!/usr/bin/env bash
set -Eeuo pipefail

# Allow optional wait for dependencies (e.g., DB) if needed
: "${PORT:=8001}"
: "${WORKERS:=4}"

MODE=${1:-start-agent-only}
echo "[entrypoint] MODE=$MODE ENV=${ENV:-unset} PORT=$PORT WORKERS=$WORKERS"

# Run application
if [[ "$MODE" == "start-with-ngrok" ]]; then
  # Wait for ngrok to be healthy if running as a sidecar (Compose)
  : "${NGROK_DOMAIN:?NGROK_DOMAIN required}"
  : "${NGROK_PROTOCOL:=https}"
  : "${PUBLIC_WEBHOOK_URL:=${NGROK_PROTOCOL}://${NGROK_DOMAIN}}"
  echo "[entrypoint] Waiting for ngrok at $PUBLIC_WEBHOOK_URL/healthz"
  for i in {1..60}; do
    if curl -fsS "$PUBLIC_WEBHOOK_URL/healthz" >/dev/null 2>&1; then
      echo "[entrypoint] ngrok reachable"
      break
    fi
    sleep 2
    if [[ $i -eq 60 ]]; then
      echo "[entrypoint] ngrok did not become reachable in time" >&2
    fi
  done
fi

exec python -m uvicorn server.server:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers "$WORKERS" \
    --proxy-headers \
    --forwarded-allow-ips="*"
