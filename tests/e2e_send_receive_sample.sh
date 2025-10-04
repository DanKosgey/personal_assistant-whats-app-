#!/usr/bin/env bash
set -euo pipefail

PUBLIC=${PUBLIC_WEBHOOK_URL:-https://${NGROK_DOMAIN}}

echo "Testing health at $PUBLIC/healthz"
curl -fsS "$PUBLIC/healthz" >/dev/null
echo "OK"

echo "Posting sample webhook"
curl -fsS -X POST "$PUBLIC/api/webhook" \
  -H 'Content-Type: application/json' \
  -d '{"entry":[{"changes":[{"value":{"messages":[{"from":"+100000000","type":"text","text":{"body":"hello"}}]}}]}]}'
