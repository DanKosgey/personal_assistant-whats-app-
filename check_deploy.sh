#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8001}
URL=${URL:-http://localhost:${PORT}/healthz}

status=$(curl -s -o /dev/null -w "%{http_code}" "$URL")
if [[ "$status" == "200" ]]; then
  echo "Healthcheck OK at $URL"
  exit 0
else
  echo "Healthcheck FAILED ($status) at $URL" >&2
  exit 1
fi
