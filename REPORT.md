# Production Hardening Report

## Overview
- Project: FastAPI WhatsApp AI backend
- Goal: Secure, observable, resilient webhook processing with stable APIs.

## Changes Implemented
- Config
  - Removed environment variable deletion; preserved external secrets.
  - Production validation now accepts `GEMINI_API_KEY` or `GEMINI_API_KEYS` via `config.get_ai_keys()`.
  - .env loading centralized/hardened.
- Performance and Security
  - Optional ORJSON default; gzip/brotli compression; shared httpx client; timing header; metrics flag.
  - Redacted PII in exception logs and webhook logs; minimized payload logging.
- AI Handler
  - Hardened `_call_openrouter` to initialize `data` and avoid masking errors; reduced prompt logging to debug.
- Persistence
  - Replaced silent `except Exception: pass` with contextual logging in cache paths.
- API Routes
  - `messages` route reuses app-scoped resources; no per-request client creation.
- Legacy Exports
  - Gated wildcard export of `server_back_up` to non-production only.

## Tests (next)
- Add unit tests for:
  - DB manager `get_collection` (Motor and pymongo paths)
  - Cache expiry enforcement
  - Config key unification (`get_ai_keys`) single/multi/mixed
  - Route resource reuse (no per-request instantiation)
  - Webhook logging redaction (no token leakage)

## Operational Notes
- Dev mode: `DISABLE_DB=1 DEV_SMOKE=1 ENABLE_METRICS=1`
- Production: provide env via secrets; do not rely on `.env`.

## Follow-ups
- Consolidate requirements and add CI (lint, type-check, tests).
- Optional Docker packaging with health probes.