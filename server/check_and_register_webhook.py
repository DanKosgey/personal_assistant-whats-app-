#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import logging
from typing import Optional

import requests


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("webhook_reg")


def env_required(name: str) -> str:
    v = os.getenv(name)
    if not v or v.strip() == "":
        raise SystemExit(f"Missing required environment variable: {name}")
    return v


def get_public_webhook_url() -> str:
    url = os.getenv("PUBLIC_WEBHOOK_URL")
    if url:
        return url.rstrip("/")
    domain = env_required("NGROK_DOMAIN")
    proto = os.getenv("NGROK_PROTOCOL", "https")
    return f"{proto}://{domain}"


def register_meta(callback_url: str, verify_token: str) -> bool:
    token = env_required("WHATSAPP_ACCESS_TOKEN")
    phone_number_id = env_required("WHATSAPP_PHONE_NUMBER_ID")
    app_secret = os.getenv("APP_SECRET")
    appsecret_proof = None
    if app_secret:
        import hmac, hashlib

        appsecret_proof = hmac.new(app_secret.encode("utf-8"), token.encode("utf-8"), hashlib.sha256).hexdigest()

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    params = {"appsecret_proof": appsecret_proof} if appsecret_proof else None
    graph_url_base = f"https://graph.facebook.com/v18.0/{phone_number_id}"

    # Get existing
    try:
        r = requests.get(graph_url_base + "/webhooks", headers=headers, params=params, timeout=10)
        logger.info("Meta webhooks GET %s", r.status_code)
    except Exception as e:
        logger.warning("GET webhooks failed: %s", e)

    payload = {
        "messaging_product": "whatsapp",
        "webhooks": {"url": f"{callback_url.rstrip('/')}/api/webhook", "events": ["messages"]},
    }
    try:
        r = requests.post(graph_url_base + "/webhooks", headers=headers, params=params, data=json.dumps(payload), timeout=10)
        logger.info("Meta webhooks POST %s", r.status_code)
        try:
            logger.info("Meta response: %s", r.json())
        except Exception:
            logger.info("Meta text: %s", r.text)
        return r.ok
    except Exception as e:
        logger.error("Meta POST error: %s", e)
        return False


def register_twilio(callback_url: str, verify_token: str) -> bool:
    # Placeholder: Twilio configuration typically happens in console or API
    # Implement if credentials provided; otherwise return True (no-op)
    logger.info("Twilio registration not implemented - please configure in Twilio Console")
    return True


def main() -> int:
    provider = os.getenv("WHATSAPP_PROVIDER", "meta").lower()
    verify_token = env_required("WEBHOOK_VERIFY_TOKEN")
    public_base = get_public_webhook_url()
    logger.info("Using PUBLIC_WEBHOOK_URL=%s", public_base)

    ok = False
    if provider == "meta":
        ok = register_meta(public_base, verify_token)
    elif provider == "twilio":
        ok = register_twilio(public_base, verify_token)
    else:
        logger.error("Unknown WHATSAPP_PROVIDER=%s", provider)
        return 2

    if not ok:
        logger.error("Webhook registration failed")
        return 1
    logger.info("Webhook registration complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
