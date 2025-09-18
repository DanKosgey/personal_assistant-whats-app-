#!/usr/bin/env python3
"""
check_and_register_webhook.py — improved, safer webhook registration script.

What it does (summary):
- Loads environment variables from repo/.env, agent/.env, backend/.env (non-destructive)
- Discovers ngrok public URL from the local ngrok API (http://127.0.0.1:4040)
- Queries the Graph API for the phone-number webhook edge to see if it's supported
- If phone-number /webhooks is unsupported (error code 2500) it will attempt an app-level subscription
  using the App ID and app-level access token (APP_ID|APP_SECRET) if available
- If phone-number webhooks are supported, it will try to POST the callback to the phone-number edge
- Avoids printing raw secrets/tokens to logs; prints helpful debugging info and fbtrace_ids when available
- Exits with clear codes for common failure modes
"""

import os
import sys
import json
import time
import logging
import requests
import hashlib
import hmac
from dotenv import dotenv_values, load_dotenv
from typing import Optional

# --- Basic logging setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("webhook_reg")

# --- Paths and .env loading ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Repo root assumed 3 levels up from backend/check_and_register_webhook.py (adjusted from original)
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", "..", ".."))
AGENT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Load .env files in a safe order (do not override existing environment variables)
for p in (os.path.join(REPO_ROOT, ".env"), os.path.join(AGENT_DIR, ".env"), os.path.join(BASE_DIR, ".env")):
    try:
        load_dotenv(dotenv_path=p, override=False)
    except Exception:
        # ignore failures to load optional env files
        pass

# Also parse backend/.env into a dict for explicit fallback values
backend_env_path = os.path.join(BASE_DIR, ".env")
cfg = dotenv_values(backend_env_path) if os.path.exists(backend_env_path) else {}

# --- Configuration from env (prefer environment variables over .env file values) ---
def _env_or_cfg(key: str) -> Optional[str]:
    return os.getenv(key) or cfg.get(key)

WHATSAPP_ACCESS_TOKEN = _env_or_cfg("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = _env_or_cfg("WHATSAPP_PHONE_NUMBER_ID")
WEBHOOK_VERIFY_TOKEN = _env_or_cfg("WEBHOOK_VERIFY_TOKEN") or "whatsapp_webhook_2025"
APP_ID = _env_or_cfg("APP_ID")
APP_SECRET = _env_or_cfg("APP_SECRET")
APP_ACCESS_TOKEN = _env_or_cfg("APP_ACCESS_TOKEN")  # optional; may be APP_ID|APP_SECRET

# If APP_ID & APP_SECRET present and no APP_ACCESS_TOKEN, construct one (app_id|app_secret)
if APP_ID and APP_SECRET and (not APP_ACCESS_TOKEN or APP_ACCESS_TOKEN.strip() == ""):
    APP_ACCESS_TOKEN = f"{APP_ID}|{APP_SECRET}"

# --- Utility to compute appsecret_proof (if both WhatsApp token and app secret present) ---
APPSECRET_PROOF: Optional[str] = None
if WHATSAPP_ACCESS_TOKEN and APP_SECRET:
    try:
        APPSECRET_PROOF = hmac.new(
            APP_SECRET.encode("utf-8"),
            WHATSAPP_ACCESS_TOKEN.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        # Do not log secrets — only indicate that proof was generated
        logger.info("appsecret_proof generated (value hidden).")
    except Exception as e:
        logger.warning("Failed to compute appsecret_proof: %s", e)
        APPSECRET_PROOF = None

logger.info("--- Local env summary (sensitive values redacted) ---")
logger.info("WHATSAPP_PHONE_NUMBER_ID: %s", WHATSAPP_PHONE_NUMBER_ID or "<missing>")
logger.info("WEBHOOK_VERIFY_TOKEN: %s", "<set>" if WEBHOOK_VERIFY_TOKEN else "<missing>")
logger.info("APP_ID: %s", APP_ID or "<missing>")
logger.info("APP_ACCESS_TOKEN: %s", "<set>" if APP_ACCESS_TOKEN else "<missing>")

# --- Discover ngrok public URL ---
ngrok_api = "http://127.0.0.1:4040/api/tunnels"
ngrok_url: Optional[str] = None
try:
    r = requests.get(ngrok_api, timeout=5)
    if r.status_code == 200:
        data = r.json()
        tunnels = data.get("tunnels", [])
        # prefer an https public_url
        for t in tunnels:
            pu = t.get("public_url")
            if pu and pu.startswith("https"):
                ngrok_url = pu
                break
        if not ngrok_url and tunnels:
            ngrok_url = tunnels[0].get("public_url")
    else:
        logger.warning("ngrok API returned status %s", r.status_code)
except Exception as e:
    logger.debug("ngrok API unavailable or error: %s", e)

logger.info("ngrok public URL: %s", ngrok_url or "<none>")

# --- Basic validation of required WhatsApp credentials ---
if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
    logger.error("Missing WHATSAPP_ACCESS_TOKEN or WHATSAPP_PHONE_NUMBER_ID. Please update backend/.env or environment.")
    sys.exit(2)

# Short helper to pretty print FB Graph responses while hiding tokens
def _print_fb_response(resp: requests.Response, prefix: str = ""):
    try:
        body = resp.json()
        # remove any access_token fields if present in echoed responses
        if isinstance(body, dict):
            if "access_token" in body:
                body["access_token"] = "<redacted>"
            if "appsecret_proof" in body:
                body["appsecret_proof"] = "<redacted>"
        logger.info("%sHTTP %s: %s", prefix, resp.status_code, json.dumps(body, indent=2))
    except Exception:
        logger.info("%sHTTP %s: %s", prefix, resp.status_code, resp.text[:1000])

# --- Function to check whether current WhatsApp access token is valid (uses debug_token if possible) ---
def check_token_validity() -> bool:
    # If no APP_ACCESS_TOKEN, we cannot call debug_token reliably; warn but don't fail here.
    if not APP_ACCESS_TOKEN:
        logger.warning("APP_ACCESS_TOKEN not available (APP_ID|APP_SECRET). Skipping debug_token check.")
        return True

    debug_url = "https://graph.facebook.com/debug_token"
    params = {
        "input_token": WHATSAPP_ACCESS_TOKEN,
        "access_token": APP_ACCESS_TOKEN,
    }
    try:
        resp = requests.get(debug_url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            data_err = data.get("data", {})
            is_valid = data_err.get("is_valid", False)
            if is_valid:
                logger.info("Access token appears valid according to debug_token.")
                return True
            else:
                logger.error("debug_token reports token invalid: %s", json.dumps(data_err))
                return False
        else:
            # If debug_token returned 400/401, treat as invalid
            _print_fb_response(resp, prefix="debug_token: ")
            if resp.status_code in (400, 401):
                return False
            return True
    except Exception as e:
        logger.warning("debug_token check failed (network/exception): %s. Proceeding cautiously.", e)
        return True

# Run token check and abort on clear invalid
if not check_token_validity():
    logger.error("ERROR: Graph API reported the access token is invalid or expired.")
    logger.error("Steps to refresh your WhatsApp access token:")
    logger.error(" 1) Visit your Meta/WhatsApp developer dashboard and generate a new token or re-authorize your app.")
    logger.error(" 2) Update WHATSAPP_ACCESS_TOKEN in backend/.env or environment.")
    logger.error(" 3) Re-run this script after updating the token.")
    sys.exit(3)

# --- Query phone-number webhook config to determine supported registration path ---
graph_url_base = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}"
logger.info("--- Querying Graph API for current webhook config (phone-number level) ---")
phone_webhooks_supported = True
last_graph_response: Optional[requests.Response] = None
try:
    params = {}
    if APPSECRET_PROOF:
        params["appsecret_proof"] = APPSECRET_PROOF
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}
    resp = requests.get(f"{graph_url_base}/webhooks", headers=headers, params=params if params else None, timeout=10)
    last_graph_response = resp
    _print_fb_response(resp, prefix="GET /webhooks: ")
    # Inspect error for 2500 path unsupported
    if resp.status_code == 400:
        try:
            b = resp.json()
            err = b.get("error", {})
            if err.get("code") == 2500:
                logger.info("Detected Graph API code 2500 — phone-number /webhooks edge not supported.")
                phone_webhooks_supported = False
        except Exception:
            pass
    elif resp.status_code == 401:
        logger.error("Graph API returned 401 when querying phone-number webhooks.")
        sys.exit(3)
except Exception as e:
    logger.warning("Error querying phone-number webhooks: %s", e)

# --- If phone-number edge unsupported, try app-level subscription (if possible) ---
if not phone_webhooks_supported:
    if APP_ID and APP_ACCESS_TOKEN:
        logger.info("Attempting app-level webhook subscription using APP_ID...")
        app_sub_url = f"https://graph.facebook.com/v18.0/{APP_ID}/subscriptions"
        callback_url = (ngrok_url.rstrip("/") + "/api/webhook") if ngrok_url else None
        if not callback_url:
            logger.error("No ngrok public URL available; cannot register app-level subscription callback_url.")
        else:
            params = {"access_token": APP_ACCESS_TOKEN}
            # payload fields for app subscription
            data = {
                "object": "whatsapp_business_account",
                "callback_url": callback_url,
                "verify_token": WEBHOOK_VERIFY_TOKEN,
                # 'fields' may include other events as needed; start with messages
                "fields": "messages",
            }
            try:
                # Post as form-encoded per Graph API subscription docs
                resp = requests.post(app_sub_url, params=params, data=data, timeout=15)
                _print_fb_response(resp, prefix="APP SUBSCRIBE POST: ")
                if resp.status_code in (200, 201):
                    logger.info("App-level subscription request succeeded.")
                else:
                    logger.warning("App-level subscription did not succeed; check App dashboard and callback response.")
                    # If verification failed, Graph tells us the callback verification failed (400)
                    if resp.status_code == 400:
                        logger.warning("If you see a 'Callback verification failed' error, ensure your callback URL responds to GET verification handshake (hub.challenge) and returns the challenge as plain text.")
            except Exception as e:
                logger.error("App-level subscription error: %s", e)
    else:
        logger.info("No APP_ID or app access token available to attempt app-level subscription programmatically.")
        logger.info("Please set APP_ID and APP_ACCESS_TOKEN (or APP_SECRET) and retry, or configure the callback URL in the Facebook App Dashboard.")
else:
    # --- Phone-number webhooks appear supported; attempt to register callback at phone-number level ---
    if ngrok_url:
        callback = ngrok_url.rstrip("/") + "/api/webhook"
        logger.info("Attempting to register phone-number webhook callback: %s", callback)
        payload = {
            "messaging_product": "whatsapp",
            "webhooks": {
                "url": callback,
                "events": ["messages"]
            }
        }
        try:
            headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}", "Content-Type": "application/json"}
            params = {}
            if APPSECRET_PROOF:
                params["appsecret_proof"] = APPSECRET_PROOF
            resp = requests.post(f"{graph_url_base}/webhooks", headers=headers, params=params if params else None, json=payload, timeout=15)
            _print_fb_response(resp, prefix="POST /webhooks: ")
            if resp.status_code in (200, 201):
                logger.info("Phone-number webhook registration succeeded (or returned success code).")
            else:
                logger.warning("Phone-number webhook registration failed or returned non-success. Check the callback URL and verify_token handling.")
                if resp.status_code == 400:
                    logger.warning("If Graph API returns 'Callback verification failed', ensure your callback responds to the GET verification handshake and returns the hub.challenge (plain text).")
                if resp.status_code == 401:
                    logger.error("Unauthorized: access token invalid or expired.")
        except Exception as e:
            logger.error("Graph API POST error when registering phone-number webhook: %s", e)
    else:
        logger.info("No ngrok URL found; skipping phone-number webhook registration. Manually set callback in App Dashboard if needed.")

logger.info("Done.")
