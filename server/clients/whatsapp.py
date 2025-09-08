from typing import Dict, Any, Optional, List
import os
import logging
import asyncio
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class WhatsAppAPIError(Exception):
    pass


class EnhancedWhatsAppClient:
    """A small WhatsApp client stub with allowlist + retry/backoff logic.

    This client intentionally avoids making real network calls unless the
    environment is configured with WHATSAPP_API_URL and WHATSAPP_ACCESS_TOKEN.
    """

    def __init__(self, access_token: Optional[str] = None, api_url: Optional[str] = None, http_client=None):
        self.access_token = access_token or os.getenv("WHATSAPP_ACCESS_TOKEN")
        self.api_url = api_url or os.getenv("WHATSAPP_API_URL")
        # Phone number id used for sending messages: endpoint is {API_BASE}/{PHONE_NUMBER_ID}/messages
        self.phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
        self.http_client = http_client
        raw_allowed = os.getenv("WHATSAPP_ALLOWED_RECIPIENTS", "")
        # When WHATSAPP_ALLOWED_RECIPIENTS is empty, allow all recipients
        self.allowed: Optional[List[str]] = [s.strip() for s in raw_allowed.split(",") if s.strip()] or None
        self.max_retries = int(os.getenv("WHATSAPP_SEND_MAX_RETRIES", "3"))
        self.backoff_base = float(os.getenv("WHATSAPP_SEND_BACKOFF_BASE", "1.0"))

    def _is_allowed(self, to: str) -> bool:
        if not self.allowed:
            return True
        return to in self.allowed

    async def _do_post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Minimal abstraction: only perform a real POST if api_url and token present.
        if not self.api_url or not self.access_token:
            logger.info("WhatsApp client not configured for real sends; returning stub response")
            await asyncio.sleep(0.01)
            return {"messages": [{"id": "stub-12345", "status": "sent"}]}

        # Perform a real HTTP POST using httpx
        try:
            import httpx

            headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}
            client = self.http_client or httpx.AsyncClient(timeout=10.0)
            _created_here = not bool(self.http_client)
            try:
                resp = await client.post(url, json=payload, headers=headers)
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as he:
                    body = None
                    try:
                        body = resp.text
                    except Exception:
                        body = '<unavailable>'
                    logger.error("HTTP error posting to WhatsApp API: %s, status=%s, body=%s", he, resp.status_code, body)
                    raise
                # return parsed JSON when available, otherwise raw text
                try:
                    return resp.json()
                except Exception:
                    return {"raw": resp.text}
            finally:
                if _created_here:
                    await client.aclose()
        except Exception as e:
            logger.exception("HTTP error posting to WhatsApp API: %s", e)
            raise

    async def send_message(self, to: str, text: str) -> Dict[str, Any]:
        if not self._is_allowed(to):
            logger.warning("Recipient %s not allowed by allowlist", to)
            raise WhatsAppAPIError(f"recipient_not_allowed: {to}")

        # Retry loop with exponential backoff for transient failures
        attempt = 0
        while True:
            try:
                payload = {
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": to,
                    "type": "text",
                    "text": {"body": text}
                }
                # Construct messages endpoint using the phone number id when available
                if self.phone_number_id:
                    url = f"{(self.api_url or '').rstrip('/')}/{self.phone_number_id}/messages"
                else:
                    # fallback to previously used behaviour (may be incorrect for Graph API)
                    url = urljoin(self.api_url or "", "messages")
                resp = await self._do_post(url, payload)
                # interpret stub resp
                msgs = resp.get("messages") if isinstance(resp, dict) else None
                if msgs:
                    return {"id": msgs[0].get("id"), "status": msgs[0].get("status"), "to": to}
                return {"id": None, "status": "unknown", "to": to}
            except WhatsAppAPIError:
                # permanent error
                raise
            except Exception as exc:
                attempt += 1
                if attempt > self.max_retries:
                    logger.exception("Failed to send WhatsApp message after %d attempts", attempt)
                    raise
                backoff = self.backoff_base * (2 ** (attempt - 1))
                logger.info("Transient error sending message, retrying in %.1fs (attempt=%d)", backoff, attempt)
                await asyncio.sleep(backoff)

