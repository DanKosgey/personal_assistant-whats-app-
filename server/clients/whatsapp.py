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
    
    Implements specific handling for WhatsApp API error 131030 to prevent endless retry loops.
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
        
        # Testing mode configuration
        self.disable_sends = os.getenv("DISABLE_WHATSAPP_SENDS", "false").lower() == "true"

    def _is_allowed(self, to: str) -> bool:
        if not self.allowed:
            return True
        return to in self.allowed

    async def _do_post(self, url: str, payload: Dict[str, Any], to: Optional[str] = None) -> Dict[str, Any]:
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
                        body_json = resp.json()
                    except Exception:
                        body = '<unavailable>'
                        body_json = {}
                    
                    # Check for recipient restriction error (Code 131030)
                    error_code = body_json.get('error', {}).get('code')
                    if error_code == 131030:  # Recipient not in allowed list
                        logger.warning("⚠️  WhatsApp API Error 131030: Recipient phone number not in allowed list: %s", to)
                        logger.info("💡 To fix this: Add recipient to WhatsApp Business account or enable testing mode with DISABLE_WHATSAPP_SENDS=true")
                        # Return a success-like response to prevent endless retries
                        return {"id": "restricted-message", "status": "restricted", "to": to, "error": "recipient_not_allowed", "error_code": 131030}
                    
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
        """Send a text message to a WhatsApp user with enhanced validation"""
        # Enhanced validation to prevent empty or invalid messages
        if not text or not isinstance(text, str):
            logger.error(f"Invalid message text: {text} (type: {type(text)})")
            # Provide a fallback message instead of failing
            text = "I'm here to help! How can I assist you today?"
        else:
            # Strip whitespace and check if message is effectively empty
            stripped_text = text.strip()
            if not stripped_text:
                logger.warning("Message text is empty or only whitespace")
                text = "I'm here to help! How can I assist you today?"
            else:
                # Clean the text to ensure it's valid for WhatsApp
                text = stripped_text
                
                # Remove any problematic characters that might cause API errors
                # Remove null bytes and other control characters
                text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
                
                # Enhanced cleaning for AI-generated responses with problematic tags
                import re
                text = re.sub(r'<s>\s*', '', text)
                text = re.sub(r'\s*</s>', '', text)
                text = re.sub(r'\[OUT\]', '', text)
                text = text.strip()
                
                # Check if text is effectively empty after cleaning
                if not text or len(text) == 0:
                    logger.warning("Message text is empty after cleaning")
                    text = "I'm here to help! How can I assist you today?"
                
                # Additional validation for meaningful content
                if len(text) < 2 or re.match(r'^[^a-zA-Z0-9]+$', text):
                    logger.warning("Message text has no meaningful content")
                    text = "I'm here to help! How can I assist you today?"
                
                # Ensure text is not too long for WhatsApp (limit to 4096 characters)
                if len(text) > 4096:
                    text = text[:4093] + "..."
                
                # If after cleaning the text is empty, provide fallback
                if not text.strip():
                    text = "I'm here to help! How can I assist you today?"
        
        # Additional validation for the 'to' parameter
        if not to or not isinstance(to, str):
            logger.error(f"Invalid recipient: {to} (type: {type(to)})")
            return {"status": "failed", "error": "Invalid recipient"}
            
        # Ensure 'to' parameter is properly formatted
        to = to.strip()
        if not to:
            logger.error("Recipient is empty or only whitespace")
            return {"status": "failed", "error": "Empty recipient"}
        
        # Fix attribute names
        url = f"{self.api_url}/{self.phone_number_id}/messages"
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {
                "body": text
            }
        }
        
        logger.info(f"Sending WhatsApp message to {to} with text length: {len(text)}")
        logger.debug(f"Message payload: {payload}")
        
        max_retries = 4
        for attempt in range(max_retries):
            try:
                resp = await self._do_post(url, payload, to=to)
                # Check if resp is a dict (stub response) or httpx.Response
                if isinstance(resp, dict):
                    # This is a stub response
                    logger.info(f"Successfully sent WhatsApp message to {to} (stub)")
                    return {"status": "success", "data": resp, "message_id": resp.get("messages", [{}])[0].get("id") if resp.get("messages") else None}
                elif hasattr(resp, 'status_code') and resp.status_code == 200:
                    data = resp.json()
                    logger.info(f"Successfully sent WhatsApp message to {to}")
                    return {"status": "success", "data": data, "message_id": data.get("messages", [{}])[0].get("id") if data.get("messages") else None}
                else:
                    status_code = getattr(resp, 'status_code', 'unknown')
                    error_text = getattr(resp, 'text', str(resp))
                    logger.error(f"Failed to send WhatsApp message: status={status_code}, body={error_text}")
                    if status_code == 400:
                        # For bad requests, log the specific error
                        try:
                            if hasattr(resp, 'json'):
                                error_data = resp.json()
                            else:
                                error_data = resp
                            logger.error(f"WhatsApp API error details: {error_data}")
                        except:
                            pass
            except Exception as e:
                logger.error(f"Error sending WhatsApp message (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Failed to send WhatsApp message after all retries")
                    return {"status": "failed", "error": str(e)}
        
        return {"status": "failed", "error": "Max retries exceeded"}
