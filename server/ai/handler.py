from typing import Optional, Dict, Any, List
import os
import logging
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AIProviderExhausted(Exception):
    pass


class AdvancedAIHandler:
    """AI handler that rotates Gemini API keys and optionally falls back to OpenRouter.

    This implementation provides a non-networking stubbed generation flow suitable
    for local testing and can be extended to perform real API calls.
    """

    def __init__(self, config=None, http_client=None):
        self.config = config
        self.http_client = http_client
        # Load API keys from environment; comma-separated lists supported
        gemini_raw = os.getenv("GEMINI_API_KEYS", "")
        self.gemini_keys: List[str] = [k.strip() for k in gemini_raw.split(",") if k.strip()]
        oraw = os.getenv("OPENROUTER_API_KEYS", "")
        self.openrouter_keys: List[str] = [k.strip() for k in oraw.split(",") if k.strip()]
        self.enable_openrouter_fallback = os.getenv("ENABLE_OPENROUTER_FALLBACK", "False").lower() in (
            "1",
            "true",
            "yes",
        )

        # rotation indices
        self._gemini_idx = 0
        self._openrouter_idx = 0

        # availability and backoff
        self.ai_available = bool(self.gemini_keys or (self.enable_openrouter_fallback and self.openrouter_keys))
        self.next_retry_time = None
        # Force DEV smoke mode when running in DEBUG config (convenience for local dev)
        try:
            self._force_dev = bool(self.config and getattr(self.config, "DEBUG", False))
        except Exception:
            self._force_dev = False
        # If DEV_SMOKE, consider AI available so MessageProcessor doesn't early-fail
        try:
            if os.getenv("DEV_SMOKE", "0").lower() in ("1", "true", "yes") or self._force_dev:
                self.ai_available = True
        except Exception:
            pass

    def _next_gemini_key(self) -> Optional[str]:
        if not self.gemini_keys:
            return None
        key = self.gemini_keys[self._gemini_idx % len(self.gemini_keys)]
        self._gemini_idx += 1
        return key

    def _next_openrouter_key(self) -> Optional[str]:
        if not self.openrouter_keys:
            return None
        key = self.openrouter_keys[self._openrouter_idx % len(self.openrouter_keys)]
        self._openrouter_idx += 1
        return key

    async def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Simple local analysis stub
        await asyncio.sleep(0)
        return {"intent": "auto", "entities": []}

    def _extract_retry_delay(self, error_msg: str) -> int:
        """Extract retry delay from Gemini error message"""
        try:
            import re
            match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_msg)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return 60  # Default retry delay

    async def _call_gemini(self, prompt: str, api_key: str) -> Dict[str, Any]:
        logger.info("Calling Gemini with prompt: %s", prompt)
        try:
            # Import and configure google.generativeai
            import google.generativeai as genai
            
            # Configure the API
            genai.configure(api_key=api_key)
            
            # Set up the generation config (no safety settings needed for Gemini 1.5)
            
            # Initialize model with the correct name - no safety settings for Gemini 1.5
            model = genai.GenerativeModel('gemini-1.5-pro-latest')
            
            # Run in executor since generate_content is blocking
            def generate():
                try:
                    response = model.generate_content(prompt)
                    return response.text
                except Exception as e:
                    error_msg = str(e)
                    if "quota exceeded" in error_msg.lower():
                        # Extract retry delay from error message
                        retry_delay = self._extract_retry_delay(error_msg)
                        self.next_retry_time = datetime.utcnow() + timedelta(seconds=retry_delay)
                        raise RuntimeError(f"gemini_key_error: quota exceeded, retry after {retry_delay}s") from e
                    raise

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, generate)
            
            logger.info("Gemini response: %s", response)
            return {"text": response, "provider": "gemini"}
            
        except Exception as e:
            logger.error("Gemini API call failed: %s", e)
            if "invalid api key" in str(e).lower():
                raise RuntimeError("gemini_key_error: invalid api_key") from e
            raise  # Re-raise other exceptions

    async def _call_openrouter(self, prompt: str, api_key: str) -> Dict[str, Any]:
        # If the user provided an OpenRouter key we'll try a real HTTP call.
        if not api_key:
            raise RuntimeError("openrouter_key_error")

        last_error = None
        max_retries = 3
        retry_delay = 1  # Start with 1 second delay

        for attempt in range(max_retries):
            try:
                import httpx

                # openrouter endpoint - use the direct HTTPS endpoint
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}", 
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/DanKosgey/whatsapp-agent",
                    "X-Title": "WhatsApp AI Agent",
                    "Accept": "application/json"
                }
                # Use a more specific model and add temperature/max tokens
                payload = {
                    "model": os.getenv("OPENROUTER_MODEL_ID", "openai/gpt-3.5-turbo"),
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }

                client = self.http_client or httpx.AsyncClient(timeout=30.0)
                _created_here = not bool(self.http_client)
                try:
                    r = await client.post(url, json=payload, headers=headers)
                    r.raise_for_status()
                    data = r.json()
                finally:
                    if _created_here:
                        await client.aclose()
                    # Try to extract a textual reply from common fields
                    if isinstance(data, dict):
                        choices = data.get("choices", [])
                        if choices and len(choices) > 0:
                            message = choices[0].get("message", {})
                            if message and "content" in message:
                                return {"text": message["content"], "provider": "openrouter"}

                    # If we get here, the response didn't match expected format
                    logger.warning("Unexpected OpenRouter response format: %s", data)
                    return {"text": "I apologize, but I received an unexpected response format. Please try again.", "provider": "openrouter"}

            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                    logger.warning("OpenRouter connection failed (attempt %d/%d), retrying in %ds: %s", 
                                 attempt + 1, max_retries, wait_time, str(e))
                    await asyncio.sleep(wait_time)
                    continue
                logger.exception("OpenRouter final retry failed")
                raise RuntimeError(f"openrouter_connection_error: {str(e)}") from e
            except Exception as e:
                logger.exception("OpenRouter request failed")
                raise RuntimeError(f"openrouter_error: {str(e)}") from e

        if last_error:
            raise last_error
        return {"text": "OpenRouter response could not be processed", "provider": "openrouter"}

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response using Gemini keys with rotation; fall back to OpenRouter if enabled.

        Returns a dict: {"text": str, "provider": "gemini"|"openrouter"}
        """
        now = datetime.utcnow()
        if self.next_retry_time and now < self.next_retry_time:
            logger.info("AI temporarily unavailable until %s", self.next_retry_time.isoformat())
            raise AIProviderExhausted("ai_unavailable")

        # If DEV_SMOKE env var is set OR the handler was constructed with a DEBUG config,
        # return a deterministic local reply immediately (convenience for local dev/testing).
        try:
            from server.config import config as _cfg2
        except Exception:
            _cfg2 = None
        dev_env = getattr(_cfg2, "DEV_SMOKE", False) if _cfg2 is not None else (os.getenv("DEV_SMOKE", "0").lower() in ("1", "true", "yes"))
        dev = dev_env or getattr(self, "_force_dev", False)
        if dev:
            logger.info("DEV_SMOKE active: returning deterministic reply")
            return {"text": f"[DEV_SMOKE reply to: {prompt}]", "provider": "dev"}

        # If no keys configured and no fallback available, raise clear error
        if not self.gemini_keys and not (self.enable_openrouter_fallback and self.openrouter_keys):
            raise RuntimeError("No AI API keys configured. Set GEMINI_API_KEYS or OPENROUTER_API_KEYS, or enable DEV_SMOKE for local development.")

        # Try Gemini keys first
        tried_gemini = 0
        gemini_count = len(self.gemini_keys)
        while tried_gemini < max(1, gemini_count):
            key = self._next_gemini_key()
            if not key:
                break
            try:
                logger.debug("Attempting Gemini with key index %d", self._gemini_idx - 1)
                resp = await self._call_gemini(prompt, key)
                self.ai_available = True
                return resp
            except Exception as e:
                logger.warning("Gemini key failed: %s", str(e))
                tried_gemini += 1

        # If we get here, Gemini keys failed or were not present
        logger.info("All Gemini keys exhausted or failed")
        # set a retry cooldown
        self.ai_available = False
        self.next_retry_time = datetime.utcnow() + timedelta(seconds=60)

        if self.enable_openrouter_fallback and self.openrouter_keys:
            # Try OpenRouter keys
            tried_or = 0
            or_count = len(self.openrouter_keys)
            while tried_or < max(1, or_count):
                or_key = self._next_openrouter_key()
                if not or_key:
                    break
                try:
                    logger.debug("Attempting OpenRouter with key index %d", self._openrouter_idx - 1)
                    resp = await self._call_openrouter(prompt, or_key)
                    # mark AI available since fallback succeeded
                    self.ai_available = True
                    self.next_retry_time = None
                    return resp
                except Exception as e:
                    logger.warning("OpenRouter key failed: %s", str(e))
                    tried_or += 1

        # Nothing worked
        logger.error("No AI providers available after rotation")
        raise AIProviderExhausted("no_providers_available")

