import os
import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import httpx
from enum import Enum

log = logging.getLogger(__name__)


class OpenRouterError(Exception):
    """Base exception for OpenRouter client errors"""
    pass


class RateLimitError(OpenRouterError):
    """Raised when rate limit is exceeded"""
    pass


class AuthenticationError(OpenRouterError):
    """Raised when API key is invalid"""
    pass


class ModelNotFoundError(OpenRouterError):
    """Raised when model is not found"""
    pass


class QuotaExceededError(OpenRouterError):
    """Raised when quota is exceeded"""
    pass


@dataclass
class KeyStatus:
    """Track status and health of an API key"""
    key_prefix: str
    error_count: int = 0
    last_error: Optional[Dict[str, Any]] = None
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    is_blocked: bool = False
    block_until: Optional[datetime] = None
    total_requests: int = 0
    successful_requests: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def is_available(self) -> bool:
        if not self.is_blocked:
            return True
        if self.block_until and datetime.now() > self.block_until:
            self.is_blocked = False
            self.block_until = None
            return True
        return False

    def record_success(self):
        self.successful_requests += 1
        self.total_requests += 1
        self.consecutive_failures = 0
        self.last_success = datetime.now()
        # Unblock after successful request
        if self.is_blocked:
            self.is_blocked = False
            self.block_until = None

    def record_failure(self, error: Dict[str, Any], block_duration: Optional[timedelta] = None):
        self.error_count += 1
        self.total_requests += 1
        self.consecutive_failures += 1
        self.last_error = error
        
        # Block key if too many consecutive failures
        if self.consecutive_failures >= 3 and block_duration:
            self.is_blocked = True
            self.block_until = datetime.now() + block_duration


@dataclass
class GenerationRequest:
    """Request configuration for text generation"""
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False

    def __post_init__(self):
        if not self.prompt and not self.messages:
            raise ValueError("Either prompt or messages must be provided")
        if self.prompt and self.messages:
            raise ValueError("Cannot provide both prompt and messages")


@dataclass 
class GenerationResponse:
    """Response from text generation"""
    text: str
    model: str
    usage: Dict[str, int]
    raw_response: Dict[str, Any]
    key_used: str
    finish_reason: Optional[str] = None


class OpenRouterClient:
    """Enhanced OpenRouter API client with key rotation, error handling, and health monitoring"""
    
    def __init__(
        self,
        model: Optional[str] = None,
        keys: Optional[List[str]] = None,
        default_timeout: int = 30,
        max_retries: int = 3,
        base_url: Optional[str] = None,
        referer: Optional[str] = None,
        title: Optional[str] = None,
        enable_key_blocking: bool = True,
        block_duration_minutes: int = 5,
    ):
        # Load keys from environment if not provided
        if keys is None:
            keys = self._load_keys_from_env()
        
        if not keys:
            raise OpenRouterError("No OpenRouter API keys found")
            
        self.keys: List[str] = keys
        self.model = model or os.getenv("OPENROUTER_MODEL_ID")
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        # Default to the documented OpenRouter base URL; allow env override
        self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip('/')
        # Optional attribution headers (HTTP-Referer / X-Title)
        self.referer = referer or os.getenv("OPENROUTER_REFERER") or ""
        self.title = title or os.getenv("OPENROUTER_TITLE") or ""
        self.enable_key_blocking = enable_key_blocking
        self.block_duration = timedelta(minutes=block_duration_minutes)

        self._idx = 0
        self._lock = asyncio.Lock()

        # Initialize key status tracking
        self._key_status: Dict[str, KeyStatus] = {
            key: KeyStatus(key_prefix=key[:8]) for key in self.keys
        }

        log.info(f"Initialized OpenRouter client with {len(self.keys)} keys")

    @staticmethod
    def _load_keys_from_env() -> List[str]:
        """Load API keys from environment variables"""
        keys = []
        for k, v in sorted(os.environ.items()):
            if k.upper().startswith("OPENROUTER_API_KEY") and v:
                keys.append(v.strip())
        return keys

    def _get_error_class(self, status_code: int) -> type:
        """Map HTTP status codes to appropriate exception classes"""
        if status_code == 401:
            return AuthenticationError
        elif status_code == 404:
            return ModelNotFoundError
        elif status_code == 429:
            return RateLimitError
        elif status_code == 402:
            return QuotaExceededError
        else:
            return OpenRouterError

    async def _get_next_available_key(self) -> str:
        """Get next available key, skipping blocked ones"""
        async with self._lock:
            available_keys = [
                key for key in self.keys 
                if self._key_status[key].is_available
            ]
            
            if not available_keys:
                # Check if any blocked keys can be unblocked
                for key in self.keys:
                    if self._key_status[key].is_available:
                        available_keys.append(key)
                
                if not available_keys:
                    raise OpenRouterError("All API keys are currently blocked or unavailable")
            
            # Simple round-robin among available keys
            key = available_keys[self._idx % len(available_keys)]
            self._idx = (self._idx + 1) % len(available_keys)
            return key

    def _build_payload(self, request: GenerationRequest) -> Dict[str, Any]:
        """Build request payload from GenerationRequest"""
        payload = {
            "model": request.model or self.model,
            "temperature": request.temperature,
        }
        
        if not payload["model"]:
            raise OpenRouterError("Model ID not specified and OPENROUTER_MODEL_ID not set")
        
        # Handle messages vs prompt
        if request.messages:
            payload["messages"] = request.messages
        else:
            payload["messages"] = [{"role": "user", "content": request.prompt}]
        
        # Optional parameters
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.top_p != 1.0:
            payload["top_p"] = request.top_p
        if request.frequency_penalty != 0.0:
            payload["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty != 0.0:
            payload["presence_penalty"] = request.presence_penalty
        if request.stop:
            payload["stop"] = request.stop
        if request.stream:
            payload["stream"] = request.stream
            
        return payload

    async def _make_request(
        self, 
        key: str, 
        payload: Dict[str, Any], 
        timeout: int
    ) -> httpx.Response:
        """Make HTTP request to OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.referer,
            "X-Title": self.title
        }
        # Allow httpx to pick up proxy settings from the environment (HTTP_PROXY/HTTPS_PROXY)
        # by enabling trust_env. This helps when running behind a corporate proxy.
        async with httpx.AsyncClient(timeout=timeout, trust_env=True) as client:
            url = f"{self.base_url}/chat/completions"
            response = await client.post(url, json=payload, headers=headers)
            return response

    async def generate(
        self, 
        request: Union[str, GenerationRequest], 
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout: Optional[int] = None
    ) -> GenerationResponse:
        """
        Generate text using OpenRouter API with automatic key rotation and error handling.
        
        Args:
            request: Either a prompt string or GenerationRequest object
            model: Model to use (overrides default)
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            
        Returns:
            GenerationResponse object containing the generated text and metadata
        """
        # Handle string prompt input
        if isinstance(request, str):
            request = GenerationRequest(
                prompt=request,
                model=model,
                temperature=temperature
            )
        elif model:
            request.model = model
        
        timeout = timeout or self.default_timeout
        last_exc = None
        key = None
        
        # Try each available key
        for attempt in range(min(self.max_retries, len(self.keys))):
            try:
                key = await self._get_next_available_key()
                key_status = self._key_status[key]
                payload = self._build_payload(request)
                
                log.debug(f"Attempting request with key {key[:8]} (attempt {attempt + 1})")
                
                response = await self._make_request(key, payload, timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    key_status.record_success()
                    
                    # Parse response
                    text = self.extract_text_from_response(data)
                    if not text:
                        raise OpenRouterError("No text content in response")
                    
                    return GenerationResponse(
                        text=text,
                        model=data.get('model', request.model or self.model),
                        usage=data.get('usage', {}),
                        raw_response=data,
                        key_used=key[:8],
                        finish_reason=data.get('choices', [{}])[0].get('finish_reason')
                    )
                else:
                    # Handle error response
                    error_info = {
                        "status_code": response.status_code,
                        "response_text": response.text[:1000],
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    block_duration = self.block_duration if self.enable_key_blocking else None
                    key_status.record_failure(error_info, block_duration)
                    
                    # Raise appropriate exception
                    error_class = self._get_error_class(response.status_code)
                    last_exc = error_class(
                        f"OpenRouter API error {response.status_code}: {response.text}"
                    )
                    
                    log.warning(f"Key {key[:8]} failed with status {response.status_code}")
                    
                    # Don't retry on certain errors
                    if response.status_code in [401, 404]:
                        break
                        
            except Exception as e:
                # Only record failure if we actually had a key
                if key and key in self._key_status:
                    error_info = {
                        "exception": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    block_duration = self.block_duration if self.enable_key_blocking else None
                    self._key_status[key].record_failure(error_info, block_duration)

                if key:
                    try:
                        log.exception(f"Request failed with key {key[:8]}")
                    except Exception:
                        log.exception("Request failed (key hidden)")
                else:
                    log.exception("Request failed before selecting a key")

                last_exc = e

                # Brief backoff between attempts
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
        
        # All attempts failed
        raise OpenRouterError("All API keys failed after retries") from last_exc

    async def test_key(
        self, 
        key: str, 
        model: Optional[str] = None, 
        timeout: int = 20
    ) -> Dict[str, Any]:
        """
        Test a single API key with a simple request.
        
        Args:
            key: API key to test
            model: Model to use for testing
            timeout: Request timeout
            
        Returns:
            Raw API response data
        """
        request = GenerationRequest(
            prompt="Hello, please respond briefly with 'ok'.",
            model=model or self.model,
            temperature=0.0
        )
        
        payload = self._build_payload(request)
        response = await self._make_request(key, payload, timeout)
        response.raise_for_status()
        return response.json()

    def get_key_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all API keys"""
        return {
            key[:8]: {
                "error_count": status.error_count,
                "success_rate": status.success_rate,
                "last_error": status.last_error,
                "last_success": status.last_success.isoformat() if status.last_success else None,
                "consecutive_failures": status.consecutive_failures,
                "is_blocked": status.is_blocked,
                "block_until": status.block_until.isoformat() if status.block_until else None,
                "total_requests": status.total_requests,
                "successful_requests": status.successful_requests,
                "is_available": status.is_available
            }
            for key, status in self._key_status.items()
        }

    def get_healthy_keys(self) -> List[str]:
        """Get list of currently healthy/available keys"""
        return [
            key[:8] for key, status in self._key_status.items()
            if status.is_available and status.success_rate > 0.5
        ]

    @staticmethod
    def extract_text_from_response(data: Dict[str, Any]) -> Optional[str]:
        """
        Extract text content from OpenRouter API response.
        Handles various response formats.
        """
        if not data or not isinstance(data, dict):
            return None
        # Standard OpenAI format
        choices = data.get('choices', [])
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get('message', {})
                if isinstance(message, dict):
                    content = message.get('content')
                    if isinstance(content, str):
                        return content
                        
                # Handle streaming format
                delta = choice.get('delta', {})
                if isinstance(delta, dict):
                    content = delta.get('content')
                    if isinstance(content, str):
                        return content
                        
                # Legacy format
                text = choice.get('text')
                if isinstance(text, str):
                    return text
        
        # Alternative formats
        text = data.get('text')
        if isinstance(text, str):
            return text
            
        return None

    async def list_models(self) -> Dict[str, Any]:
        """List available models from OpenRouter"""
        key = await self._get_next_available_key()
        headers = {"Authorization": f"Bearer {key}"}
        # Use trust_env here too so listing models can go through a configured proxy
        async with httpx.AsyncClient(timeout=self.default_timeout, trust_env=True) as client:
            response = await client.get(f"{self.base_url}/models", headers=headers)
            response.raise_for_status()
            return response.json()

    async def close(self):
        """Cleanup method for proper resource management"""
        # Any cleanup code would go here
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Backwards-compatible module-level alias for older tests/code that import this name
def extract_text_from_openrouter_response(data: Dict[str, Any]) -> Optional[str]:
    """Compatibility wrapper: delegate to OpenRouterClient.extract_text_from_response."""
    return OpenRouterClient.extract_text_from_response(data)


# Convenience functions
async def generate_text(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs
) -> str:
    """Simple convenience function for one-off text generation"""
    async with OpenRouterClient(model=model, **kwargs) as client:
        response = await client.generate(prompt, temperature=temperature)
        return response.text


async def test_all_keys() -> Dict[str, Any]:
    """Test all configured API keys and return their status"""
    client = OpenRouterClient()
    results = {}
    
    for key in client.keys:
        try:
            await client.test_key(key)
            results[key[:8]] = {"status": "success", "error": None}
        except Exception as e:
            results[key[:8]] = {"status": "failed", "error": str(e)}
    
    return results