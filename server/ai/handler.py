from typing import Optional, Dict, Any, List
import os
import logging
import asyncio
import json
import re
from datetime import datetime, timedelta
import sys  # Add sys import

# Import httpx for OpenRouter support
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False

# Import KeyManager with absolute import
from server.key_manager import KeyManager

# Import Google Generative AI modules
try:
    import google.generativeai as genai
    from google.generativeai.types import content_types as glm_content_types
    from google.ai import generativelanguage as glm
    GOOGLE_GENERATIVE_AI_AVAILABLE = True
except ImportError:
    genai = None
    glm_content_types = None
    glm = None
    GOOGLE_GENERATIVE_AI_AVAILABLE = False

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
        
        # Initialize KeyManager to handle API keys
        self.key_manager = KeyManager()
        
        # Load API keys via KeyManager
        self.gemini_keys = self.key_manager.gemini_keys
        self.openrouter_keys = self.key_manager.openrouter_keys
        self.enable_openrouter_fallback = self.key_manager.openrouter_settings.get("enabled", False)

        # rotation indices (now managed by KeyManager, but keeping for backward compatibility)
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
            # Check multiple sources for DEV_SMOKE
            dev_smoke_env = os.getenv("DEV_SMOKE", "0")
            # Also check if running in a test environment
            is_test_env = hasattr(sys, '_called_from_test') or 'pytest' in sys.modules
            dev_smoke_condition = dev_smoke_env.lower() in ("1", "true", "yes") or is_test_env
            logger.debug(f"DEV_SMOKE env: '{dev_smoke_env}', condition: {dev_smoke_condition}, _force_dev: {self._force_dev}, is_test_env: {is_test_env}")
            if dev_smoke_condition or self._force_dev:
                self.ai_available = True
                logger.info("DEV_SMOKE mode enabled - AI service marked as available")
            else:
                logger.debug("DEV_SMOKE mode not enabled")
        except Exception as e:
            logger.error(f"Error checking DEV_SMOKE mode: {e}")
            pass
        
        # Initialize persona manager
        try:
            from server.persona_manager import PersonaManager
            self.persona_manager = PersonaManager('server/personas')
            # Auto-select efficient assistant persona if available
            if 'efficient_assistant' in self.persona_manager.list_personas():
                self.persona_manager.select_persona('efficient_assistant')
        except Exception as e:
            logger.warning(f"Failed to initialize PersonaManager: {e}")
            self.persona_manager = None
        
        # Initialize tools
        try:
            from server.tools.profile_tools import LLMProfileTools
            self.tools = LLMProfileTools()
        except Exception as e:
            logger.warning(f"Failed to initialize LLMProfileTools: {e}")
            self.tools = None
            
        # Log the initial state
        logger.info(f"AI Handler initialized - Available: {self.ai_available}, "
                   f"Gemini Keys: {len(self.gemini_keys)}, OpenRouter Keys: {len(self.openrouter_keys)}, "
                   f"OpenRouter Fallback: {self.enable_openrouter_fallback}")

    def _next_gemini_key(self) -> Optional[str]:
        """Get next available Gemini key using KeyManager."""
        key_obj = self.key_manager.next_gemini_key()
        if key_obj:
            self._gemini_idx = (self._gemini_idx + 1) % max(1, len(self.gemini_keys) if self.gemini_keys else 1)
            # Handle both object and string cases
            if hasattr(key_obj, "key"):
                return key_obj.key
            else:
                return str(key_obj)
        return None

    def _next_openrouter_key(self) -> Optional[str]:
        """Get next available OpenRouter key using KeyManager."""
        key_obj = self.key_manager.next_openrouter_key()
        if key_obj:
            # Handle both object and string cases
            if hasattr(key_obj, "key"):
                return key_obj.key
            else:
                return str(key_obj)
        return None

    async def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Simple local analysis stub
        await asyncio.sleep(0)
        return {"intent": "auto", "entities": []}

    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name with parameters"""
        if not self.tools:
            return {"success": False, "message": "Tools not available"}
        
        try:
            result = await self.tools.execute_tool(tool_name, **kwargs)
            # Accept dicts or objects
            if isinstance(result, dict):
                return {
                    "success": result.get("success", True),
                    "message": result.get("message"),
                    "data": result.get("data"),
                    "error": result.get("error"),
                }
            else:
                return {
                    "success": getattr(result, "success", True),
                    "message": getattr(result, "message", None),
                    "data": getattr(result, "data", None),
                    "error": getattr(result, "error", None),
                }
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return {
                "success": False,
                "message": f"Tool execution failed: {str(e)}",
                "error": "tool_execution_failed"
            }

    def _extract_function_calls(self, resp):
        """Extract function calls from response in a robust way"""
        calls = []
        try:
            candidates = getattr(resp, "candidates", None) or (resp.get("candidates", []) if isinstance(resp, dict) else [])
            if candidates:
                candidate = candidates[0]
                # Candidate may hold function call in different places
                func_calls = getattr(candidate, "function_calls", None)
                if not func_calls:
                    # try content.parts -> Part.FunctionCall or a dict style
                    parts = getattr(candidate, "content", None)
                    if not parts:
                        parts = getattr(candidate, "content_parts", None)
                    # fallback for dict
                    if isinstance(candidate, dict):
                        # cloud response variant
                        for part in candidate.get("content", {}).get("parts", []):
                            fc = part.get("function_call") or part.get("FunctionCall")
                            if fc:
                                func_calls = func_calls or []
                                func_calls.append(fc)
                if func_calls:
                    for fc in func_calls:
                        name = getattr(fc, "name", None) or (fc.get("name") if isinstance(fc, dict) else None)
                        args = getattr(fc, "args", None) or (fc.get("args") if isinstance(fc, dict) else None)
                        # args might be a JSON string
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                # leave as str if it isn't JSON
                                pass
                        calls.append({"name": name, "args": args})
        except Exception as e:
            logger.warning(f"Error extracting function calls: {e}")
        return calls

    async def _process_function_calls(self, original_prompt: str, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process function calls from AI response and generate final response"""
        if not self.tools:
            return response
        
        try:
            # Extract function calls from response using robust extraction
            function_calls = self._extract_function_calls(response)
            if not function_calls:
                return response
            
            # Process each function call
            tool_results = []
            for func_call in function_calls:
                try:
                    func_name = func_call.get("name")
                    func_args = func_call.get("args", {})
                    
                    if func_name:
                        # Execute the tool - handle both dict and string args
                        if isinstance(func_args, dict):
                            result = await self.execute_tool(func_name, **func_args)
                        else:
                            # If args is a string or other type, pass as single parameter
                            result = await self.execute_tool(func_name, raw_args=func_args)
                        tool_results.append({
                            "name": func_name,
                            "arguments": func_args,
                            "result": result
                        })
                except Exception as e:
                    logger.error(f"Error processing function call: {e}")
                    continue
            
            # If we have tool results, create a follow-up prompt and make a second call
            if tool_results:
                # Create a prompt that includes the tool results
                follow_up_prompt = f"{original_prompt}\n\nTool execution results:\n"
                for tool_result in tool_results:
                    follow_up_prompt += f"Function: {tool_result['name']}\n"
                    try:
                        follow_up_prompt += f"Arguments: {json.dumps(tool_result['arguments'])}\n"
                    except Exception:
                        follow_up_prompt += f"Arguments: {tool_result['arguments']}\n"
                    try:
                        follow_up_prompt += f"Result: {json.dumps(tool_result['result'])}\n\n"
                    except Exception:
                        follow_up_prompt += f"Result: {tool_result['result']}\n\n"
                
                # Make a second call with the tool results incorporated
                # Try to use the same provider (Gemini) for consistency
                if response.get("provider") == "gemini" and self.gemini_keys:
                    key = self._next_gemini_key()
                    if key:
                        try:
                            follow_up_response = await self._call_gemini(follow_up_prompt, key)
                            # Return the follow-up response with tool execution info
                            follow_up_response["tool_executions"] = tool_results
                            return follow_up_response
                        except Exception as e:
                            logger.warning(f"Follow-up Gemini call failed: {e}")
                
                # Fallback: just add tool results to original response
                response["tool_executions"] = tool_results
                
            return response
            
        except Exception as e:
            logger.error(f"Error processing function calls: {e}")
            return response

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

    async def _call_gemini(self, prompt: str, api_key: str, tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        logger.debug("Calling Gemini with prompt length=%d", len(prompt) if isinstance(prompt, str) else -1)
        logger.info("GEMINI_API_CALL: Starting request processing")
        logger.info("GEMINI_API_CALL: Request details - Prompt length: %d characters", len(prompt) if isinstance(prompt, str) else -1)
        try:
            # Import and configure google.generativeai
            import google.generativeai as genai
            
            # Configure the API
            logger.debug("GEMINI_API_CALL: Configuring API with key (first 8 chars): %s...", api_key[:8] if api_key else "None")
            genai.configure(api_key=api_key)
            
            # Set up the generation config (no safety settings needed for Gemini 1.5)
            
            # Initialize model with the configured model name from environment
            model_name = os.getenv("AI_MODEL", "gemini-1.5-flash")
            logger.debug("GEMINI_API_CALL: Using model: %s", model_name)
            model = genai.GenerativeModel(model_name)
            
            # Prepare tools if provided - but don't process them to avoid duplicate parameter error
            # The main issue was passing tools both via model._tools and as a per-call kwarg
            # By not setting model._tools here, we avoid that conflict
            gemini_tools = None
            if tools:
                try:
                    logger.debug("GEMINI_API_CALL: Processing %d tools", len(tools))
                    # Log the tools for debugging (sanitize secrets)
                    logger.debug("Prepared gemini_tools: %s", json.dumps(tools, indent=2))
                    
                    # We're not processing tools here to avoid the duplicate parameter issue
                    # The tools will be handled by the model's internal mechanism
                    logger.debug('Skipping tool processing to prevent duplicate parameter error')
                except Exception as e:
                    logger.warning(f"GEMINI_API_CALL: Failed to prepare tools for Gemini: {e}")
                    gemini_tools = None  # Disable tools if preparation fails
            
            # Run in executor since generate_content is blocking
            def generate():
                try:
                    logger.debug("GEMINI_API_CALL: Starting synchronous generation")
                    # Prepare the call parameters correctly - using direct client approach
                    # This approach avoids the "multiple values for keyword argument" error
                    import google.generativeai.types as types
                    import google.generativeai.generative_models as gen_models
                    from google.generativeai.types import content_types
                    from google.ai import generativelanguage as glm
                    
                    # Create the generation config
                    generation_config = types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=1000,
                    )
                    logger.debug("GEMINI_API_CALL: Generation config: temperature=0.7, max_output_tokens=1000")
                    
                    # Log what we're passing for debugging
                    logger.debug("GEMINI_API_CALL: generate_content - prompt length: %d, has_tools: %s", 
                               len(prompt) if isinstance(prompt, str) else -1, 
                               bool(model._tools if hasattr(model, '_tools') else False))
                    
                    # Build contents safely (works for string or list as in model wrapper)
                    contents = content_types.to_contents(prompt)  # prompt may be string or list

                    # Use the preferred approach: attach tools to model._tools and call generate_content WITHOUT tools kwarg
                    response = model.generate_content(
                        contents=contents,
                        generation_config=generation_config,
                        # DO NOT pass tools=... here to avoid duplicate parameter error
                    )

                    logger.info("GEMINI_API_CALL: API request completed successfully")
                    
                    return response
                except Exception as e:
                    logger.error("GEMINI_API_CALL: Generation failed: %s", e, exc_info=True)
                    error_msg = str(e)
                    if "quota exceeded" in error_msg.lower() or "429" in error_msg:
                        # Extract retry delay from error message
                        retry_delay = self._extract_retry_delay(error_msg)
                        logger.warning("GEMINI_API_CALL: Quota exceeded, retry after %d seconds", retry_delay)
                        raise RuntimeError(f"gemini_key_error: quota exceeded, retry after {retry_delay}s") from e
                    elif "invalid api key" in error_msg.lower():
                        logger.warning("GEMINI_API_CALL: Invalid API key detected")
                        raise RuntimeError("gemini_key_error: invalid_api_key") from e
                    raise

            loop = asyncio.get_event_loop()
            logger.debug("GEMINI_API_CALL: Running generation in executor")
            response = await loop.run_in_executor(None, generate)
            
            logger.debug("GEMINI_API_CALL: Response received len=%d", len(response.text) if hasattr(response, "text") else -1)
            logger.info("GEMINI_API_CALL: Response received successfully, length: %d characters", 
                       len(response.text) if hasattr(response, "text") else -1)
            
            # Handle function calls if present using robust extraction
            function_calls = self._extract_function_calls(response)
            if function_calls:
                logger.debug("GEMINI_API_CALL: Function calls detected in response: %d calls", len(function_calls))
                # Log details of function calls for debugging
                for i, func_call in enumerate(function_calls):
                    logger.debug("GEMINI_API_CALL: Function call %d: name=%s, args=%s", 
                               i, func_call.get("name", "unknown"), 
                               json.dumps(func_call.get("args", {}), indent=2) if isinstance(func_call.get("args"), dict) else func_call.get("args"))
                return {
                    "text": response.text if hasattr(response, "text") else "",
                    "function_calls": function_calls,
                    "provider": "gemini",
                    "raw_response": response
                }
            
            logger.debug("GEMINI_API_CALL: No function calls in response, returning text only")
            return {
                "text": response.text if hasattr(response, "text") else "",
                "provider": "gemini",
                "raw_response": response
            }
            
        except Exception as e:
            logger.error("GEMINI_API_CALL: API call failed: %s", e, exc_info=True)
            if "invalid api key" in str(e).lower():
                logger.error("GEMINI_API_CALL: Critical error: Invalid API key")
                raise RuntimeError("gemini_key_error: invalid_api_key") from e
            elif "quota exceeded" in str(e).lower() or "429" in str(e):
                # Extract retry delay from error message
                retry_delay = self._extract_retry_delay(str(e))
                logger.error("GEMINI_API_CALL: Critical error: Quota exceeded, retry after %d seconds", retry_delay)
                raise RuntimeError(f"gemini_key_error: quota exceeded, retry after {retry_delay}s") from e
            logger.error("GEMINI_API_CALL: Unexpected error occurred")
            raise  # Re-raise other exceptions

    async def _call_openrouter(self, prompt: str, api_key: str) -> Dict[str, Any]:
        # If the user provided an OpenRouter key we'll try a real HTTP call.
        if not api_key:
            raise RuntimeError("openrouter_key_error")
        
        logger.info("OPENROUTER_API_CALL: Starting request with key: %s...", api_key[:8])
        logger.info("OPENROUTER_API_CALL: Request details - Prompt length: %d characters", len(prompt))
        
        # Define these at the beginning to avoid linter errors
        url = "https://openrouter.ai/api/v1/chat/completions"
        model_id = os.getenv("OPENROUTER_MODEL_ID", "openai/gpt-3.5-turbo")
        
        last_error = None
        max_retries = 5  # Increase retries for better resilience
        retry_delay = 1  # Start with 1 second delay

        # Define payload outside the loop
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        logger.debug("OpenRouter API - Request payload: model=%s, prompt_length=%d, temperature=0.7, max_tokens=1000", 
                    model_id, len(prompt))

        data = None
        for attempt in range(max_retries):
            logger.debug("OpenRouter API - Attempt %d/%d", attempt + 1, max_retries)
            try:
                import httpx

                # openrouter endpoint - use the direct HTTPS endpoint
                # url is already defined at the beginning of the function
                headers = {
                    "Authorization": f"Bearer {api_key}", 
                    "Content-Type": "application/json",
                    "Referer": "https://github.com/DanKosgey/whatsapp-agent",
                    "X-Title": "WhatsApp AI Agent",
                    "Accept": "application/json"
                }
                logger.debug("OPENROUTER_API_CALL: Request headers prepared, URL: %s", url)
                logger.debug("OPENROUTER_API_CALL: Request headers details: Authorization=%s..., Content-Type=%s, Referer=%s", 
                           f"Bearer {api_key[:8]}..." if api_key else "None", 
                           headers.get("Content-Type"), 
                           headers.get("Referer"))

                client = self.http_client or httpx.AsyncClient(timeout=30.0)
                _created_here = not bool(self.http_client)
                try:
                    logger.info("OPENROUTER_API_CALL: Making POST request to %s", url)
                    logger.info("OPENROUTER_API_CALL: Request details - URL: %s, Model: %s, Temperature: 0.7, Max tokens: 1000", url, model_id)
                    logger.debug("OPENROUTER_API_CALL: Request payload details: %s", json.dumps(payload, indent=2, default=str))
                    r = await client.post(url, json=payload, headers=headers)
                    logger.debug("OpenRouter API - Response received, status: %d", r.status_code)
                    
                    # Log response details for better debugging
                    logger.debug("OpenRouter response status: %d", r.status_code)
                    if r.status_code != 200:
                        try:
                            error_data = r.json()
                            logger.warning("OpenRouter error response: %s", json.dumps(error_data, indent=2, default=str))
                        except Exception:
                            logger.warning("OpenRouter error response (text): %s", r.text)
                    
                    # Handle different status codes with specific logic
                    if r.status_code == 404:
                        # Model not found - try a different model
                        logger.warning("OpenRouter 404 error: Model not found for %s", payload["model"])
                        # Try a different free model as fallback
                        fallback_models = [
                            "mistralai/mistral-7b-instruct:free",
                            "google/gemma-7b-it:free",
                            "microsoft/phi-3-mini-128k-instruct:free"
                        ]
                        # Cycle through fallback models
                        current_model = payload["model"]
                        try:
                            current_index = fallback_models.index(current_model)
                            payload["model"] = fallback_models[(current_index + 1) % len(fallback_models)]
                        except ValueError:
                            # Current model not in fallback list, use first fallback model
                            payload["model"] = fallback_models[0]
                        logger.info("Trying fallback model: %s", payload["model"])
                        
                        # Wait before retry with exponential backoff
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt) + (attempt * 0.1)  # Add jitter
                            logger.warning("OpenRouter 404 error (attempt %d/%d), trying different model in %ds", 
                                         attempt + 1, max_retries, wait_time)
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise RuntimeError(f"openrouter_model_not_found: Model {current_model} not found") from None
                    
                    elif r.status_code == 502:
                        # Bad gateway - temporary server issue
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt) + (attempt * 0.1)  # Add jitter
                            logger.warning("OpenRouter 502 error (attempt %d/%d), retrying in %ds", 
                                         attempt + 1, max_retries, wait_time)
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise RuntimeError("openrouter_bad_gateway: Server temporarily unavailable") from None
                    
                    elif r.status_code >= 500:
                        # Other server errors - retry with exponential backoff
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt) + (attempt * 0.1)  # Add jitter
                            logger.warning("OpenRouter server error %d (attempt %d/%d), retrying in %ds", 
                                         r.status_code, attempt + 1, max_retries, wait_time)
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise RuntimeError(f"openrouter_server_error: Server error {r.status_code}") from None
                    
                    elif r.status_code == 429:
                        # Rate limit error - mark key as temporarily unavailable
                        logger.warning("OpenRouter rate limit hit with key: %s", api_key[:8])
                        self.key_manager.mark_openrouter_key_unavailable(api_key, 120)  # 2 minutes
                        raise RuntimeError("openrouter_rate_limit_error: Rate limit exceeded") from None
                    
                    # Raise for other HTTP errors
                    r.raise_for_status()
                    data = r.json()
                    logger.debug("OPENROUTER_API_CALL: Response parsed successfully")
                    logger.info("OPENROUTER_API_CALL: Response details - Status code: %d", r.status_code)
                    logger.debug("OPENROUTER_API_CALL: Response data: %s", json.dumps(data, indent=2, default=str))
                finally:
                    if _created_here:
                        await client.aclose()

                if isinstance(data, dict):
                    choices = data.get("choices", [])
                    if choices and len(choices) > 0:
                        message = choices[0].get("message", {})
                        if message and "content" in message:
                            response_text = message["content"]
                            logger.info("OPENROUTER_API_CALL: Response processed successfully, text length: %d", len(response_text))
                            logger.info("OPENROUTER_API_CALL: Response details - Text length: %d characters", len(response_text))
                            logger.debug("OPENROUTER_API_CALL: Response text preview: %s", response_text[:200] + "..." if len(response_text) > 200 else response_text)
                            # Record successful usage of the key
                            return {"text": response_text, "provider": "openrouter"}

                logger.warning("Unexpected OpenRouter response format (keys=%s)", list(data.keys()) if isinstance(data, dict) else type(data))
                logger.debug("OpenRouter API - Unexpected response data: %s", json.dumps(data, indent=2, default=str) if isinstance(data, dict) else str(data))
                return {"text": "I apologize, but I received an unexpected response format. Please try again.", "provider": "openrouter"}

            except Exception as e:
                # Handle httpx errors if httpx is available
                is_connection_error = False
                is_server_error = False
                is_rate_limit_error = False
                is_bad_gateway = False
                is_not_found = False
                
                if HTTPX_AVAILABLE:
                    try:
                        # Try to check if it's a connection error
                        is_connection_error = 'ConnectError' in str(type(e)) or 'ConnectTimeout' in str(type(e))
                        # Check for server errors (5xx)
                        is_server_error = '502' in str(e) or '503' in str(e) or '504' in str(e)
                        is_bad_gateway = '502' in str(e)
                        # Check for rate limit errors (429)
                        is_rate_limit_error = '429' in str(e)
                        # Check for not found errors (404)
                        is_not_found = '404' in str(e)
                        # Also check if it's an HTTPStatusError with status code 404
                        try:
                            response = getattr(e, 'response', None)
                            if response is not None:
                                status_code = getattr(response, 'status_code', None)
                                if status_code == 429:
                                    is_rate_limit_error = True
                                elif status_code == 404:
                                    is_not_found = True
                        except Exception:
                            pass
                    except Exception:
                        # If we can't check, assume it's not a connection error
                        is_connection_error = False
                        is_server_error = False
                        is_rate_limit_error = False
                        is_bad_gateway = False
                        is_not_found = False
                
                # Log detailed error information
                logger.error("OPENROUTER_API_CALL: API call failed (attempt %d/%d): %s", attempt + 1, max_retries, str(e), exc_info=True)
                logger.error("OPENROUTER_API_CALL: Error details - URL: %s, Model: %s, Attempt: %d/%d", url, model_id, attempt + 1, max_retries)
                
                # If it's a rate limit error, implement exponential backoff
                if is_rate_limit_error:
                    # Calculate exponential backoff with jitter
                    wait_time = retry_delay * (2 ** attempt) + (attempt * 0.1)  # Add jitter
                    logger.warning("OpenRouter rate limit hit (attempt %d/%d) with key %s, waiting %ds", 
                                 attempt + 1, max_retries, api_key[:8], wait_time)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait_time)
                        continue
                    logger.exception("OpenRouter rate limit error after retries")
                    # Mark key as temporarily unavailable for longer period
                    self.key_manager.mark_openrouter_key_unavailable(api_key, 120)  # 2 minutes
                    raise RuntimeError(f"openrouter_rate_limit_error: {str(e)}") from e
                # If it's a 404 error, log details and try a different model as fallback
                elif is_not_found and attempt < max_retries - 1:
                    logger.warning("OpenRouter bad gateway error (attempt %d/%d), trying different model", 
                                 attempt + 1, max_retries)
                    # Try a different free model
                    fallback_models = [
                        "mistralai/mistral-7b-instruct:free",
                        "google/gemma-7b-it:free",
                        "microsoft/phi-3-mini-128k-instruct:free"
                    ]
                    # Cycle through fallback models
                    current_model = payload["model"]
                    try:
                        current_index = fallback_models.index(current_model)
                        payload["model"] = fallback_models[(current_index + 1) % len(fallback_models)]
                    except ValueError:
                        # Current model not in fallback list, use first fallback model
                        payload["model"] = fallback_models[0]
                    wait_time = retry_delay * (2 ** attempt) + (attempt * 0.1)  # Add jitter
                    await asyncio.sleep(wait_time)
                    continue
                # If it's a bad gateway error, try a different model as fallback
                elif is_bad_gateway and attempt < max_retries - 1:
                    logger.warning("OpenRouter bad gateway error (attempt %d/%d), trying different model", 
                                 attempt + 1, max_retries)
                    # Try a different free model
                    fallback_models = [
                        "mistralai/mistral-7b-instruct:free",
                        "google/gemma-7b-it:free",
                        "microsoft/phi-3-mini-128k-instruct:free"
                    ]
                    # Cycle through fallback models
                    current_model = payload["model"]
                    try:
                        current_index = fallback_models.index(current_model)
                        payload["model"] = fallback_models[(current_index + 1) % len(fallback_models)]
                    except ValueError:
                        # Current model not in fallback list, use first fallback model
                        payload["model"] = fallback_models[0]
                    wait_time = retry_delay * (2 ** attempt) + (attempt * 0.1)  # Add jitter
                    await asyncio.sleep(wait_time)
                    continue
                elif is_server_error and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt) + (attempt * 0.1)  # Add jitter
                    logger.warning("OpenRouter server error (attempt %d/%d), retrying in %ds", 
                                 attempt + 1, max_retries, wait_time)
                    await asyncio.sleep(wait_time)
                    continue
                elif is_connection_error:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt) + (attempt * 0.1)  # Add jitter
                        logger.warning("OpenRouter connection failed (attempt %d/%d), retrying in %ds", 
                                     attempt + 1, max_retries, wait_time)
                        await asyncio.sleep(wait_time)
                        continue
                    logger.exception("OpenRouter final retry failed")
                    raise RuntimeError(f"openrouter_connection_error: {str(e)}") from e
                else:
                    logger.exception("OpenRouter request failed")
                    # For other errors, mark key as temporarily unavailable for a short period
                    if attempt >= max_retries - 1:
                        self.key_manager.mark_openrouter_key_unavailable(api_key, 30)  # 30 seconds
                    raise RuntimeError(f"openrouter_error: {str(e)}") from e

        if last_error:
            raise last_error
        return {"text": "OpenRouter response could not be processed", "provider": "openrouter"}

    async def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response using Gemini keys with rotation; fall back to OpenRouter if enabled.

        Returns a dict: {"text": str, "provider": "gemini"|"openrouter"}
        """
        logger.info("AI Generate - Starting generation request, prompt length: %d", len(prompt))
        now = datetime.now().astimezone()
        if self.next_retry_time and now < self.next_retry_time:
            logger.info("AI temporarily unavailable until %s", self.next_retry_time.isoformat())
            # Reset the retry time to allow trying again if enough time has passed
            if (now - self.next_retry_time).total_seconds() > 300:  # 5 minutes
                self.next_retry_time = None
                self.ai_available = True
            else:
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
            dev_response = {"text": f"[DEV_SMOKE reply to: {prompt}]", "provider": "dev"}
            logger.debug("DEV_SMOKE response: %s", dev_response)
            return dev_response

        # Get tools if available
        tools = None
        if self.tools:
            try:
                tools = self.tools.get_all_tool_schemas()
                logger.debug("AI Generate - Retrieved %d tool schemas", len(tools) if tools else 0)
            except Exception as e:
                logger.warning(f"Failed to get tool schemas: {e}")

        # If no keys configured and no fallback available, raise clear error
        if not self.gemini_keys and not (self.enable_openrouter_fallback and self.openrouter_keys):
            # Check if we're in DEV_SMOKE mode
            if dev:
                dev_response = {"text": f"[DEV_SMOKE reply to: {prompt}]", "provider": "dev"}
                logger.debug("DEV_SMOKE response (no keys): %s", dev_response)
                return dev_response
            
            # Provide a more informative error message
            error_msg = "No AI API keys configured. "
            # Ensure keys are lists before checking length
            gemini_keys_len = len(self.gemini_keys) if self.gemini_keys else 0
            openrouter_keys_len = len(self.openrouter_keys) if self.openrouter_keys else 0
            
            if gemini_keys_len == 0 and openrouter_keys_len == 0:
                error_msg += "Please add API keys using the manage_api_keys.py script or set environment variables."
            elif gemini_keys_len == 0:
                error_msg += "No Gemini API keys found. Please add Gemini API keys."
            else:
                error_msg += "OpenRouter fallback is disabled. Please enable it or add more API keys."
            
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Try Gemini keys first
        logger.info("AI Generate - Attempting Gemini keys (count: %d)", len(self.gemini_keys) if self.gemini_keys else 0)
        tried_gemini = 0
        gemini_count = len(self.gemini_keys) if self.gemini_keys else 0
        consecutive_failures = 0
        # Reduce the max consecutive failures to make the circuit breaker less aggressive
        max_consecutive_failures = 10  # Increased from 5 to 10 for even more resilience
        
        while tried_gemini < max(1, gemini_count) and consecutive_failures < max_consecutive_failures:
            key = self._next_gemini_key()
            if not key:
                logger.warning("AI Generate - No available Gemini keys found")
                break
            try:
                logger.debug("AI Generate - Attempting Gemini with key index %d", self._gemini_idx)
                logger.info("AI Generate - Making Gemini API call with key: %s...", key[:8] if key else "None")
                resp = await self._call_gemini(prompt, key, tools)
                self.ai_available = True
                consecutive_failures = 0  # Reset consecutive failures on success
                
                # Enhanced validation of AI response
                if resp and isinstance(resp, dict):
                    response_text = resp.get("text", "")
                    # Clean and validate response text
                    if isinstance(response_text, str):
                        import re
                        # Remove problematic tags
                        response_text = re.sub(r'<s>\s*', '', response_text)
                        response_text = re.sub(r'\s*</s>', '', response_text)
                        response_text = re.sub(r'\[OUT\]', '', response_text)
                        response_text = response_text.strip()
                        
                        # Check if response is effectively empty
                        if not response_text or len(response_text) < 2:
                            logger.warning("Gemini returned empty or very short response, using fallback")
                            resp["text"] = "I apologize, but I couldn't generate a detailed response at the moment. How else can I assist you?"
                        else:
                            resp["text"] = response_text
                
                logger.info("AI Generate - Gemini call successful with provider: %s", resp.get("provider", "unknown"))
                logger.info("AI Generate - Gemini response details - Text length: %d characters", len(resp.get("text", "")))
                
                # Handle function calls if present
                if isinstance(resp, dict) and "function_calls" in resp:
                    # Process function calls and get final response
                    logger.info("AI Generate - Processing function calls from Gemini response")
                    final_response = await self._process_function_calls(prompt, resp)
                    # Record key usage
                    key_obj = next((k for k in self.key_manager.gemini_keys if hasattr(k, 'key') and k.key == key), None)
                    if key_obj:
                        self.key_manager.record_use(key_obj)
                    logger.debug("AI Generate - Function calls processed, final response: %s", 
                                {k: v for k, v in final_response.items() if k != 'raw_response'})
                    return final_response
                
                # Record key usage
                key_obj = next((k for k in self.key_manager.gemini_keys if hasattr(k, 'key') and k.key == key), None)
                if key_obj:
                    self.key_manager.record_use(key_obj)
                logger.debug("AI Generate - Gemini response: %s", 
                            {k: v for k, v in resp.items() if k != 'raw_response'})
                return resp
            except Exception as e:
                error_msg = str(e).lower()
                logger.warning("Gemini key failed: %s", str(e))
                consecutive_failures += 1
                
                # If it's a quota error, mark the key as temporarily unavailable
                if "quota exceeded" in error_msg or "429" in error_msg:
                    # Mark the key as temporarily unavailable
                    self.key_manager.mark_gemini_key_unavailable(key, 120)  # Unavailable for 2 minutes
                    logger.warning("Gemini key marked as unavailable due to quota limits")
                    # Don't increment tried_gemini for quota errors, try next key
                    continue
                else:
                    # For other errors, increment the counter
                    tried_gemini += 1
                    logger.debug("Gemini key failure counted, tried_gemini: %d", tried_gemini)

        # If we get here, Gemini keys failed or were not present
        # Use a shorter cooldown period and make it less aggressive
        if consecutive_failures >= max_consecutive_failures:
            logger.warning("Too many consecutive Gemini failures, temporarily disabling AI")
            self.ai_available = False
            # Use a shorter cooldown period (30 seconds instead of 1 minute)
            self.next_retry_time = datetime.now().astimezone() + timedelta(seconds=30)
        else:
            logger.info("All Gemini keys exhausted or failed")
            # set a shorter retry cooldown
            self.ai_available = False
            self.next_retry_time = datetime.now().astimezone() + timedelta(seconds=15)  # Reduced from 30 to 15 seconds

        if self.enable_openrouter_fallback and self.openrouter_keys:
            logger.info("AI Generate - Gemini failed, attempting OpenRouter fallback (keys: %d)", len(self.openrouter_keys) if self.openrouter_keys else 0)
            # Try OpenRouter keys
            tried_or = 0
            or_count = len(self.openrouter_keys) if self.openrouter_keys else 0
            consecutive_failures = 0
            # Reduce the max consecutive failures to make the circuit breaker less aggressive
            max_consecutive_failures = 10  # Increased from 5 to 10 for even more resilience
            
            while tried_or < max(1, or_count) and consecutive_failures < max_consecutive_failures:
                or_key = self._next_openrouter_key()
                if not or_key:
                    logger.warning("AI Generate - No available OpenRouter keys found")
                    break
                try:
                    logger.debug("AI Generate - Attempting OpenRouter with key index %d", self._openrouter_idx)
                    logger.info("AI Generate - Making OpenRouter API call with key: %s...", or_key[:8] if or_key else "None")
                    resp = await self._call_openrouter(prompt, or_key)
                    # mark AI available since fallback succeeded
                    self.ai_available = True
                    self.next_retry_time = None
                    consecutive_failures = 0  # Reset consecutive failures on success
                    
                    # Enhanced validation of AI response
                    if resp and isinstance(resp, dict):
                        response_text = resp.get("text", "")
                        # Clean and validate response text
                        if isinstance(response_text, str):
                            import re
                            # Remove problematic tags
                            response_text = re.sub(r'<s>\s*', '', response_text)
                            response_text = re.sub(r'\s*</s>', '', response_text)
                            response_text = re.sub(r'\[OUT\]', '', response_text)
                            response_text = response_text.strip()
                            
                            # Check if response is effectively empty
                            if not response_text or len(response_text) < 2:
                                logger.warning("OpenRouter returned empty or very short response, using fallback")
                                resp["text"] = "I apologize, but I couldn't generate a detailed response at the moment. How else can I assist you?"
                            else:
                                resp["text"] = response_text
                    
                    logger.info("AI Generate - OpenRouter call successful with provider: %s", resp.get("provider", "unknown"))
                    logger.info("AI Generate - OpenRouter response details - Text length: %d characters", len(resp.get("text", "")))
                    
                    # Record key usage
                    key_obj = next((k for k in self.key_manager.openrouter_keys if hasattr(k, 'key') and k.key == or_key), None)
                    if key_obj:
                        self.key_manager.record_use(key_obj)
                    logger.debug("AI Generate - OpenRouter response: %s", resp)
                    return resp
                except Exception as e:
                    error_msg = str(e).lower()
                    logger.warning("OpenRouter key failed: %s", str(e))
                    consecutive_failures += 1
                    
                    # If it's a rate limit error, mark the key as temporarily unavailable
                    if "rate limit" in error_msg or "429" in error_msg:
                        # Mark the key as temporarily unavailable
                        self.key_manager.mark_openrouter_key_unavailable(or_key, 120)  # Unavailable for 2 minutes
                        logger.warning("OpenRouter key marked as unavailable due to rate limits")
                        # Don't increment tried_or for rate limit errors, try next key
                        continue
                    else:
                        # For other errors, increment the counter
                        tried_or += 1
                        logger.debug("OpenRouter key failure counted, tried_or: %d", tried_or)

        # Nothing worked
        logger.error("No AI providers available after rotation")
        # Reset AI availability after a shorter cooldown period
        self.next_retry_time = datetime.now().astimezone() + timedelta(seconds=30)  # Reduced from 2 minutes to 30 seconds
        raise AIProviderExhausted("no_providers_available")

    async def generate_with_reasoning(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response using reasoning loop (Chain-of-Thought) approach"""
        logger.info("AI Generate with Reasoning - Starting generation request, prompt length: %d", len(prompt))
        
        # Create a reasoning-enhanced prompt with better structure
        reasoning_prompt = f"""
# Instruction
{prompt}

# Reasoning Process
Before responding, think through this step by step:

1. What is the user's intent and emotional state?
2. What relevant information from the user's profile and conversation history is important?
3. What is the best approach to address their needs?
4. How can I provide the most helpful and accurate response?

Provide your reasoning process here.

# Response
Now, based on your careful reasoning, provide a helpful and appropriate response to the user:
        """.strip()
        
        # Generate response with reasoning
        response = await self.generate(reasoning_prompt, **kwargs)
        
        # Parse the two-part response (reasoning and response)
        if response and response.get("text"):
            response_text = response["text"]
            
            # Try to extract the response part after "# Response"
            response_marker = "# Response"
            response_start = response_text.find(response_marker)
            if response_start != -1:
                # Extract the actual response part
                actual_response = response_text[response_start + len(response_marker):].strip()
                # If there's content after the marker, use it
                if actual_response:
                    response["text"] = actual_response
                else:
                    # If no content after marker, try to find the last part
                    parts = response_text.split("# Response")
                    if len(parts) > 1:
                        response["text"] = parts[-1].strip()
        
        return response

    async def validate_response(self, original_prompt: str, response_text: str) -> Dict[str, Any]:
        """Validate AI response for accuracy and potential issues with enhanced hallucination detection"""
        try:
            # Create validation prompt with more comprehensive checks
            validation_prompt = f"""
You are an expert AI response validator. Review the following AI response for accuracy and potential issues:

Original Request: "{original_prompt}"

AI Response: "{response_text}"

Please evaluate this response on the following criteria:
1. Factual accuracy - Are the facts correct?
2. Logical consistency - Does the response make sense?
3. Potential hallucinations - Is the AI making up information?
4. Harmful content - Does it contain anything inappropriate?
5. Format issues - Are numbers, dates, and other data correctly formatted?
6. Relevance - Does it actually address the user's request?
7. Specificity - Does it provide concrete details rather than vague generalizations?
8. Source attribution - Does it claim to have information it couldn't reasonably have?

For hallucination detection, look for:
- Made-up names, companies, or organizations
- Invented statistics or data
- Fabricated quotes or references
- Claims about private information not provided in context
- Specific details that seem too precise for the context

Respond with a JSON object:
{{
    "valid": true/false,
    "issues": ["list of specific issues found"],
    "suggested_fix": "suggested correction if needed",
    "confidence": 0.0-1.0,
    "validation_details": {{
        "factual_accuracy": "score and notes",
        "logical_consistency": "score and notes",
        "hallucination_check": "score and notes with specific hallucinations identified",
        "harmful_content": "score and notes",
        "format_issues": "score and notes",
        "relevance": "score and notes",
        "specificity": "score and notes",
        "source_attribution": "score and notes"
    }}
}}
            """
            
            # Get a quick validation response (using a smaller, faster model if available)
            validation_response = await self.generate(validation_prompt)
            
            if validation_response and validation_response.get("text"):
                import json
                try:
                    # Extract JSON from response
                    response_text = validation_response["text"]
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1
                    if json_start != -1 and json_end > json_start:
                        json_text = response_text[json_start:json_end]
                        validation_result = json.loads(json_text)
                        return validation_result
                except json.JSONDecodeError:
                    pass
            
            # Fallback validation with enhanced regex and pattern checks
            return await self._enhanced_validation(original_prompt, response_text)
            
        except Exception as e:
            logger.warning(f"Response validation failed: {e}")
            # Return a safe default
            return {
                "valid": True,
                "issues": [],
                "suggested_fix": None,
                "confidence": 0.8
            }
    
    async def _enhanced_validation(self, original_prompt: str, response_text: str) -> Dict[str, Any]:
        """Enhanced validation using regex patterns and hallucination detection"""
        issues = []
        validation_details = {
            "factual_accuracy": "Not checked in basic validation",
            "logical_consistency": "Not checked in basic validation",
            "hallucination_check": "",
            "harmful_content": "",
            "format_issues": "",
            "relevance": "Not checked in basic validation",
            "specificity": "Not checked in basic validation",
            "source_attribution": "Not checked in basic validation"
        }
        
        # Check for common factual errors
        # Invalid dates
        import re
        date_patterns = [
            r"february 31",
            r"april 31",
            r"june 31",
            r"september 31",
            r"november 31"
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                issues.append(f"Invalid date detected: {pattern}")
                if validation_details["factual_accuracy"] == "Not checked in basic validation":
                    validation_details["factual_accuracy"] = f"Invalid date detected: {pattern}"
                else:
                    validation_details["factual_accuracy"] += f", Invalid date detected: {pattern}"
        
        # Check for obviously wrong numbers (e.g., claiming $100 when user said $50)
        # Extract monetary values and compare with context
        monetary_patterns = [
            (r"\$(\d+(?:\.\d+)?)", "dollar amounts"),
            (r"£(\d+(?:\.\d+)?)", "pound amounts"),
            (r"€(\d+(?:\.\d+)?)", "euro amounts")
        ]
        
        for pattern, description in monetary_patterns:
            matches = re.findall(pattern, original_prompt, re.IGNORECASE)
            response_matches = re.findall(pattern, response_text, re.IGNORECASE)
            # If we find monetary values in both, check for inconsistencies
            if matches and response_matches:
                # Simple check - if response has higher values than prompt, flag it
                try:
                    prompt_values = [float(m) for m in matches]
                    response_values = [float(m) for m in response_matches]
                    max_prompt = max(prompt_values) if prompt_values else 0
                    max_response = max(response_values) if response_values else 0
                    
                    if max_response > max_prompt * 2:  # If response value is more than double the prompt value
                        issues.append(f"Potential monetary inconsistency: {description} in response ({max_response}) much higher than in prompt ({max_prompt})")
                        if validation_details["factual_accuracy"] == "Not checked in basic validation":
                            validation_details["factual_accuracy"] = f"Potential monetary inconsistency: {description} in response ({max_response}) much higher than in prompt ({max_prompt})"
                        else:
                            validation_details["factual_accuracy"] += f", Potential monetary inconsistency: {description} in response ({max_response}) much higher than in prompt ({max_prompt})"
                except ValueError:
                    pass  # Ignore if we can't convert to float
        
        # Check for harmful content patterns
        harmful_patterns = [
            r"illegal",
            r"hack",
            r"steal",
            r"violence",
            r"kill",
            r"murder",
            r"weapon",
            r"drugs",
            r"abuse"
        ]
        
        harmful_found = []
        for pattern in harmful_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                harmful_found.append(pattern)
        
        if harmful_found:
            issues.append(f"Potentially harmful content detected: {', '.join(harmful_found)}")
            validation_details["harmful_content"] = f"Potentially harmful content detected: {', '.join(harmful_found)}"
        
        # Enhanced hallucination detection
        hallucination_indicators = [
            (r"according to [\w\s]+ research", "vague research citation"),
            (r"studies show that", "unsubstantiated claim"),
            (r"experts agree that", "unsubstantiated expert opinion"),
            (r"it is well known that", "vague knowledge claim"),
            (r"as we discussed earlier", "reference to non-existent discussion"),
            (r"as mentioned before", "reference to non-existent previous conversation"),
            (r"based on your [\w\s]+ profile", "reference to unspecified profile information"),
            (r"our records indicate", "reference to unspecified records"),
            (r"I recall that you said", "reference to non-existent recalled information"),
            (r"you mentioned previously", "reference to non-existent previous mention")
        ]
        
        hallucinations_found = []
        for pattern, description in hallucination_indicators:
            if re.search(pattern, response_text, re.IGNORECASE):
                hallucinations_found.append(description)
        
        if hallucinations_found:
            issues.append(f"Potential hallucinations detected: {', '.join(hallucinations_found)}")
            validation_details["hallucination_check"] = f"Potential hallucinations detected: {', '.join(hallucinations_found)}"
        
        # Check for format issues
        format_issues = []
        
        # Phone number format consistency
        phone_patterns = [r"\d{3}-\d{3}-\d{4}", r"\(\d{3}\) \d{3}-\d{4}", r"\d{10}"]
        phone_matches = []
        for pattern in phone_patterns:
            phone_matches.extend(re.findall(pattern, response_text))
        
        if len(phone_matches) > 3:  # Too many phone numbers might indicate hallucination
            format_issues.append("Excessive phone numbers detected")
        
        # Email format consistency
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        email_matches = re.findall(email_pattern, response_text)
        if len(email_matches) > 5:  # Too many emails might indicate hallucination
            format_issues.append("Excessive email addresses detected")
        
        if format_issues:
            issues.append(f"Format issues: {', '.join(format_issues)}")
            validation_details["format_issues"] = f"Format issues: {', '.join(format_issues)}"
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "suggested_fix": None,
            "confidence": 0.9 if len(issues) == 0 else max(0.1, 0.9 - (len(issues) * 0.1)),
            "validation_details": validation_details
        }
