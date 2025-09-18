"""
Fixed implementation of Google's generativeai model wrapper.
This is a local re-implementation that addresses several bugs in the original.
"""

from typing import Optional, List, Any, Dict, Union
import logging

logger = logging.getLogger(__name__)

# Mock classes to represent the Google Generative AI types
class glm:
    class GenerateContentRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class Candidate:
        class FinishReason:
            FINISH_REASON_UNSPECIFIED = 0
            STOP = 1
            MAX_TOKENS = 2

class content_types:
    @staticmethod
    def to_contents(contents):
        # Simplified implementation
        return contents if contents else []
    
    @staticmethod
    def to_tools(tools):
        # Simplified implementation
        return tools

class generation_types:
    @staticmethod
    def to_generation_config_dict(config):
        # Simplified implementation
        return config if config else {}
    
    class BlockedPromptException(Exception):
        pass
    
    class BrokenResponseError(Exception):
        pass
    
    class StopCandidateException(Exception):
        pass

class safety_types:
    @staticmethod
    def to_easy_safety_dict(settings, harm_category_set):
        # Simplified implementation
        return settings if settings else {}
    
    @staticmethod
    def normalize_safety_settings(settings, harm_category_set):
        # Simplified implementation
        return settings

class GenerationConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class GenerativeModel:
    """Fixed GenerativeModel implementation that addresses several bugs."""
    
    def __init__(self, model_name: str, generation_config=None, safety_settings=None, tools=None):
        self._model_name = model_name
        self._generation_config = generation_types.to_generation_config_dict(generation_config or {})
        self._safety_settings = safety_types.to_easy_safety_dict(safety_settings, harm_category_set="new")
        self._tools = content_types.to_tools(tools) if tools is not None else []
        self._history = []
        self._last_sent = None
        self._last_received = None
        self._MODEL_ROLE = "model"
    
    def _prepare_request(
        self,
        *,
        contents: Any,
        generation_config: Any | None = None,
        safety_settings: Any | None = None,
        **kwargs,
    ) -> glm.GenerateContentRequest:
        """Creates a `glm.GenerateContentRequest` from raw inputs."""
        if not contents:
            raise TypeError("contents must not be empty")

        contents = content_types.to_contents(contents)

        generation_config_dict = generation_types.to_generation_config_dict(generation_config)
        merged_gc = self._generation_config.copy()
        if generation_config_dict:
            merged_gc.update(generation_config_dict)

        safety_settings_dict = safety_types.to_easy_safety_dict(safety_settings, harm_category_set="new")
        merged_ss = self._safety_settings.copy()
        if safety_settings_dict:
            merged_ss.update(safety_settings_dict)
        merged_ss = safety_types.normalize_safety_settings(merged_ss, harm_category_set="new")

        # Defensive: allow callers to override tools via kwargs, but avoid passing tools twice.
        supplied_tools = kwargs.pop("tools", None)
        tools_to_use = content_types.to_tools(supplied_tools) if supplied_tools is not None else self._tools

        # Remove any accidental duplicate keys that glm.GenerateContentRequest may not accept
        # (this also prevents "multiple values for keyword argument 'tools'" errors).
        return glm.GenerateContentRequest(
            model=self._model_name,
            contents=contents,
            generation_config=merged_gc,
            safety_settings=merged_ss,
            tools=tools_to_use,
            **kwargs,
        )
    
    def generate_content(self, contents, generation_config=None, safety_settings=None, stream=False, tools=None, **kwargs):
        """Generate content with fixed implementation."""
        # Use our fixed _prepare_request method
        request = self._prepare_request(
            contents=contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            **kwargs
        )
        
        # Simulate API call
        # In a real implementation, this would make an actual API call
        response = self._mock_generate_content(request)
        return response
    
    def _mock_generate_content(self, request):
        """Mock implementation of the API call."""
        # This is just a placeholder for demonstration
        class MockResponse:
            def __init__(self):
                self.candidates = [
                    type('Candidate', (), {
                        'content': type('Content', (), {
                            'role': 'model',
                            'parts': [{'text': 'Mock response'}]
                        })(),
                        'finish_reason': glm.Candidate.FinishReason.STOP
                    })()
                ]
                self.prompt_feedback = None
        
        return MockResponse()
    
    @property
    def history(self) -> list:
        """The chat history."""
        last = self._last_received
        if last is None:
            return self._history

        candidates = getattr(last, "candidates", None) or []
        if not candidates:
            raise generation_types.BrokenResponseError("Broken response: no candidates")

        # check finish_reason safely
        finish_reason = getattr(candidates[0], "finish_reason", None)
        if finish_reason not in (
            glm.Candidate.FinishReason.FINISH_REASON_UNSPECIFIED,
            glm.Candidate.FinishReason.STOP,
            glm.Candidate.FinishReason.MAX_TOKENS,
        ):
            error = generation_types.StopCandidateException(candidates[0])
            # safely attach error if possible
            try:
                setattr(last, "_error", error)
            except Exception:
                pass

        last_error = getattr(last, "_error", None)
        if last_error is not None:
            raise generation_types.BrokenResponseError(
                "Can not build a coherent chat history after a broken streaming response. "
                "Inspect `chat.last` or call `chat.rewind()` to remove the last request/response."
            ) from last_error

        sent = self._last_sent
        received = candidates[0].content
        if not getattr(received, "role", None):
            received.role = self._MODEL_ROLE
        self._history.extend([sent, received])

        self._last_sent = None
        self._last_received = None

        return self._history
    
    @history.setter
    def history(self, history):
        self._history = content_types.to_contents(history)
        # fix typo: reset the last sent/received tracking properly
        self._last_sent = None
        self._last_received = None
    
    def send_message(self, message, generation_config=None, safety_settings=None, stream=False):
        """Send a message with fixed implementation."""
        # Prepare contents
        contents = content_types.to_contents([message])
        
        # Generate response
        response = self.generate_content(
            contents=contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=stream
        )
        
        # safe prompt feedback check
        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback is not None and getattr(prompt_feedback, "block_reason", None):
            raise generation_types.BlockedPromptException(prompt_feedback)

        if not stream:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                raise generation_types.BrokenResponseError("No candidates in response")
            finish_reason = getattr(candidates[0], "finish_reason", None)
            if finish_reason not in (
                glm.Candidate.FinishReason.FINISH_REASON_UNSPECIFIED,
                glm.Candidate.FinishReason.STOP,
                glm.Candidate.FinishReason.MAX_TOKENS,
            ):
                raise generation_types.StopCandidateException(candidates[0])
        
        # Update history tracking
        self._last_sent = contents
        self._last_received = response
        
        return response
    
    def send_message_async(self, message, generation_config=None, safety_settings=None, stream=False):
        """Async version of send_message with fixed implementation."""
        # This would be an async implementation in reality
        return self.send_message(message, generation_config, safety_settings, stream)
    
    def rewind(self) -> tuple:
        """Removes the last request/response pair from the chat history."""
        if self._last_received is None:
            # ensure there are at least two items to pop
            if len(self._history) < 2:
                raise IndexError("Not enough history to rewind")
            last_request = self._history.pop(-2)
            last_response = self._history.pop()
            return last_request, last_response
        else:
            # streaming case: last_sent and last_received are buffered
            if self._last_sent is None or self._last_received is None:
                raise IndexError("No buffered last request/response to rewind")
            candidate_content = getattr(self._last_received.candidates[0], 'content', None)
            result = self._last_sent, candidate_content
            self._last_sent = None
            self._last_received = None
            return result

class ChatSession:
    """Fixed ChatSession implementation that addresses several bugs."""
    
    def __init__(self, model: GenerativeModel, history=None):
        self.model = model
        if history:
            self.model.history = history
    
    def send_message(self, content, generation_config=None, safety_settings=None, stream=False):
        """Send a message in the chat session."""
        return self.model.send_message(
            message=content,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=stream
        )
    
    @property
    def history(self):
        """Get the chat history."""
        return self.model.history
    
    def rewind(self):
        """Rewind the chat history."""
        return self.model.rewind()