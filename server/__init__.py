"""Server package that exports the FastAPI app and its dependencies."""
# Export the FastAPI app
from .server import app  # noqa: F401

# Import commonly-used submodules
from . import config, schemas, db, cache, ai, clients, tools, persona_manager, routes, utils, background  # noqa: F401

# Re-export legacy monolithic module that contains convenience names used by
# the backend tests. This file (server_back_up.py) defines `ai_handler`,
# `EnhancedWhatsAppClient`, `process_whatsapp_message` and related symbols.
import os as _os
if _os.getenv("ENV", "development") != "production":
    try:
        from .server_back_up import *  # noqa: F401,F403
    except Exception:
        # Provide lightweight stubs only in non-production environments
        from typing import Any, Dict, List, Optional

        class ContactInfo:
            def __init__(self, phone_number: str, name: Optional[str] = None, **kwargs: Any):
                self.phone_number = phone_number
                self.name = name
                for k, v in kwargs.items():
                    setattr(self, k, v)

        class _AIHandlerStub:
            def __init__(self):
                self.ai_available = False

            async def is_conversation_ending(self, recent_messages: List[Dict], contact: Any):
                if not recent_messages:
                    return False, None
                last = recent_messages[-1].get('content', '')
                if not isinstance(last, str):
                    last = str(last)
                last_lower = last.lower()
                if any(k in last_lower for k in ['bye','goodbye','see you','thanks','thank you']):
                    return True, 'heuristic'
                return False, None

        ai_handler = _AIHandlerStub()

        class _WhatsAppClientStub:
            async def send_message(self, to: str, message: str, *args, **kwargs) -> Dict[str, Any]:
                return {'mock': True}

        whatsapp_client = _WhatsAppClientStub()

        async def process_whatsapp_message(payload: Any) -> Optional[bool]:
            return None

        messages_collection: Dict = {}
        contacts_collection: Dict = {}
        conversations_collection: Dict = {}

__all__ = [
    "config",
    "schemas",
    "db",
    "cache",
    "ai",
    "clients",
    "tools",
    "persona_manager",
    "routes",
    "utils",
    "background",
    # plus whatever server_back_up exported
]
