"""Lightweight backend shim that re-exports the top-level `server` compatibility
package for tests run from the `agent/whats-app-agent/backend` directory.

This module purposefully avoids importing or executing the large monolithic
server implementation (which pulls in heavy dependencies like pydantic,
sklearn, google SDKs, etc). Instead it imports the repo-level `server`
package which already handles compatibility between the monolith and the
modular refactor.
"""
import os
import sys
from importlib import import_module

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    # Attempt to import the repo-level 'server' package by loading its
    # __init__.py directly and registering it under the package name 'server'.
    # This ensures imports like `import server.ai.handler` work for tests
    # running from the backend directory where a local `server.py` would
    # otherwise shadow the package.
    repo_init = os.path.normpath(os.path.join(ROOT, 'server', '__init__.py'))
    if os.path.exists(repo_init):
        import importlib.util
        spec = importlib.util.spec_from_file_location('server', repo_init)
        server_pkg = importlib.util.module_from_spec(spec)
        # ensure the module has a package __path__ so submodules can be found
        server_pkg.__package__ = 'server'
        server_pkg.__path__ = [os.path.normpath(os.path.join(ROOT, 'server'))]
        # Execute the package module
        spec.loader.exec_module(server_pkg)  # type: ignore
        # Register in sys.modules under the canonical name 'server'
        sys.modules['server'] = server_pkg
        # Export public names into this shim's globals for direct imports
        for _k, _v in server_pkg.__dict__.items():
            if not _k.startswith('_') and _k not in globals():
                globals()[_k] = _v
    else:
        # Fallback to normal import (best-effort)
        _server = import_module('server')
        for _k, _v in _server.__dict__.items():
            if not _k.startswith('_') and _k not in globals():
                globals()[_k] = _v
except Exception:
    # Best-effort: leave this shim minimal so tests can import `server`.
    pass

# Provide minimal fallbacks for legacy tests that import symbols from `server`.
import importlib.util

def _load_attr_from_path(path: str, attr: str):
    if not os.path.exists(path):
        return None
    try:
        spec = importlib.util.spec_from_file_location('tmp_mod', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return getattr(mod, attr, None)
    except Exception:
        return None

# whatsapp_client fallback: try to load EnhancedWhatsAppClient from the agent server
if 'whatsapp_client' not in globals():
    wa_path = os.path.normpath(os.path.join(ROOT, 'agent', 'whats-app-agent', 'server', 'clients', 'whatsapp.py'))
    _WAC = _load_attr_from_path(wa_path, 'EnhancedWhatsAppClient')
    if _WAC is None:
        # Try loading from repo-level server package if available
        try:
            from server.clients.whatsapp import EnhancedWhatsAppClient as _WAC2
            _WAC = _WAC2
        except Exception:
            _WAC = None

    if _WAC:
        try:
            whatsapp_client = _WAC()
            globals()['whatsapp_client'] = whatsapp_client
        except Exception:
            # Provide a simple mutable dummy object so tests can monkeypatch methods
            class _DummyClient:
                pass
            whatsapp_client = _DummyClient()
            globals()['whatsapp_client'] = whatsapp_client

# process_whatsapp_message fallback: try to import from repo-level package
if 'process_whatsapp_message' not in globals():
    try:
        from server import process_whatsapp_message as _pm  # type: ignore
        globals()['process_whatsapp_message'] = _pm
    except Exception:
        # Define a lightweight async wrapper that returns False if not implemented
        async def process_whatsapp_message(payload):
            return False
        globals()['process_whatsapp_message'] = process_whatsapp_message

# Ensure collection placeholders exist for tests
for _col in ('messages_collection', 'contacts_collection', 'conversations_collection'):
    if _col not in globals():
        globals()[_col] = None
