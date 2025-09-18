import json
import os
import threading
from pathlib import Path
from typing import Dict

CTX_PATH = Path(__file__).parent / 'context.json'
_CTX_CACHE: Dict[str, str] = {}
_CTX_MTIME: float = -1.0
_CTX_LOCK = threading.Lock()


def load_context() -> Dict[str, str]:
    """Load assistant/company context with simple in-process caching.

    Uses file mtime to invalidate cache automatically when `context.json` changes.
    """
    try:
        mtime = os.path.getmtime(CTX_PATH)
    except Exception:
        mtime = -1.0

    with _CTX_LOCK:
        global _CTX_MTIME, _CTX_CACHE
        if mtime == _CTX_MTIME and _CTX_CACHE:
            return _CTX_CACHE

        try:
            with CTX_PATH.open('r', encoding='utf-8') as f:
                data = json.load(f)
                # Normalize to str for stability
                _CTX_CACHE = {k: str(v) for k, v in data.items()}
                _CTX_MTIME = mtime
                return _CTX_CACHE
        except Exception:
            # On failure, return previous cache if any, else {}
            return _CTX_CACHE or {}
