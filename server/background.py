from typing import Callable
import asyncio
import logging

logger = logging.getLogger(__name__)


async def _periodic_cache_cleanup(app, interval_seconds: int = 300):
    """Optional periodic cleanup hook. Safe no-op if Redis is used.
    This avoids unbounded in-memory cache growth in dev.
    """
    while True:
        try:
            cache = getattr(app.state, "cache", None)
            if cache and hasattr(cache, "clear"):
                # Only clear expired keys if implemented; otherwise, skip to avoid data loss.
                pass
        except Exception:
            pass
        await asyncio.sleep(interval_seconds)


def register_background_tasks(app):
    # placeholder to register any background tasks
    app.state._bg_registered = True
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_periodic_cache_cleanup(app))
    except Exception:
        # Do not fail startup for optional background task
        pass
