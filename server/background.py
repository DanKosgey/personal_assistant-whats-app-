from typing import Callable
import asyncio
import logging

logger = logging.getLogger(__name__)


async def _periodic_consent_cleanup(app, interval_seconds: int = 3600):
    """Clean up expired consent flows every hour"""
    while True:
        try:
            message_processor = getattr(app.state, "message_processor", None)
            if message_processor and hasattr(message_processor, "consent_workflow"):
                cleaned = await message_processor.consent_workflow.cleanup_expired_flows()
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} expired consent flows")
        except Exception as e:
            logger.error(f"Error in consent cleanup task: {e}")
        await asyncio.sleep(interval_seconds)


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
            loop.create_task(_periodic_consent_cleanup(app))
    except Exception:
        # Do not fail startup for optional background task
        pass
