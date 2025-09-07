import redis.asyncio as aioredis
from typing import Any, Optional
import logging
import time
from .config import config

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        self._cache = {}
        self._redis: Optional[aioredis.Redis] = None
        
    async def initialize(self):
        """Initialize Redis connection if configured"""
        if config.REDIS_URL:
            try:
                self._redis = await aioredis.from_url(
                    config.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self._redis.ping()
                logger.info("✅ Redis connected successfully")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}. Using memory cache.")
                self._redis = None

    async def close(self):
        """Close Redis connection if active"""
        if self._redis:
            await self._redis.close()
            logger.info("✅ Redis connection closed")

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with expiry enforcement for in-memory store."""
        if self._redis:
            return await self._redis.get(key) or default
        # Enforce expiry for in-memory cache
        expires_at = self._cache.get(f"{key}:expires")
        if expires_at is not None and time.time() > float(expires_at):
            self._cache.pop(key, None)
            self._cache.pop(f"{key}:expires", None)
            return default
        return self._cache.get(key, default)

    async def set(self, key: str, value: Any, expire: int = None) -> None:
        """Set value in cache with optional expiration in seconds"""
        if self._redis:
            if expire:
                await self._redis.setex(key, expire, value)
            else:
                await self._redis.set(key, value)
        else:
            self._cache[key] = value
            if expire:
                # Implement simple expiration for memory cache
                expiry_time = time.time() + expire
                self._cache[f"{key}:expires"] = expiry_time

    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        if self._redis:
            await self._redis.delete(key)
        else:
            self._cache.pop(key, None)
            self._cache.pop(f"{key}:expires", None)

    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment value in cache"""
        if self._redis:
            return await self._redis.incr(key, amount)
        else:
            current = int(self._cache.get(key, 0))
            new_value = current + amount
            self._cache[key] = new_value
            return new_value

    async def expire(self, key: str, seconds: int) -> None:
        """Set expiration for key"""
        if self._redis:
            await self._redis.expire(key, seconds)
        else:
            expiry_time = time.time() + seconds
            self._cache[f"{key}:expires"] = expiry_time

    async def clear(self) -> None:
        """Clear all cache entries"""
        if self._redis:
            await self._redis.flushdb()
        else:
            self._cache.clear()


# Global cache instance
cache_manager = CacheManager()
