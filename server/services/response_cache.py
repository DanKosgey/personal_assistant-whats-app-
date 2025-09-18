import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib
import json
import asyncio

logger = logging.getLogger(__name__)

class ResponseCache:
    """Intelligent caching system for AI responses with TTL"""
    
    def __init__(self, ttl_minutes: int = 5):
        self.cache = {}  # In-memory cache
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_size = 1000  # Maximum cache size
        self.access_count = {}  # Track access frequency for LRU
        self.lock = asyncio.Lock()  # Thread safety for concurrent access
        
    def _generate_key(self, prompt: str, user_id: str) -> str:
        """Generate a cache key from prompt and user ID"""
        key_data = f"{user_id}:{prompt}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def get(self, prompt: str, user_id: str) -> Optional[str]:
        """Get cached response if available and not expired"""
        key = self._generate_key(prompt, user_id)
        
        if key in self.cache:
            cached_item = self.cache[key]
            # Check if expired
            if datetime.now() - cached_item['timestamp'] < self.ttl:
                # Update access count for LRU
                self.access_count[key] = self.access_count.get(key, 0) + 1
                logger.debug(f"Cache hit for key: {key[:8]}...")
                return cached_item['response']
            else:
                # Remove expired item
                del self.cache[key]
                if key in self.access_count:
                    del self.access_count[key]
                logger.debug(f"Removed expired cache entry: {key[:8]}...")
                
        logger.debug(f"Cache miss for key: {key[:8]}...")
        return None
        
    def set(self, prompt: str, user_id: str, response: str) -> None:
        """Store response in cache"""
        # Check cache size and clean if needed
        if len(self.cache) >= self.max_size:
            self._cleanup()
            
        key = self._generate_key(prompt, user_id)
        self.cache[key] = {
            'response': response,
            'timestamp': datetime.now(),
            'user_id': user_id
        }
        logger.debug(f"Cached response for key: {key[:8]}...")
        
    def _cleanup(self) -> None:
        """Remove expired entries and reduce cache size if needed"""
        current_time = datetime.now()
        expired_keys = [
            key for key, item in self.cache.items()
            if current_time - item['timestamp'] >= self.ttl
        ]
        
        # Remove expired entries
        for key in expired_keys:
            del self.cache[key]
            if key in self.access_count:
                del self.access_count[key]
            
        # If still too large, remove least recently used entries
        if len(self.cache) >= self.max_size:
            # Sort by access count (ascending) and remove least used 20%
            sorted_items = sorted(self.access_count.items(), key=lambda x: x[1])
            remove_count = int(len(sorted_items) * 0.2)
            for i in range(remove_count):
                key_to_remove = sorted_items[i][0]
                if key_to_remove in self.cache:
                    del self.cache[key_to_remove]
                if key_to_remove in self.access_count:
                    del self.access_count[key_to_remove]
                
        logger.info(f"Cache cleanup: removed {len(expired_keys)} expired entries")
        
    def invalidate_user_cache(self, user_id: str) -> None:
        """Remove all cache entries for a specific user"""
        keys_to_remove = [
            key for key, item in self.cache.items()
            if item['user_id'] == user_id
        ]
        for key in keys_to_remove:
            del self.cache[key]
            if key in self.access_count:
                del self.access_count[key]
        logger.info(f"Invalidated cache for user {user_id}: {len(keys_to_remove)} entries removed")
        
    def clear(self) -> None:
        """Clear entire cache"""
        self.cache.clear()
        self.access_count.clear()
        logger.info("Cache cleared")
        
    async def get_async(self, prompt: str, user_id: str) -> Optional[str]:
        """Async version of get with thread safety"""
        async with self.lock:
            return self.get(prompt, user_id)
            
    async def set_async(self, prompt: str, user_id: str, response: str) -> None:
        """Async version of set with thread safety"""
        async with self.lock:
            self.set(prompt, user_id, response)