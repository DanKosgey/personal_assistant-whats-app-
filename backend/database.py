"""Database manager implementation with proper typing."""

from typing import Dict, Optional, Any
from pymongo.mongo_client import MongoClient
from pymongo.database import Database
import redis.asyncio as aioredis
import logging
from db_utils import CollectionAccessor

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Enhanced database manager with proper type safety."""
    
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self._collections: Dict[str, CollectionAccessor] = {}
        self.redis_client: Optional[aioredis.Redis] = None
    
    async def initialize(self, mongo_url: str, database_name: str, redis_url: Optional[str] = None):
        """Initialize database connections with proper error handling."""
        try:
            # MongoDB connection
            self.client = MongoClient(
                mongo_url,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                maxPoolSize=100,
                retryWrites=True
            )
            
            # Test connection
            await self._test_mongo_connection()
            self.db = self.client[database_name]
            
            # Initialize collections
            await self._init_collections()
            logger.info("✅ MongoDB connected successfully")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise
        
        if redis_url:
            try:
                # Redis connection
                self.redis_client = await aioredis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=20
                )
                
                # Test Redis connection
                await self.redis_client.ping()
                logger.info("✅ Redis connected successfully")
                
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}. Falling back to memory cache.")
                self.redis_client = None
    
    async def _test_mongo_connection(self):
        """Test MongoDB connection."""
        if not self.client:
            raise RuntimeError("MongoDB client not initialized")
            
        from server import asyncio
        await asyncio.to_thread(self.client.admin.command, 'ping')
    
    async def _init_collections(self):
        """Initialize MongoDB collections with proper typing."""
        if not self.db:
            raise RuntimeError("Database not initialized")
            
        collection_names = [
            'conversations',
            'contacts',
            'messages',
            'analytics',
            'feedback',
            'templates',
            'audit_log'
        ]
        
        for name in collection_names:
            collection = self.db[name]
            self._collections[name] = CollectionAccessor(collection)
            
        await self._create_indexes()
    
    async def _create_indexes(self):
        """Create optimized indexes."""
        from server import asyncio
        
        try:
            # Messages indexes
            messages = self._collections.get('messages')
            if messages and messages.is_available:
                await asyncio.to_thread(
                    messages._collection.create_index,  # type: ignore
                    [("conversation_id", 1), ("timestamp", -1)]
                )
                await asyncio.to_thread(
                    messages._collection.create_index,  # type: ignore
                    [("phone_number", 1), ("timestamp", -1)]
                )
            
            # Conversations indexes
            conversations = self._collections.get('conversations')
            if conversations and conversations.is_available:
                await asyncio.to_thread(
                    conversations._collection.create_index,  # type: ignore
                    [("phone_number", 1), ("start_time", -1)]
                )
            
            # Contacts indexes
            contacts = self._collections.get('contacts')
            if contacts and contacts.is_available:
                await asyncio.to_thread(
                    contacts._collection.create_index,  # type: ignore
                    [("phone_number", 1)],
                    unique=True
                )
            
            logger.info("✅ Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating indexes: {e}")
    
    def get_collection(self, name: str) -> Optional[CollectionAccessor]:
        """Safely get a collection accessor."""
        return self._collections.get(name)
    
    def has_collection(self, name: str) -> bool:
        """Check if a collection exists and is available."""
        accessor = self._collections.get(name)
        return bool(accessor and accessor.is_available)
    
    async def close(self):
        """Close all database connections."""
        if self.client:
            self.client.close()
            self.client = None
            
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
            
        self._collections.clear()
        logger.info("✅ Database connections closed")
