"""Database manager with Motor async client when available, otherwise fall back to pymongo with async wrappers."""
from typing import Any, Optional, Dict, List, Union
import logging
import asyncio
from contextlib import asynccontextmanager
from .config import config

logger = logging.getLogger(__name__)

# Check for Motor availability
try:
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
    _HAS_MOTOR = True
except ImportError:
    _HAS_MOTOR = False
    logger.info("Motor not available, will use pymongo with async wrappers")

# Import pymongo exceptions for error handling
try:
    from pymongo.errors import DuplicateKeyError, ServerSelectionTimeoutError, ConnectionFailure
    import pymongo
except ImportError:
    # Fallback in case pymongo is not available
    class DuplicateKeyError(Exception):
        pass
    class ServerSelectionTimeoutError(Exception):
        pass
    class ConnectionFailure(Exception):
        pass


class DatabaseManager:
    """
    Unified database manager that works with Motor (async) when available,
    or falls back to pymongo with asyncio.to_thread wrappers.
    """
    
    def __init__(self):
        self.uri = config.MONGO_URL
        self.db_name = config.MONGO_DB_NAME
        self.client: Optional[Union[AsyncIOMotorClient, pymongo.MongoClient]] = None
        self.db: Optional[Union[AsyncIOMotorDatabase, pymongo.database.Database]] = None
        self._using_motor = _HAS_MOTOR
        self._connected = False

    async def connect(self) -> None:
        """Establish database connection and setup indexes."""
        if self._connected:
            logger.warning("Database already connected")
            return

        try:
            if self._using_motor:
                await self._connect_motor()
            else:
                await self._connect_pymongo()
            
            await self._setup_indexes()
            self._connected = True
            logger.info(f"✅ MongoDB connected ({'motor' if self._using_motor else 'pymongo'})")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self._connected = False
            raise

    async def _connect_motor(self) -> None:
        """Connect using Motor async client."""
        self.client = AsyncIOMotorClient(self.uri, serverSelectionTimeoutMS=5000)
        # Test connection
        await self.client.admin.command('ping')
        self.db = self.client[self.db_name]

    async def _connect_pymongo(self) -> None:
        """Connect using pymongo with async wrapper."""
        def _sync_connect():
            client = pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            # Test connection
            client.admin.command('ping')
            return client, client[self.db_name]
        
        self.client, self.db = await asyncio.to_thread(_sync_connect)

    async def close(self) -> None:
        """Close database connection."""
        if not self._connected or not self.client:
            return

        try:
            if self._using_motor:
                self.client.close()
            else:
                await asyncio.to_thread(self.client.close)
            
            logger.info(f"✅ MongoDB connection closed ({'motor' if self._using_motor else 'pymongo'})")
            
        except Exception as e:
            logger.warning(f"Error closing database connection: {e}")
        finally:
            self._connected = False
            self.client = None
            self.db = None

    async def _setup_indexes(self) -> None:
        """Setup database indexes for optimal performance."""
        indexes_config = [
            # Conversations collection indexes
            ("conversations", [("phone_number", 1)]),
            ("conversations", [("status", 1)]),
            ("conversations", [("last_message_time", -1)]),
            
            # Messages collection indexes
            ("messages", [("conversation_id", 1)]),
            ("messages", [("timestamp", -1)]),
        ]

        try:
            for collection_name, index_spec in indexes_config:
                await self._create_index(collection_name, index_spec)

            # Handle unique index separately due to potential duplicates
            await self._create_unique_message_id_index()
            
            logger.info("✅ Database indexes processed")
            
        except Exception as e:
            logger.warning(f"Could not create all indexes: {e}")

    async def _create_index(self, collection_name: str, index_spec: List[tuple]) -> None:
        """Create a single index on specified collection."""
        if self._using_motor:
            await self.db[collection_name].create_index(index_spec)
        else:
            await asyncio.to_thread(
                lambda: self.db[collection_name].create_index(index_spec)
            )

    async def _create_unique_message_id_index(self) -> None:
        """Create unique index on message_id, handling existing duplicates gracefully."""
        try:
            # First, try to remove duplicates if they exist
            await self._remove_duplicate_message_ids()
            
            # Then create the unique index
            if self._using_motor:
                await self.db.messages.create_index([("message_id", 1)], unique=True)
            else:
                await asyncio.to_thread(
                    lambda: self.db.messages.create_index([("message_id", 1)], unique=True)
                )
            logger.info("✅ Unique index on message_id created successfully")
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['duplicate', 'e11000', 'duplicatekey']):
                logger.warning("Could not create unique index on message_id: duplicate keys present")
            else:
                logger.warning(f"Could not create unique index on message_id: {e}")

    async def _remove_duplicate_message_ids(self) -> None:
        """Remove duplicate message_id entries, keeping the most recent one."""
        try:
            # Find duplicate message_ids using a more compatible approach
            if self._using_motor:
                pipeline = [
                    {"$group": {
                        "_id": "$message_id",
                        "count": {"$sum": 1},
                        "ids": {"$push": "$_id"},
                        "timestamps": {"$push": "$timestamp"}
                    }},
                    {"$match": {"count": {"$gt": 1}}}
                ]
                
                cursor = self.db.messages.aggregate(pipeline)
                duplicates = await cursor.to_list(length=None)
            else:
                def _find_duplicates():
                    pipeline = [
                        {"$group": {
                            "_id": "$message_id",
                            "count": {"$sum": 1},
                            "ids": {"$push": "$_id"},
                            "timestamps": {"$push": "$timestamp"}
                        }},
                        {"$match": {"count": {"$gt": 1}}}
                    ]
                    return list(self.db.messages.aggregate(pipeline))
                
                duplicates = await asyncio.to_thread(_find_duplicates)
            
            # Remove duplicate documents, keeping the most recent one
            removed_count = 0
            for group in duplicates:
                message_ids = group["ids"]
                timestamps = group["timestamps"]
                
                # Pair ids with timestamps and sort by timestamp (descending)
                id_timestamp_pairs = list(zip(message_ids, timestamps))
                id_timestamp_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Keep the first (most recent) and remove the rest
                ids_to_remove = [pair[0] for pair in id_timestamp_pairs[1:]]
                
                if ids_to_remove:
                    if self._using_motor:
                        result = await self.db.messages.delete_many({"_id": {"$in": ids_to_remove}})
                    else:
                        result = await asyncio.to_thread(
                            lambda: self.db.messages.delete_many({"_id": {"$in": ids_to_remove}})
                        )
                    removed_count += result.deleted_count
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} duplicate message entries")
                
        except Exception as e:
            # Log the error but don't fail - this is a best-effort operation
            logger.debug(f"Could not remove duplicate messages: {e}")

    def _ensure_connected(self) -> None:
        """Ensure database is connected before operations."""
        if not self._connected:
            raise RuntimeError("Database not connected. Call connect() first.")

    # CRUD Operations
    
    async def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get a conversation by ID."""
        self._ensure_connected()
        
        if self._using_motor:
            return await self.db.conversations.find_one({"_id": conversation_id})
        else:
            return await asyncio.to_thread(
                lambda: self.db.conversations.find_one({"_id": conversation_id})
            )

    async def create_conversation(self, conversation: Dict) -> str:
        """Create a new conversation and return its ID."""
        self._ensure_connected()
        
        if self._using_motor:
            result = await self.db.conversations.insert_one(conversation)
        else:
            result = await asyncio.to_thread(
                lambda: self.db.conversations.insert_one(conversation)
            )
        return str(result.inserted_id)

    async def update_conversation(self, conversation_id: str, update: Dict) -> bool:
        """Update a conversation and return True if modified."""
        self._ensure_connected()
        
        if self._using_motor:
            result = await self.db.conversations.update_one(
                {"_id": conversation_id}, {"$set": update}
            )
        else:
            result = await asyncio.to_thread(
                lambda: self.db.conversations.update_one(
                    {"_id": conversation_id}, {"$set": update}
                )
            )
        return result.modified_count > 0

    async def get_messages(
        self, 
        conversation_id: str, 
        limit: int = 50, 
        before_timestamp: Optional[float] = None
    ) -> List[Dict]:
        """Get messages for a conversation with optional pagination."""
        self._ensure_connected()
        
        query = {"conversation_id": conversation_id}
        if before_timestamp is not None:
            query["timestamp"] = {"$lt": before_timestamp}

        if self._using_motor:
            cursor = self.db.messages.find(query).sort("timestamp", -1).limit(limit)
            return await cursor.to_list(length=limit)
        else:
            def _get_messages():
                cursor = self.db.messages.find(query).sort("timestamp", -1).limit(limit)
                return list(cursor)
            return await asyncio.to_thread(_get_messages)

    async def save_message(self, message: Dict) -> str:
        """Save a message and return its ID."""
        self._ensure_connected()
        
        if self._using_motor:
            result = await self.db.messages.insert_one(message)
        else:
            result = await asyncio.to_thread(
                lambda: self.db.messages.insert_one(message)
            )
        return str(result.inserted_id)

    async def mark_messages_read(self, message_ids: List[str]) -> int:
        """Mark messages as read and return count of modified documents."""
        self._ensure_connected()
        
        if not message_ids:
            return 0
            
        if self._using_motor:
            result = await self.db.messages.update_many(
                {"message_id": {"$in": message_ids}}, 
                {"$set": {"read": True}}
            )
        else:
            result = await asyncio.to_thread(
                lambda: self.db.messages.update_many(
                    {"message_id": {"$in": message_ids}}, 
                    {"$set": {"read": True}}
                )
            )
        return result.modified_count

    # Collection access methods
    
    def get_collection(self, name: str) -> Union[AsyncIOMotorCollection, pymongo.collection.Collection]:
        """
        Get a collection by name. Works with both Motor and pymongo.
        Used by services/persistence.py and scripts.
        """
        self._ensure_connected()
        
        if hasattr(self.db, name):
            return getattr(self.db, name)
        
        try:
            return self.db[name]
        except Exception as e:
            raise AttributeError(
                f"Database Manager: couldn't find collection '{name}' "
                f"on db instance {type(self.db)}: {e}"
            )

    # Alias for backward compatibility
    collection = get_collection

    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for database transactions (Motor only).
        Falls back to no-op for pymongo.
        """
        if self._using_motor and hasattr(self.client, 'start_session'):
            session = await self.client.start_session()
            try:
                async with session.start_transaction():
                    yield session
            finally:
                await session.end_session()
        else:
            # No transaction support for pymongo fallback
            yield None

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected

    @property
    def using_motor(self) -> bool:
        """Check if using Motor async driver."""
        return self._using_motor

    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the database connection."""
        if not self._connected:
            return {"status": "disconnected", "driver": None, "connected": False}
        
        try:
            if self._using_motor:
                await self.client.admin.command('ping')
            else:
                await asyncio.to_thread(lambda: self.client.admin.command('ping'))
            
            return {
                "status": "healthy",
                "driver": "motor" if self._using_motor else "pymongo",
                "database": self.db_name,
                "connected": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "driver": "motor" if self._using_motor else "pymongo",
                "error": str(e),
                "connected": False
            }


# Global database instance
db_manager = DatabaseManager()