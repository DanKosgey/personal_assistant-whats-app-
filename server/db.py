"""Database manager with Motor async client when available, otherwise fall back to pymongo with async wrappers."""
from typing import Any, Optional, Dict, List
import logging
import asyncio
from .config import config

logger = logging.getLogger(__name__)

try:
    from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase  # type: ignore
    _HAS_MOTOR = True
except Exception:
    _HAS_MOTOR = False

if _HAS_MOTOR:
    class DatabaseManager:
        def __init__(self):
            self.uri = config.MONGO_URL
            self.db_name = config.MONGO_DB_NAME
            self.client: Optional[AsyncIOMotorClient] = None
            self.db: Optional[AsyncIOMotorDatabase] = None

        async def connect(self):
            try:
                self.client = AsyncIOMotorClient(self.uri, serverSelectionTimeoutMS=5000)
                await self.client.admin.command('ping')
                self.db = self.client[self.db_name]
                logger.info("✅ MongoDB connected (motor)")
                await self._setup_indexes()
            except Exception as e:
                logger.error(f"❌ MongoDB (motor) connection failed: {e}")
                raise

        async def close(self):
            if self.client:
                self.client.close()
                logger.info("✅ MongoDB (motor) connection closed")

        async def _setup_indexes(self):
            try:
                await self.db.conversations.create_index([("phone_number", 1)])
                await self.db.conversations.create_index([("status", 1)])
                await self.db.conversations.create_index([("last_message_time", -1)])
                await self.db.messages.create_index([("conversation_id", 1)])
                await self.db.messages.create_index([("timestamp", -1)])
                try:
                    await self.db.messages.create_index([("message_id", 1)], unique=True)
                except Exception as ie:
                    # Duplicate key during unique index build is common if legacy data exists.
                    msg = str(ie)
                    if 'E11000' in msg or 'duplicate key' in msg.lower():
                        logger.warning("Could not create unique index message_id: duplicate keys present; leaving existing data as-is")
                    else:
                        logger.warning(f"Could not create unique index message_id: {ie}")

                logger.info("✅ Database indexes processed (motor)")
            except Exception as e:
                logger.warning(f"Could not create motor indexes: {e}")

        async def get_conversation(self, conversation_id: str) -> Optional[Dict]:
            return await self.db.conversations.find_one({"_id": conversation_id})

        async def create_conversation(self, conversation: Dict) -> str:
            result = await self.db.conversations.insert_one(conversation)
            return str(result.inserted_id)

        async def update_conversation(self, conversation_id: str, update: Dict) -> bool:
            result = await self.db.conversations.update_one({"_id": conversation_id}, {"$set": update})
            return result.modified_count > 0

        async def get_messages(self, conversation_id: str, limit: int = 50, before_timestamp: Optional[float] = None) -> List[Dict]:
            query = {"conversation_id": conversation_id}
            if before_timestamp:
                query["timestamp"] = {"$lt": before_timestamp}
            cursor = self.db.messages.find(query).sort("timestamp", -1).limit(limit)
            return await cursor.to_list(length=limit)

        async def save_message(self, message: Dict) -> str:
            result = await self.db.messages.insert_one(message)
            return str(result.inserted_id)

        async def mark_messages_read(self, message_ids: List[str]) -> int:
            result = await self.db.messages.update_many({"message_id": {"$in": message_ids}}, {"$set": {"read": True}})
            return result.modified_count

        def get_collection(self, name: str):
            """
            Compatibility helper used by services/persistence.py and scripts that
            expect db_manager.get_collection('messages'). Works for Motor or pymongo.
            """
            # Attribute-style (Motor often exposes collections as attributes)
            try:
                if hasattr(self.db, name):
                    col = getattr(self.db, name)
                    if col is not None:
                        return col
            except Exception:
                pass

            # Mapping-style (pymongo)
            try:
                return self.db[name]
            except Exception:
                raise AttributeError(f"Database Manager: couldn't find collection '{name}' on db instance {type(self.db)}")

        # alias
        collection = get_collection

else:
    # Fallback to synchronous pymongo wrapped with asyncio.to_thread
    import pymongo

    class DatabaseManager:
        def __init__(self):
            self.uri = config.MONGO_URL
            self.db_name = config.MONGO_DB_NAME
            self.client: Optional[pymongo.MongoClient] = None
            self.db = None

        async def connect(self):
            def _connect():
                self.client = pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=5000)
                # Verify connection
                self.client.admin.command('ping')
                self.db = self.client[self.db_name]
                # Setup indexes synchronously
                try:
                    self.db.conversations.create_index([("phone_number", 1)])
                    self.db.conversations.create_index([("status", 1)])
                    self.db.conversations.create_index([("last_message_time", -1)])
                    self.db.messages.create_index([("conversation_id", 1)])
                    self.db.messages.create_index([("timestamp", -1)])
                    try:
                        self.db.messages.create_index([("message_id", 1)], unique=True)
                    except Exception as ie:
                        msg = str(ie)
                        if 'E11000' in msg or 'duplicate key' in msg.lower():
                            logger.warning("Could not create unique index message_id: duplicate keys present; leaving existing data as-is")
                        else:
                            logger.warning(f"Could not create unique index message_id: {ie}")

                    logger.info("✅ Database indexes processed (pymongo)")
                except Exception as e:
                    logger.warning(f"Could not create pymongo indexes: {e}")

            try:
                await asyncio.to_thread(_connect)
                logger.info("✅ MongoDB connected (pymongo)")
            except Exception as e:
                logger.error(f"❌ MongoDB (pymongo) connection failed: {e}")
                raise

        async def close(self):
            if self.client:
                await asyncio.to_thread(self.client.close)
                logger.info("✅ MongoDB (pymongo) connection closed")

        async def get_conversation(self, conversation_id: str) -> Optional[Dict]:
            return await asyncio.to_thread(lambda: self.db.conversations.find_one({"_id": conversation_id}))

        async def create_conversation(self, conversation: Dict) -> str:
            def _insert():
                result = self.db.conversations.insert_one(conversation)
                return str(result.inserted_id)
            return await asyncio.to_thread(_insert)

        async def update_conversation(self, conversation_id: str, update: Dict) -> bool:
            def _update():
                result = self.db.conversations.update_one({"_id": conversation_id}, {"$set": update})
                return result.modified_count > 0
            return await asyncio.to_thread(_update)

        async def get_messages(self, conversation_id: str, limit: int = 50, before_timestamp: Optional[float] = None) -> List[Dict]:
            def _get():
                query = {"conversation_id": conversation_id}
                if before_timestamp:
                    query["timestamp"] = {"$lt": before_timestamp}
                cursor = self.db.messages.find(query).sort("timestamp", -1).limit(limit)
                return list(cursor)
            return await asyncio.to_thread(_get)

        async def save_message(self, message: Dict) -> str:
            def _save():
                result = self.db.messages.insert_one(message)
                return str(result.inserted_id)
            return await asyncio.to_thread(_save)

        async def mark_messages_read(self, message_ids: List[str]) -> int:
            def _mark():
                result = self.db.messages.update_many({"message_id": {"$in": message_ids}}, {"$set": {"read": True}})
                return result.modified_count
            return await asyncio.to_thread(_mark)

        def get_collection(self, name: str):
            """
            Compatibility helper used by services/persistence.py and scripts that
            expect db_manager.get_collection('messages'). Works for Motor or pymongo.
            """
            try:
                if hasattr(self.db, name):
                    col = getattr(self.db, name)
                    if col is not None:
                        return col
            except Exception:
                pass

            try:
                return self.db[name]
            except Exception:
                raise AttributeError(f"Database Manager: couldn't find collection '{name}' on db instance {type(self.db)}")

        # alias
        collection = get_collection


# Global database instance
db_manager = DatabaseManager()
