from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import asyncio
from ..db import db_manager

logger = logging.getLogger(__name__)

class UserMemory:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.preferences: Dict[str, Any] = {}
        self.session_context: List[Dict[str, Any]] = []
        self.last_interaction: Optional[datetime] = None
        self.contacts: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "contacts": self.contacts,
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
            "context_size": len(self.session_context)
        }

class MemoryManager:
    """Manages user memory, preferences, and conversation context"""
    
    def __init__(self, context_ttl_minutes: int = 30):
        self.context_ttl = timedelta(minutes=context_ttl_minutes)
        self.user_memories: Dict[str, UserMemory] = {}

    async def get_user_memory(self, user_id: str) -> UserMemory:
        """Get or create user memory"""
        if user_id not in self.user_memories:
            # Try to load from database
            stored_memory = await self._load_from_db(user_id)
            if stored_memory:
                self.user_memories[user_id] = stored_memory
            else:
                self.user_memories[user_id] = UserMemory(user_id)
        return self.user_memories[user_id]

    async def add_to_context(self, user_id: str, message: str, response: str) -> None:
        """Add a message-response pair to user's context"""
        memory = await self.get_user_memory(user_id)
        now = datetime.utcnow()
        
        # Clear old context if last interaction was too long ago
        if memory.last_interaction and now - memory.last_interaction > self.context_ttl:
            memory.session_context.clear()
        
        memory.session_context.append({
            "timestamp": now,
            "message": message,
            "response": response
        })
        memory.last_interaction = now
        
        # Keep only last 10 interactions in context
        if len(memory.session_context) > 10:
            memory.session_context = memory.session_context[-10:]
        
        await self._save_to_db(user_id, memory)

    async def set_preference(self, user_id: str, key: str, value: Any) -> None:
        """Set a user preference"""
        memory = await self.get_user_memory(user_id)
        memory.preferences[key] = value
        await self._save_to_db(user_id, memory)

    async def get_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        """Get a user preference"""
        memory = await self.get_user_memory(user_id)
        return memory.preferences.get(key, default)

    async def add_contact(self, user_id: str, contact_info: Dict[str, Any]) -> None:
        """Add or update a contact for the user"""
        memory = await self.get_user_memory(user_id)
        contact_id = contact_info.get("id") or contact_info.get("phone")
        if contact_id:
            memory.contacts[contact_id] = {
                **contact_info,
                "last_updated": datetime.utcnow().isoformat()
            }
            await self._save_to_db(user_id, memory)

    async def get_contacts(self, user_id: str) -> Dict[str, Any]:
        """Get user's contacts"""
        memory = await self.get_user_memory(user_id)
        return memory.contacts

    async def get_recent_context(self, user_id: str, minutes: int = None) -> List[Dict[str, Any]]:
        """Get recent conversation context"""
        memory = await self.get_user_memory(user_id)
        if not minutes:
            return memory.session_context
        
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        return [
            ctx for ctx in memory.session_context 
            if ctx["timestamp"] >= cutoff
        ]

    async def _save_to_db(self, user_id: str, memory: UserMemory) -> None:
        """Save memory to database"""
        try:
            # Try to get a valid users collection via DatabaseManager
            col = None
            get_col = getattr(db_manager, 'get_collection', None)
            if callable(get_col):
                try:
                    col = get_col('users')
                except Exception:
                    col = None

            if col is None:
                # Fallback to raw db attribute
                db_attr = getattr(db_manager, 'db', None)
                if db_attr is not None:
                    try:
                        col = getattr(db_attr, 'users', None) or db_attr['users']
                    except Exception:
                        col = None

            if col is None or not hasattr(col, 'update_one'):
                logger.debug("No users collection available - skipping DB save for user %s", user_id)
                return

            # Use thread executor for sync pymongo drivers
            await asyncio.to_thread(col.update_one, {"_id": user_id}, {"$set": memory.to_dict()}, True)
        except Exception as e:
            logger.error(f"Failed to save memory for user {user_id}: {e}")

    async def _load_from_db(self, user_id: str) -> Optional[UserMemory]:
        """Load memory from database"""
        try:
            # Resolve users collection safely
            col = None
            get_col = getattr(db_manager, 'get_collection', None)
            if callable(get_col):
                try:
                    col = get_col('users')
                except Exception:
                    col = None

            if col is None:
                db_attr = getattr(db_manager, 'db', None)
                if db_attr is not None:
                    try:
                        col = getattr(db_attr, 'users', None) or db_attr['users']
                    except Exception:
                        col = None

            if col is None or not hasattr(col, 'find_one'):
                logger.debug("No users collection available - skipping DB load for user %s", user_id)
                return None

            data = await asyncio.to_thread(col.find_one, {"_id": user_id})
            if data:
                memory = UserMemory(user_id)
                memory.preferences = data.get("preferences", {})
                memory.contacts = data.get("contacts", {})
                if data.get("last_interaction"):
                    try:
                        memory.last_interaction = datetime.fromisoformat(data["last_interaction"])
                    except Exception:
                        memory.last_interaction = None
                return memory
        except Exception as e:
            logger.error(f"Failed to load memory for user {user_id}: {e}")
        return None
