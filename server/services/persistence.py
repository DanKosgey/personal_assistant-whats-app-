"""Persistence helpers extracted from server_back_up.py

These functions are intentionally small wrappers that accept the db_manager
and cache_manager objects to avoid circular imports with the monolithic
server module. They mirror the previous logic but live in a dedicated
services module.
"""
from typing import Any, Dict, List, Optional
import asyncio
from datetime import datetime, timezone, timedelta


async def save_message(message: Any, db_manager: Any, cache_manager: Any) -> None:
    """Save message to the configured messages collection."""
    messages = db_manager.get_collection('messages')
    if messages is not None:
        try:
            message_doc = message.dict() if hasattr(message, 'dict') else dict(message)
            # Ensure timestamp is serialized
            try:
                ts = message.timestamp if hasattr(message, 'timestamp') else message_doc.get('timestamp')
                if hasattr(ts, 'isoformat'):
                    message_doc['timestamp'] = ts.isoformat()
                else:
                    message_doc['timestamp'] = str(ts)
            except Exception:
                message_doc['timestamp'] = str(message_doc.get('timestamp', ''))

            # Use to_thread for sync pymongo operations
            await asyncio.to_thread(messages.insert_one, message_doc)
        except Exception as e:
            # Avoid importing logger here to keep function simple; callers should log if needed
            try:
                import logging
                logging.getLogger(__name__).error(f"Error saving message to database: {e}")
            except Exception:
                pass


async def update_entities(
    conversation_id: str,
    contact: Any,
    priority: Any,
    category: Any,
    db_manager: Any,
    cache_manager: Any,
    contact_manager: Any
) -> None:
    """Update conversation and contact entities in the DB and cache."""
    try:
        conversations = db_manager.get_collection('conversations')
        if conversations is not None:
            try:
                await asyncio.to_thread(
                    conversations.update_one,
                    {"conversation_id": conversation_id},
                    {
                        "$set": {
                            "last_activity": datetime.now(timezone.utc),
                            "priority": getattr(priority, 'value', str(priority)),
                            "category": getattr(category, 'value', str(category))
                        },
                        "$inc": {"message_count": 2}
                    }
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Error updating conversation: {e}")

        # Update contact via provided manager if available
        try:
            if contact_manager is not None and hasattr(contact_manager, 'update_contact'):
                await contact_manager.update_contact(
                    contact.phone_number,
                    {
                        "last_interaction": datetime.now(timezone.utc),
                        "interaction_count": getattr(contact, 'interaction_count', 0) + 1
                    }
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error updating contact: {e}")
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"update_entities failed: {e}")


async def get_or_create_contact(phone_number: str, db_manager: Any, cache_manager: Any) -> Dict:
    """Return contact document (dict). Creates a new contact if none exists."""
    cache_key = f"contact:{phone_number}"
    try:
        cached = await cache_manager.get(cache_key)
        if cached:
            return cached
    except Exception:
        pass

    contacts = db_manager.get_collection('contacts')
    if contacts is None:
        # Return minimal contact dict when DB not available
        return {
            "phone_number": phone_number,
            "priority_level": "MEDIUM",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "interaction_count": 0
        }

    try:
        doc = await asyncio.to_thread(contacts.find_one, {"phone_number": phone_number})
        if doc:
            doc['_id'] = str(doc.get('_id', ''))
            for field in ['created_at', 'last_interaction']:
                v = doc.get(field)
                if hasattr(v, 'isoformat'):
                    try:
                        doc[field] = v.isoformat()
                    except Exception:
                        doc[field] = str(v)
            await cache_manager.set(cache_key, doc)
            return doc
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error fetching contact from database: {e}")

    # Create new contact
    new_contact = {
        "phone_number": phone_number,
        "priority_level": "MEDIUM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "interaction_count": 0,
        "is_known": False,
        "is_vip": False,
        "tags": [],
        "preferences": {}
    }

    try:
        await asyncio.to_thread(contacts.insert_one, new_contact)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error saving new contact to database: {e}")

    try:
        await cache_manager.set(cache_key, new_contact)
    except Exception:
        pass

    return new_contact


async def update_contact(phone_number: str, updates: Dict, db_manager: Any, cache_manager: Any) -> Optional[Dict]:
    """Apply updates to a contact and return the updated document (dict) if possible."""
    contacts = db_manager.get_collection('contacts')
    try:
        if contacts is not None:
            await asyncio.to_thread(contacts.update_one, {"phone_number": phone_number}, {"$set": updates}, upsert=True)
            doc = await asyncio.to_thread(contacts.find_one, {"phone_number": phone_number})
            if doc:
                doc['_id'] = str(doc.get('_id', ''))
                await cache_manager.set(f"contact:{phone_number}", doc)
                return doc
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error updating contact in database: {e}")
    # Fallback: return an updated minimal dict
    try:
        cached = await cache_manager.get(f"contact:{phone_number}") or {}
        cached.update(updates)
        await cache_manager.set(f"contact:{phone_number}", cached)
        return cached
    except Exception:
        return None


async def get_or_create_conversation(phone_number: str, db_manager: Any, cache_manager: Any) -> str:
    """Return an active conversation id for the phone number, creating one if needed."""
    cache_key = f"active_conv:{phone_number}"
    try:
        cached = await cache_manager.get(cache_key)
        if cached:
            return cached
    except Exception:
        pass

    conversations = db_manager.get_collection('conversations')
    if conversations is not None:
        try:
            recent_time = datetime.now(timezone.utc) - timedelta(hours=4)
            recent_conv = await asyncio.to_thread(
                conversations.find_one,
                {"phone_number": phone_number, "status": "active", "last_activity": {"$gte": recent_time}},
                sort=[("last_activity", -1)]
            )
            if recent_conv:
                conv_id = recent_conv.get('conversation_id')
                await cache_manager.set(cache_key, conv_id, ttl=14400)
                return conv_id
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error fetching conversation from database: {e}")

    # Create new conversation
    import uuid as _uuid
    conversation_id = str(_uuid.uuid4())
    conversation_doc = {
        "conversation_id": conversation_id,
        "phone_number": phone_number,
        "status": "active",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "last_activity": datetime.now(timezone.utc).isoformat(),
        "message_count": 0,
        "priority": "MEDIUM",
        "category": "general",
        "context": {},
        "summary": None
    }

    try:
        if conversations is not None:
            await asyncio.to_thread(conversations.insert_one, conversation_doc)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error saving conversation to database: {e}")

    try:
        await cache_manager.set(cache_key, conversation_id, ttl=14400)
    except Exception:
        pass

    return conversation_id


async def get_conversation_context(conversation_id: str, limit: int, db_manager: Any) -> List[Dict]:
    """Fetch conversation messages and return a chronological list (oldest first)."""
    messages_collection = db_manager.get_collection('messages')
    if messages_collection is None or not hasattr(messages_collection, 'find'):
        return []

    try:
        msgs = await asyncio.to_thread(lambda: list(messages_collection.find({"conversation_id": conversation_id}).sort("timestamp", -1).limit(limit)))
        msgs.reverse()
        for msg in msgs:
            msg['_id'] = str(msg.get('_id', ''))
            t = msg.get('timestamp')
            if hasattr(t, 'isoformat'):
                try:
                    msg['timestamp'] = t.isoformat()
                except Exception:
                    msg['timestamp'] = str(t)
            else:
                msg['timestamp'] = str(t)
        return msgs
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error fetching conversation context: {e}")
        return []


async def update_conversation_activity(conversation_id: str, db_manager: Any, inc: int = 1) -> None:
    """Update conversation last_activity timestamp and increment message count."""
    try:
        conversations = db_manager.get_collection('conversations')
        if conversations is None:
            return
        await asyncio.to_thread(
            conversations.update_one,
            {"conversation_id": conversation_id},
            {"$set": {"last_activity": datetime.now(timezone.utc)}, "$inc": {"message_count": inc}},
            upsert=False,
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error updating conversation activity: {e}")


async def cleanup_old_messages(db_manager: Any, days: int = 30, exclude_priority: str = "CRITICAL") -> int:
    """Delete messages older than `days` that are not of the exclude_priority. Returns deleted count."""
    try:
        messages = db_manager.get_collection('messages')
        if messages is None:
            return 0
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        result = await asyncio.to_thread(
            messages.delete_many,
            {"timestamp": {"$lt": cutoff_date}, "priority": {"$ne": exclude_priority}}
        )
        return getattr(result, 'deleted_count', 0)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error cleaning up old messages: {e}")
        return 0


async def query_conversations(
    db_manager: Any,
    limit: int = 50,
    priority: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    phone_number: Optional[str] = None,
) -> List[Dict]:
    """Query conversations with filters and return formatted list (timestamps isoformatted)."""
    try:
        convs = db_manager.get_collection('conversations')
        if convs is None:
            return []

        query = {}
        if priority:
            query["priority"] = priority
        if category:
            query["category"] = category
        if status:
            query["status"] = status
        if phone_number:
            query["phone_number"] = phone_number

        def _fetch():
            cursor = convs.find(query).sort("last_activity", -1).limit(limit)
            return list(cursor)

        rows = await asyncio.to_thread(_fetch)
        enhanced = []
        for conv in rows:
            conv["_id"] = str(conv.get("_id"))
            for field in ["start_time", "last_activity", "end_time"]:
                v = conv.get(field)
                if hasattr(v, 'isoformat'):
                    try:
                        conv[field] = v.isoformat()
                    except Exception:
                        conv[field] = str(v)
            enhanced.append(conv)
        return enhanced
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error querying conversations: {e}")
        return []
