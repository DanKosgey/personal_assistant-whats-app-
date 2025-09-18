"""Persistence helpers extracted from server_back_up.py

These functions are intentionally small wrappers that accept the db_manager
and cache_manager objects to avoid circular imports with the monolithic
server module. They mirror the previous logic but live in a dedicated
services module.
"""
from typing import Any, Dict, List, Optional
import asyncio
import logging
import uuid as _uuid
from datetime import datetime, timezone, timedelta

# Configure logger for this module
logger = logging.getLogger(__name__)


def _serialize_datetime(dt: Any) -> str:
    """Helper to serialize datetime objects consistently."""
    if dt is None:
        return ""
    if hasattr(dt, 'isoformat'):
        try:
            return dt.isoformat()
        except Exception:
            return str(dt)
    return str(dt)


def _serialize_object_id(obj_id: Any) -> str:
    """Helper to serialize MongoDB ObjectId consistently."""
    return str(obj_id) if obj_id is not None else ""


async def _safe_cache_get(cache_manager: Any, key: str) -> Any:
    """Safely get from cache with error handling."""
    try:
        return await cache_manager.get(key)
    except Exception as e:
        logger.debug(f"Cache get failed for key={key}: {e}")
        return None


async def _safe_cache_set(cache_manager: Any, key: str, value: Any, ttl: Optional[int] = None) -> None:
    """Safely set cache with error handling."""
    try:
        if ttl is not None:
            await cache_manager.set(key, value, ttl=ttl)
        else:
            await cache_manager.set(key, value)
    except Exception as e:
        logger.debug(f"Cache set failed for key={key}: {e}")


async def _execute_db_operation(operation, *args, **kwargs):
    """Execute database operation, handling both sync and async collections."""
    try:
        if asyncio.iscoroutinefunction(operation):
            result = await operation(*args, **kwargs)
        else:
            result = await asyncio.to_thread(operation, *args, **kwargs)
        
        # Additional check for Future-like objects to prevent '_asyncio.Future' object has no attribute 'get' error
        if hasattr(result, '__class__') and 'Future' in result.__class__.__name__:
            try:
                result = await result
            except Exception as future_error:
                logger.error(f"Error awaiting Future object in _execute_db_operation: {future_error}")
                raise
        elif asyncio.iscoroutine(result):
            result = await result
            
        return result
    except Exception as e:
        logger.error(f"Error in _execute_db_operation: {e}")
        # Log full traceback for debugging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


async def save_message(message: Any, db_manager: Any, cache_manager: Any) -> None:
    """Save message to the configured messages collection."""
    try:
        messages = db_manager.get_collection('messages')
        if messages is None:
            logger.warning("Messages collection not available")
            return

        # Convert message to dict
        if hasattr(message, 'dict'):
            message_doc = message.dict()
        elif hasattr(message, '__dict__'):
            message_doc = message.__dict__.copy()
        else:
            message_doc = dict(message)

        # Serialize timestamp consistently
        timestamp = getattr(message, 'timestamp', None) or message_doc.get('timestamp')
        message_doc['timestamp'] = _serialize_datetime(timestamp)

        await _execute_db_operation(messages.insert_one, message_doc)
        logger.debug(f"Message saved successfully for conversation: {message_doc.get('conversation_id')}")
        
    except Exception as e:
        logger.error(f"Error saving message to database: {e}")


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
    if not conversation_id:
        logger.warning("update_entities called with empty conversation_id")
        return

    # Update conversation
    try:
        conversations = db_manager.get_collection('conversations')
        if conversations is not None:
            priority_value = getattr(priority, 'value', str(priority)) if priority else "MEDIUM"
            category_value = getattr(category, 'value', str(category)) if category else "general"
            
            await _execute_db_operation(
                conversations.update_one,
                {"conversation_id": conversation_id},
                {
                    "$set": {
                        "last_activity": datetime.now(timezone.utc),
                        "priority": priority_value,
                        "category": category_value
                    },
                    "$inc": {"message_count": 2}
                }
            )
            logger.debug(f"Conversation {conversation_id} updated successfully")
        else:
            logger.warning("Conversations collection not available")
    except Exception as e:
        logger.error(f"Error updating conversation {conversation_id}: {e}")

    # Update contact via contact manager
    if contact_manager is not None and contact is not None:
        try:
            if hasattr(contact_manager, 'update_contact'):
                phone_number = getattr(contact, 'phone_number', None)
                if phone_number:
                    interaction_count = getattr(contact, 'interaction_count', 0)
                    await contact_manager.update_contact(
                        phone_number,
                        {
                            "last_interaction": datetime.now(timezone.utc),
                            "interaction_count": interaction_count + 1
                        }
                    )
                    logger.debug(f"Contact {phone_number} updated via contact_manager")
                else:
                    logger.warning("Contact has no phone_number attribute")
            else:
                logger.warning("contact_manager has no update_contact method")
        except Exception as e:
            logger.error(f"Error updating contact via contact_manager: {e}")


async def get_or_create_contact(phone_number: str, db_manager: Any, cache_manager: Any) -> Dict:
    """Return contact document (dict). Creates a new contact if none exists."""
    if not phone_number:
        logger.warning("get_or_create_contact called with empty phone_number")
        return _create_default_contact("")

    cache_key = f"contact:{phone_number}"
    
    # Try cache first
    cached = await _safe_cache_get(cache_manager, cache_key)
    if cached:
        logger.debug(f"Contact {phone_number} found in cache")
        return cached

    # Try database
    try:
        contacts = db_manager.get_collection('contacts')
        if contacts is None:
            logger.warning("Contacts collection not available, returning default contact")
            return _create_default_contact(phone_number)

        # Handle database operation more carefully to avoid Future.get() errors
        logger.debug(f"Attempting to find contact {phone_number} in database")
        
        # Check if find_one is async or sync
        find_one_method = contacts.find_one
        if asyncio.iscoroutinefunction(find_one_method):
            logger.debug("Using async find_one")
            doc = await find_one_method({"phone_number": phone_number})
        else:
            logger.debug("Using sync find_one with asyncio.to_thread")
            doc = await asyncio.to_thread(find_one_method, {"phone_number": phone_number})
        
        # Additional check for Future objects to prevent '_asyncio.Future' object has no attribute 'get' error
        if hasattr(doc, '__class__') and 'Future' in doc.__class__.__name__:
            logger.debug("Result is a Future-like object, awaiting it properly")
            try:
                doc = await doc
            except Exception as future_error:
                logger.error(f"Error awaiting Future object: {future_error}")
                doc = None
        elif asyncio.iscoroutine(doc):
            logger.debug("Result is a coroutine, awaiting it")
            doc = await doc
            
        logger.debug(f"Database query result type: {type(doc)}")
            
        if doc:
            # Serialize the document
            doc['_id'] = _serialize_object_id(doc.get('_id'))
            for field in ['created_at', 'last_interaction']:
                doc[field] = _serialize_datetime(doc.get(field))
            
            await _safe_cache_set(cache_manager, cache_key, doc)
            logger.debug(f"Contact {phone_number} found in database")
            return doc
        else:
            logger.debug(f"No existing contact found for {phone_number}")
            
    except Exception as e:
        logger.error(f"Error fetching contact {phone_number} from database: {e}")
        # Log more details about the error
        logger.error(f"Error type: {type(e)}, Error args: {e.args}")
        # Log full traceback for debugging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

    # Create new contact
    logger.debug(f"Creating new contact for {phone_number}")
    new_contact = _create_default_contact(phone_number)
    
    try:
        contacts = db_manager.get_collection('contacts')
        if contacts is not None:
            # Handle insert operation carefully too
            insert_method = contacts.insert_one
            if asyncio.iscoroutinefunction(insert_method):
                result = await insert_method(new_contact.copy())
            else:
                result = await asyncio.to_thread(insert_method, new_contact.copy())
                
            # Check if result is a Future-like object
            if hasattr(result, '__class__') and 'Future' in result.__class__.__name__:
                try:
                    result = await result
                except Exception as future_error:
                    logger.error(f"Error awaiting Future object for insert: {future_error}")
            elif asyncio.iscoroutine(result):
                result = await result
                
            logger.info(f"New contact {phone_number} created in database")
    except Exception as e:
        logger.error(f"Error saving new contact {phone_number} to database: {e}")
        logger.error(f"Error type: {type(e)}, Error args: {e.args}")
        # Log full traceback for debugging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

    await _safe_cache_set(cache_manager, cache_key, new_contact)
    return new_contact


def _create_default_contact(phone_number: str) -> Dict:
    """Create a default contact dictionary."""
    return {
        "phone_number": phone_number,
        "priority_level": "MEDIUM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_interaction": None,
        "interaction_count": 0,
        "is_known": False,
        "is_vip": False,
        "tags": [],
        "preferences": {}
    }


async def update_contact(phone_number: str, updates: Dict, db_manager: Any, cache_manager: Any) -> Optional[Dict]:
    """Apply updates to a contact and return the updated document (dict) if possible."""
    if not phone_number or not updates:
        logger.warning("update_contact called with empty phone_number or updates")
        return None

    cache_key = f"contact:{phone_number}"
    
    try:
        contacts = db_manager.get_collection('contacts')
        if contacts is not None:
            # Serialize datetime objects in updates
            serialized_updates = {}
            for key, value in updates.items():
                if hasattr(value, 'isoformat'):
                    serialized_updates[key] = _serialize_datetime(value)
                else:
                    serialized_updates[key] = value
            
            # Handle update operation with proper Future handling
            update_method = contacts.update_one
            if asyncio.iscoroutinefunction(update_method):
                update_result = await update_method(
                    {"phone_number": phone_number},
                    {"$set": serialized_updates},
                    upsert=True
                )
            else:
                update_result = await asyncio.to_thread(
                    update_method,
                    {"phone_number": phone_number},
                    {"$set": serialized_updates},
                    upsert=True
                )
            
            # Check if result is a Future-like object
            if hasattr(update_result, '__class__') and 'Future' in update_result.__class__.__name__:
                try:
                    update_result = await update_result
                except Exception as future_error:
                    logger.error(f"Error awaiting Future object for update: {future_error}")
            elif asyncio.iscoroutine(update_result):
                update_result = await update_result
            
            # Handle find operation with proper Future handling
            find_method = contacts.find_one
            if asyncio.iscoroutinefunction(find_method):
                doc = await find_method({"phone_number": phone_number})
            else:
                doc = await asyncio.to_thread(find_method, {"phone_number": phone_number})
                
            # Check if result is a Future-like object
            if hasattr(doc, '__class__') and 'Future' in doc.__class__.__name__:
                try:
                    doc = await doc
                except Exception as future_error:
                    logger.error(f"Error awaiting Future object for find: {future_error}")
                    doc = None
            elif asyncio.iscoroutine(doc):
                doc = await doc
                
            if doc:
                doc['_id'] = _serialize_object_id(doc.get('_id'))
                for field in ['created_at', 'last_interaction']:
                    doc[field] = _serialize_datetime(doc.get(field))
                
                await _safe_cache_set(cache_manager, cache_key, doc)
                logger.debug(f"Contact {phone_number} updated in database")
                return doc
        else:
            logger.warning("Contacts collection not available")
            
    except Exception as e:
        logger.error(f"Error updating contact {phone_number} in database: {e}")
        # Log full traceback for debugging
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
    # Fallback: update cached version
    try:
        cached = await _safe_cache_get(cache_manager, cache_key) or _create_default_contact(phone_number)
        cached.update(updates)
        # Serialize datetime fields
        for field in ['created_at', 'last_interaction']:
            if field in cached:
                cached[field] = _serialize_datetime(cached[field])
        
        await _safe_cache_set(cache_manager, cache_key, cached)
        logger.debug(f"Contact {phone_number} updated in cache as fallback")
        return cached
    except Exception as e:
        logger.warning(f"Cache update failed for contact {phone_number}: {e}")
        return None


async def get_or_create_conversation(phone_number: str, db_manager: Any, cache_manager: Any) -> str:
    """Return an active conversation id for the phone number, creating one if needed."""
    if not phone_number:
        logger.warning("get_or_create_conversation called with empty phone_number")
        return str(_uuid.uuid4())

    cache_key = f"active_conv:{phone_number}"
    
    # Try cache first
    cached = await _safe_cache_get(cache_manager, cache_key)
    if cached:
        logger.debug(f"Active conversation found in cache for {phone_number}")
        return cached

    # Try to find recent active conversation
    try:
        conversations = db_manager.get_collection('conversations')
        if conversations is not None:
            recent_time = datetime.now(timezone.utc) - timedelta(hours=4)
            
            def _find_recent():
                return conversations.find_one(
                    {
                        "phone_number": phone_number, 
                        "status": "active", 
                        "last_activity": {"$gte": recent_time}
                    },
                    sort=[("last_activity", -1)]
                )
            
            recent_conv = await asyncio.to_thread(_find_recent)
            
            if recent_conv:
                conv_id = recent_conv.get('conversation_id')
                if conv_id:
                    await _safe_cache_set(cache_manager, cache_key, conv_id, ttl=14400)
                    logger.debug(f"Recent conversation {conv_id} found for {phone_number}")
                    return conv_id
    except Exception as e:
        logger.error(f"Error fetching conversation for {phone_number}: {e}")

    # Create new conversation
    conversation_id = str(_uuid.uuid4())
    conversation_doc = {
        "conversation_id": conversation_id,
        "phone_number": phone_number,
        "status": "active",
        "state": "active",  # Add conversation state
        "start_time": datetime.now(timezone.utc),
        "last_activity": datetime.now(timezone.utc),
        "message_count": 0,
        "priority": "MEDIUM",
        "category": "general",
        "context": {},
        "summary": None,
        # EOC metadata fields
        "eoc_confidence": None,
        "eoc_detected_by": None,
        "eoc_example_id": None,
        "turns_count": 0,  # Track conversation turns
    }

    try:
        conversations = db_manager.get_collection('conversations')
        if conversations is not None:
            await _execute_db_operation(conversations.insert_one, conversation_doc)
            logger.info(f"New conversation {conversation_id} created for {phone_number}")
    except Exception as e:
        logger.error(f"Error saving conversation {conversation_id} to database: {e}")

    await _safe_cache_set(cache_manager, cache_key, conversation_id, ttl=14400)
    return conversation_id


async def get_conversation_context(conversation_id: str, limit: int, db_manager: Any) -> List[Dict]:
    """Fetch conversation messages and return a chronological list (oldest first)."""
    if not conversation_id:
        logger.warning("get_conversation_context called with empty conversation_id")
        return []

    try:
        messages_collection = db_manager.get_collection('messages')
        if messages_collection is None:
            logger.warning("Messages collection not available")
            return []

        def _fetch_messages():
            cursor = messages_collection.find(
                {"conversation_id": conversation_id}
            ).sort("timestamp", -1).limit(limit)
            return list(cursor)

        msgs = await asyncio.to_thread(_fetch_messages)
        msgs.reverse()  # Make chronological (oldest first)
        
        # Serialize messages
        for msg in msgs:
            msg['_id'] = _serialize_object_id(msg.get('_id'))
            msg['timestamp'] = _serialize_datetime(msg.get('timestamp'))
            
        logger.debug(f"Retrieved {len(msgs)} messages for conversation {conversation_id}")
        return msgs
        
    except Exception as e:
        logger.error(f"Error fetching conversation context for {conversation_id}: {e}")
        return []


async def update_conversation_activity(conversation_id: str, db_manager: Any, inc: int = 1) -> None:
    """Update conversation last_activity timestamp and increment message count."""
    if not conversation_id:
        logger.warning("update_conversation_activity called with empty conversation_id")
        return

    try:
        conversations = db_manager.get_collection('conversations')
        if conversations is None:
            logger.warning("Conversations collection not available")
            return
            
        await _execute_db_operation(
            conversations.update_one,
            {"conversation_id": conversation_id},
            {
                "$set": {"last_activity": datetime.now(timezone.utc)}, 
                "$inc": {"message_count": inc, "turns_count": inc}  # Also increment turns count
            },
            upsert=False
        )
        logger.debug(f"Conversation {conversation_id} activity updated")
        
    except Exception as e:
        logger.error(f"Error updating conversation {conversation_id} activity: {e}")


async def cleanup_old_messages(db_manager: Any, days: int = 30, exclude_priority: str = "CRITICAL") -> int:
    """Delete messages older than `days` that are not of the exclude_priority. Returns deleted count."""
    if days <= 0:
        logger.warning("cleanup_old_messages called with invalid days parameter")
        return 0

    try:
        messages = db_manager.get_collection('messages')
        if messages is None:
            logger.warning("Messages collection not available")
            return 0
            
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        result = await _execute_db_operation(
            messages.delete_many,
            {
                "timestamp": {"$lt": cutoff_date}, 
                "priority": {"$ne": exclude_priority}
            }
        )
        
        deleted_count = getattr(result, 'deleted_count', 0)
        logger.info(f"Cleaned up {deleted_count} old messages (older than {days} days)")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error cleaning up old messages: {e}")
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
    if limit <= 0:
        logger.warning("query_conversations called with invalid limit")
        return []

    try:
        convs = db_manager.get_collection('conversations')
        if convs is None:
            logger.warning("Conversations collection not available")
            return []

        # Build query
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
        
        # Serialize results
        enhanced = []
        for conv in rows:
            conv["_id"] = _serialize_object_id(conv.get("_id"))
            for field in ["start_time", "last_activity", "end_time"]:
                conv[field] = _serialize_datetime(conv.get(field))
            enhanced.append(conv)
            
        logger.debug(f"Retrieved {len(enhanced)} conversations with query: {query}")
        return enhanced
        
    except Exception as e:
        logger.error(f"Error querying conversations: {e}")
        return []