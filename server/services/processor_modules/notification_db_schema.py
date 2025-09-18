"""
Database schema for the autonomous notification system.
This file defines the schema for conversations, summaries, notifications, and feedback.
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Schema definitions for MongoDB collections

CONVERSATIONS_SCHEMA = {
    "collection_name": "conversations",
    "indexes": [
        {"keys": [("phone_number", 1)], "unique": False},
        {"keys": [("state", 1)], "unique": False},
        {"keys": [("last_activity", -1)], "unique": False},
        {"keys": [("owner_id", 1)], "unique": False}
    ],
    "fields": {
        "_id": {"type": "ObjectId", "required": True},
        "phone_number": {"type": "string", "required": True},
        "owner_id": {"type": "string", "required": True},
        "state": {"type": "string", "required": True, "enum": ["active", "pending_end", "ended", "reopened"]},
        "current_state": {"type": "string", "required": False},
        "last_message_ts": {"type": "datetime", "required": False},
        "memory_pointers": {"type": "object", "required": False},
        "message_count": {"type": "integer", "required": False, "default": 0},
        "turns_count": {"type": "integer", "required": False, "default": 0},
        "summary": {"type": "string", "required": False},
        "end_time": {"type": "datetime", "required": False},
        "eoc_confidence": {"type": "float", "required": False},
        "eoc_detected_by": {"type": "string", "required": False},
        "eoc_example_id": {"type": "string", "required": False},
        "eoc_detected_time": {"type": "datetime", "required": False},
        "last_activity": {"type": "datetime", "required": True},
        "created_at": {"type": "datetime", "required": True}
    }
}

SUMMARIES_SCHEMA = {
    "collection_name": "summaries",
    "indexes": [
        {"keys": [("conversation_id", 1)], "unique": False},
        {"keys": [("created_at", -1)], "unique": False},
        {"keys": [("owner_id", 1)], "unique": False}
    ],
    "fields": {
        "_id": {"type": "ObjectId", "required": True},
        "summary_id": {"type": "string", "required": True},
        "conversation_id": {"type": "string", "required": True},
        "owner_id": {"type": "string", "required": True},
        "summary_text": {"type": "string", "required": True},
        "eoc_confidence": {"type": "float", "required": True},
        "model_metadata": {"type": "object", "required": False},
        "features": {"type": "object", "required": False},
        "importance_score": {"type": "float", "required": False},
        "created_at": {"type": "datetime", "required": True}
    }
}

NOTIFICATIONS_SCHEMA = {
    "collection_name": "notifications",
    "indexes": [
        {"keys": [("conversation_id", 1)], "unique": False},
        {"keys": [("owner_id", 1)], "unique": False},
        {"keys": [("created_at", -1)], "unique": False},
        {"keys": [("notify_owner", 1)], "unique": False}
    ],
    "fields": {
        "_id": {"type": "ObjectId", "required": True},
        "id": {"type": "string", "required": True},  # Notification ID
        "owner_id": {"type": "string", "required": True},
        "conversation_id": {"type": "string", "required": True},
        "eoc_confidence": {"type": "float", "required": True},
        "importance_score": {"type": "float", "required": True},
        "notify_owner": {"type": "boolean", "required": True},
        "auto_send_enabled": {"type": "boolean", "required": True},
        "thresholds": {"type": "object", "required": True},
        "features": {"type": "object", "required": False},
        "reasoning": {"type": "string", "required": False},
        "timestamp": {"type": "float", "required": True},
        "contact_name": {"type": "string", "required": False},
        "contact_phone": {"type": "string", "required": False},
        "send_result": {"type": "object", "required": False},
        "created_at": {"type": "datetime", "required": True}
    }
}

NOTIFICATION_IDEMPOTENCY_SCHEMA = {
    "collection_name": "notification_idempotency",
    "indexes": [
        {"keys": [("key", 1)], "unique": True},
        {"keys": [("ttl", 1)], "unique": False}
    ],
    "fields": {
        "_id": {"type": "ObjectId", "required": True},
        "key": {"type": "string", "required": True},
        "ttl": {"type": "float", "required": True},
        "created_at": {"type": "float", "required": True}
    }
}

NOTIFICATION_FEEDBACK_SCHEMA = {
    "collection_name": "notification_feedback",
    "indexes": [
        {"keys": [("notification_id", 1)], "unique": False},
        {"keys": [("owner_id", 1)], "unique": False},
        {"keys": [("created_at", -1)], "unique": False},
        {"keys": [("helpful", 1)], "unique": False}
    ],
    "fields": {
        "_id": {"type": "ObjectId", "required": True},
        "id": {"type": "string", "required": True},
        "notification_id": {"type": "string", "required": True},
        "owner_id": {"type": "string", "required": True},
        "helpful": {"type": "boolean", "required": True},
        "notes": {"type": "string", "required": False},
        "created_at": {"type": "float", "required": True}
    }
}

OWNER_PREFERENCES_SCHEMA = {
    "collection_name": "owner_preferences",
    "indexes": [
        {"keys": [("owner_id", 1)], "unique": True}
    ],
    "fields": {
        "_id": {"type": "ObjectId", "required": True},
        "owner_id": {"type": "string", "required": True},
        "EOC_THRESHOLD": {"type": "float", "required": False},
        "NOTIFY_THRESHOLD": {"type": "float", "required": False},
        "IDEMPOTENCY_TTL_SEC": {"type": "integer", "required": False},
        "FAILSAFE_MODE": {"type": "string", "required": False, "enum": ["suppress", "fallback_rules", "retry"]},
        "MAX_SEND_RETRIES": {"type": "integer", "required": False},
        "AUTO_SEND": {"type": "boolean", "required": False},
        "channels": {"type": "array", "required": False},
        "created_at": {"type": "datetime", "required": True},
        "updated_at": {"type": "datetime", "required": True}
    }
}

# Function to initialize the database schema
def initialize_notification_db_schema(db_manager):
    """
    Initialize the database collections and indexes for the notification system.
    """
    try:
        # Get the collections
        collections = [
            CONVERSATIONS_SCHEMA,
            SUMMARIES_SCHEMA,
            NOTIFICATIONS_SCHEMA,
            NOTIFICATION_IDEMPOTENCY_SCHEMA,
            NOTIFICATION_FEEDBACK_SCHEMA,
            OWNER_PREFERENCES_SCHEMA
        ]
        
        for schema in collections:
            collection_name = schema["collection_name"]
            indexes = schema["indexes"]
            
            # Get or create collection
            collection = db_manager.get_collection(collection_name)
            
            # Create indexes
            for index in indexes:
                try:
                    collection.create_index(index["keys"], unique=index.get("unique", False))
                except Exception as e:
                    logger.warning(f"Failed to create index for {collection_name}: {e}")
            
            logger.info(f"Initialized collection: {collection_name}")
            
        logger.info("All notification database collections initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize notification database schema: {e}")
        return False

# Function to validate a document against a schema
def validate_document(document: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate a document against a schema.
    """
    try:
        fields = schema.get("fields", {})
        for field_name, field_spec in fields.items():
            # Skip internal fields
            if field_name.startswith("_"):
                continue
                
            # Check required fields
            if field_spec.get("required", False) and field_name not in document:
                logger.warning(f"Required field {field_name} missing from document")
                return False
                
            # Check field type if present
            if field_name in document:
                value = document[field_name]
                expected_type = field_spec.get("type")
                
                # Type checking
                if expected_type == "string" and not isinstance(value, str):
                    logger.warning(f"Field {field_name} should be string, got {type(value)}")
                    return False
                elif expected_type == "integer" and not isinstance(value, int):
                    logger.warning(f"Field {field_name} should be integer, got {type(value)}")
                    return False
                elif expected_type == "float" and not isinstance(value, float):
                    # Allow int to float conversion
                    if not isinstance(value, (int, float)):
                        logger.warning(f"Field {field_name} should be float, got {type(value)}")
                        return False
                elif expected_type == "boolean" and not isinstance(value, bool):
                    logger.warning(f"Field {field_name} should be boolean, got {type(value)}")
                    return False
                elif expected_type == "datetime" and not isinstance(value, datetime):
                    logger.warning(f"Field {field_name} should be datetime, got {type(value)}")
                    return False
                elif expected_type == "object" and not isinstance(value, dict):
                    logger.warning(f"Field {field_name} should be object, got {type(value)}")
                    return False
                elif expected_type == "array" and not isinstance(value, list):
                    logger.warning(f"Field {field_name} should be array, got {type(value)}")
                    return False
                    
                # Check enum values
                enum_values = field_spec.get("enum")
                if enum_values and value not in enum_values:
                    logger.warning(f"Field {field_name} value {value} not in allowed enum values {enum_values}")
                    return False
                    
        return True
    except Exception as e:
        logger.error(f"Error validating document: {e}")
        return False

# Example usage
if __name__ == "__main__":
    # This is just for documentation purposes
    print("Notification Database Schema Definitions:")
    print("- Conversations collection for tracking conversation states")
    print("- Summaries collection for storing conversation summaries")
    print("- Notifications collection for storing notification decisions")
    print("- Notification idempotency collection for preventing duplicate notifications")
    print("- Notification feedback collection for storing owner feedback")
    print("- Owner preferences collection for per-owner configuration")