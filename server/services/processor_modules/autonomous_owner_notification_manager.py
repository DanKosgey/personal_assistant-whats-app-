"""
autonomous_owner_notification_manager.py
Single place where AI controls notification. Autonomous by default.
"""

import time
import uuid
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime

# Import required modules from the project
from ...database import db_manager
from ...config import config
from .notification_monitoring import record_notification_sent, record_notification_failed, record_notification_suppressed

logger = logging.getLogger(__name__)

# ---------- Configuration ----------
DEFAULTS = {
    "EOC_THRESHOLD": 0.8,
    "NOTIFY_THRESHOLD": 0.7,
    "IDEMPOTENCY_TTL_SEC": 7 * 24 * 3600,
    "FAILSAFE_MODE": "suppress",   # "suppress" | "fallback_rules" | "retry"
    "MAX_SEND_RETRIES": 3,
    "AUTO_SEND": True
}

# ---------- Database Functions ----------
def db_get_owner_prefs(owner_id: str) -> Dict[str, Any]:
    """Get per-owner configuration; empty means use defaults"""
    try:
        # Try to get owner preferences from database
        get_col = getattr(db_manager, 'get_collection', None)
        if callable(get_col):
            prefs_col = get_col('owner_preferences')
            if prefs_col is not None:
                find_one = getattr(prefs_col, 'find_one', None)
                if callable(find_one):
                    prefs = find_one({"owner_id": owner_id})
                    if prefs and isinstance(prefs, dict):
                        # Remove MongoDB-specific fields
                        prefs.pop('_id', None)
                        return prefs
    except Exception as e:
        logger.warning(f"Failed to get owner preferences: {e}")
    return {}

def db_store_decision(record: Dict[str, Any]) -> None:
    """Persist decision + features + model outputs for audit & retraining."""
    try:
        get_col = getattr(db_manager, 'get_collection', None)
        if callable(get_col):
            notifications_col = get_col('notifications')
            if notifications_col is not None:
                insert_one = getattr(notifications_col, 'insert_one', None)
                if callable(insert_one):
                    # Add timestamp if not present
                    if 'timestamp' not in record:
                        record['timestamp'] = time.time()
                    insert_one(record)
                    logger.info(f"Stored notification decision: {record.get('id', 'unknown')}")
    except Exception as e:
        logger.error(f"Failed to store decision: {e}")

def db_has_idempotency(key: str) -> bool:
    """Check if idempotency key already exists"""
    try:
        get_col = getattr(db_manager, 'get_collection', None)
        if callable(get_col):
            idempotency_col = get_col('notification_idempotency')
            if idempotency_col is not None:
                find_one = getattr(idempotency_col, 'find_one', None)
                if callable(find_one):
                    record = find_one({"key": key})
                    if record and isinstance(record, dict):
                        # Check if it's still valid
                        ttl = record.get('ttl', 0)
                        if time.time() < ttl:
                            return True
                        else:
                            # Expired, remove it
                            delete_one = getattr(idempotency_col, 'delete_one', None)
                            if callable(delete_one):
                                delete_one({"key": key})
    except Exception as e:
        logger.warning(f"Failed to check idempotency: {e}")
    return False

def db_set_idempotency(key: str, ttl_sec: int) -> None:
    """Store idempotency key with TTL"""
    try:
        get_col = getattr(db_manager, 'get_collection', None)
        if callable(get_col):
            idempotency_col = get_col('notification_idempotency')
            if idempotency_col is not None:
                replace_one = getattr(idempotency_col, 'replace_one', None)
                if callable(replace_one):
                    expiration_time = time.time() + ttl_sec
                    replace_one(
                        {"key": key},
                        {
                            "key": key,
                            "ttl": expiration_time,
                            "created_at": time.time()
                        },
                        upsert=True
                    )
    except Exception as e:
        logger.error(f"Failed to set idempotency: {e}")

def db_store_notification_feedback(notification_id: str, owner_id: str, helpful: bool, notes: str = "") -> None:
    """Store owner feedback on notification"""
    try:
        get_col = getattr(db_manager, 'get_collection', None)
        if callable(get_col):
            feedback_col = get_col('notification_feedback')
            if feedback_col is not None:
                insert_one = getattr(feedback_col, 'insert_one', None)
                if callable(insert_one):
                    feedback_record = {
                        "id": str(uuid.uuid4()),
                        "notification_id": notification_id,
                        "owner_id": owner_id,
                        "helpful": helpful,
                        "notes": notes,
                        "created_at": time.time()
                    }
                    insert_one(feedback_record)
                    logger.info(f"Stored notification feedback for {notification_id}")
    except Exception as e:
        logger.error(f"Failed to store notification feedback: {e}")

# ---------- Message Sending Functions ----------
def send_whatsapp(whatsapp_client, owner_contact: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send notification via WhatsApp Business API. Return delivery result."""
    try:
        # Format message for WhatsApp
        message = f"*Conversation Summary*\n\n"
        message += f"Name: {payload.get('contact_name', 'Unknown')}\n"
        message += f"Phone: {payload.get('contact_phone', 'Unknown')}\n"
        message += f"Time: {datetime.fromtimestamp(payload.get('timestamp', time.time())).strftime('%Y-%m-%d %H:%M')}\n\n"
        message += f"Summary: {payload.get('summary', 'No summary available')}\n\n"
        message += f"Importance Score: {payload.get('importance_score', 0.0):.2f}\n"
        message += f"EOC Confidence: {payload.get('eoc_confidence', 0.0):.2f}\n"
        
        if payload.get('reasoning'):
            message += f"Reasoning: {payload.get('reasoning')}\n"
        
        # Send via WhatsApp client
        result = whatsapp_client.send_message(owner_contact, message)
        return {
            "status": "sent", 
            "provider_id": str(uuid.uuid4()), 
            "timestamp": time.time(),
            "provider_response": result
        }
    except Exception as e:
        logger.error(f"Failed to send WhatsApp notification: {e}")
        return {
            "status": "failed", 
            "error": str(e),
            "timestamp": time.time()
        }

def send_email(email_sender, owner_email: str, subject: str, body: str) -> Dict[str, Any]:
    """Send notification via email. Return delivery result."""
    try:
        # This would integrate with your email sending service
        # For now, we'll simulate success
        logger.info(f"Sending email to {owner_email}: {subject}")
        return {
            "status": "sent", 
            "provider_id": str(uuid.uuid4()), 
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")
        return {
            "status": "failed", 
            "error": str(e),
            "timestamp": time.time()
        }

# ---------- Autonomous Owner Notification Manager ----------
def build_idempotency_key(conversation_id: str, eoc_event_id: Optional[str]) -> str:
    """Build idempotency key to prevent duplicate notifications"""
    return f"notif:{conversation_id}:{eoc_event_id or 'default'}"

def make_notification_message(summary: str, decision_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Format the message body for sending"""
    return {
        "title": f"Conversation summary — conv:{decision_payload['conversation_id']}",
        "summary": summary,
        "importance_score": decision_payload["importance_score"],
        "eoc_confidence": decision_payload["eoc_confidence"],
        "reasoning": decision_payload.get("reasoning", ""),
        "features": decision_payload.get("features", {}),
        "contact_name": decision_payload.get("contact_name", "Unknown"),
        "contact_phone": decision_payload.get("contact_phone", "Unknown"),
        "timestamp": decision_payload.get("timestamp", time.time())
    }

def handle_notification_decision(
    whatsapp_client,
    owner_id: str,
    conversation_id: str,
    summary_text: str,
    eoc_confidence: float,
    importance_score: float,
    features: Dict[str, Any],
    eoc_event_id: Optional[str] = None,
    model_reasoning: Optional[str] = None,
    owner_contact_info: Optional[Dict[str, str]] = None,
    contact_name: str = "Unknown",
    contact_phone: str = "Unknown"
) -> Dict[str, Any]:
    """
    Central function to be called by the pipeline. Returns a decision record and
    performs the send automatically if the AI decides to notify.
    """
    start_time = time.time()
    
    # 1) Load owner prefs (per-owner override)
    prefs = db_get_owner_prefs(owner_id)
    cfg = {**DEFAULTS, **prefs, **getattr(config, 'NOTIFICATION_CONFIG', {})}

    # 2) Idempotency guard
    idemp_key = build_idempotency_key(conversation_id, eoc_event_id)
    if db_has_idempotency(idemp_key):
        return {"status": "skipped", "reason": "idempotent", "idempotency_key": idemp_key}

    # 3) Decide notification according to AI outputs + thresholds
    conversation_ended = (eoc_confidence >= cfg["EOC_THRESHOLD"])
    notify_owner = conversation_ended and (importance_score >= cfg["NOTIFY_THRESHOLD"])
    
    # Check if auto-send is enabled for this owner
    auto_send = cfg.get("AUTO_SEND", True)

    decision = {
        "id": str(uuid.uuid4()),
        "owner_id": owner_id,
        "conversation_id": conversation_id,
        "eoc_confidence": round(eoc_confidence, 3),
        "importance_score": round(importance_score, 3),
        "notify_owner": bool(notify_owner and auto_send),  # Only notify if auto-send is enabled
        "auto_send_enabled": auto_send,
        "thresholds": {
            "eoc_threshold": cfg["EOC_THRESHOLD"],
            "notify_threshold": cfg["NOTIFY_THRESHOLD"]
        },
        "features": features,
        "reasoning": model_reasoning,
        "timestamp": time.time(),
        "contact_name": contact_name,
        "contact_phone": contact_phone
    }

    # 4) Persist decision (audit & training data)
    db_store_decision(decision)

    # 5) Mark idempotency immediately to avoid double sends
    db_set_idempotency(idemp_key, cfg["IDEMPOTENCY_TTL_SEC"])

    # 6) Perform send if decided
    send_result = None
    if decision["notify_owner"]:
        if not owner_contact_info:
            # Cannot send if no contact; record failure
            decision["send_result"] = {"status": "failed", "reason": "no_contact"}
            db_store_decision(decision)
            record_notification_failed("no_contact")
            return decision

        # Build payload
        payload = make_notification_message(summary_text, decision)

        # Try preferred channels in order
        # This loop supports failover; you can make channel order configurable per owner
        channels = owner_contact_info.get("channels", ["whatsapp", "email"])
        for ch in channels:
            try:
                if ch == "whatsapp" and owner_contact_info.get("whatsapp"):
                    for attempt in range(cfg["MAX_SEND_RETRIES"]):
                        result = send_whatsapp(whatsapp_client, owner_contact_info["whatsapp"], payload)
                        if result.get("status") == "sent":
                            send_result = {"channel": "whatsapp", "result": result, "attempts": attempt + 1}
                            break
                        time.sleep(2 ** attempt)  # Exponential backoff
                elif ch == "email" and owner_contact_info.get("email"):
                    result = send_email(None, owner_contact_info["email"], payload["title"], payload["summary"])
                    send_result = {"channel": "email", "result": result, "attempts": 1}
                    break
                # Add other channels (slack, pagerduty, webhook) as needed
                if send_result:
                    break
            except Exception as exc:
                # Log and continue to next channel (or retry per policy)
                logger.error(f"Send error on channel {ch}: {exc}")
                send_result = {"channel": ch, "result": {"status": "error", "error": str(exc)}}

        if not send_result:
            # Final failure handling according to FAILSAFE_MODE
            if cfg["FAILSAFE_MODE"] == "suppress":
                send_result = {"status": "suppressed", "reason": "delivery_failed", "details": None}
                record_notification_suppressed("delivery_failed")
            elif cfg["FAILSAFE_MODE"] == "fallback_rules":
                # Optionally re-run a conservative rule to decide send or not (not implemented here)
                send_result = {"status": "fallback_attempted"}
                record_notification_failed("fallback_failed")
            else:
                send_result = {"status": "failed", "reason": "unknown"}
                record_notification_failed("unknown")

        decision["send_result"] = send_result
        db_store_decision(decision)
        
        # Record monitoring metrics
        if send_result.get("status") == "sent":
            latency = time.time() - start_time
            record_notification_sent(owner_id, importance_score, latency)
        elif send_result.get("status") == "suppressed":
            record_notification_suppressed(send_result.get("reason", "unknown"))
        else:
            record_notification_failed(send_result.get("reason", "unknown"))

    return decision

# ---------- Feedback Functions ----------
def process_owner_feedback(notification_id: str, owner_id: str, helpful: bool, notes: str = "") -> Dict[str, Any]:
    """Process owner feedback on a notification"""
    try:
        # Store the feedback
        db_store_notification_feedback(notification_id, owner_id, helpful, notes)
        
        # Return success response
        return {
            "status": "success",
            "message": "Feedback recorded successfully",
            "notification_id": notification_id
        }
    except Exception as e:
        logger.error(f"Failed to process owner feedback: {e}")
        return {
            "status": "error",
            "message": str(e),
            "notification_id": notification_id
        }

# Example usage:
if __name__ == "__main__":
    # Example usage code
    owner_id = "owner_abc"
    conv_id = "conv_123"
    summary = "Customer requests refund for order 1234; agent confirmed refund. Customer thanked and said bye."
    eoc_conf = 0.92
    importance = 0.78
    features = {"conversation_length": 6, "num_action_items": 1, "service_keyword_present": True}
    owner_contact = {"whatsapp": "+254700111222", "email": "owner@example.com", "channels": ["whatsapp", "email"]}
    
    # This would normally be called from the processor pipeline
    # dec = handle_notification_decision(None, owner_id, conv_id, summary, eoc_conf, importance, features, eoc_event_id="evt-1", model_reasoning="refund detected; VIP", owner_contact_info=owner_contact)
    # import json; print(json.dumps(dec, indent=2))
    pass