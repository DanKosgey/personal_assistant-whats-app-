"""
Simplified WhatsApp Webhook Handler
Implements the exact flow specified:
1. Normalize incoming message (phone_number, message_text, message_id, timestamp)
2. Lookup user by phone in users table
3. If found → fetch name and personalization fields
4. If not found → mark is_new = true
5. Pass to message processor
"""

from fastapi import APIRouter, Request, Query, HTTPException, BackgroundTasks, Header
from fastapi.responses import PlainTextResponse, JSONResponse
import hmac
import hashlib
import os
import time
import json
from typing import Optional, Dict, Any
from ..config import config
from ..utils import get_logger

# Import the new simplified message processor
from ..services.simplified_processor import SimplifiedMessageProcessor

router = APIRouter()
logger = get_logger(__name__)

# In-memory tracking for immediate deduplication
_processed_messages = {}
_processing_messages = set()

def _verify_signature(body_bytes: bytes, signature_header: Optional[str]) -> bool:
    """
    Verify X-Hub-Signature-256 header (sha256=<hex>) using APP_SECRET.
    Returns True when signature matches; False otherwise.
    If APP_SECRET is not set, signature verification is considered disabled (development).
    """
    APP_SECRET = getattr(config, "APP_SECRET", None) or os.getenv("APP_SECRET", "")
    
    if not APP_SECRET:
        logger.warning("APP_SECRET not configured — webhook POST signature verification is disabled. This is unsafe in production.")
        return True  # Allow in development

    if not signature_header:
        logger.debug("Missing X-Hub-Signature-256 header")
        return False
        
    prefix = "sha256="
    if not signature_header.startswith(prefix):
        logger.debug("Malformed X-Hub-Signature-256 header: %s", signature_header)
        return False
        
    sig = signature_header[len(prefix):]
    computed = hmac.new(APP_SECRET.encode(), body_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, computed)


@router.get("/webhook")
async def webhook_verify(
    hub_mode: str = Query(alias="hub.mode"),
    hub_challenge: str = Query(alias="hub.challenge"), 
    hub_verify_token: str = Query(alias="hub.verify_token")
):
    """WhatsApp webhook verification"""
    logger.info("🔍 Webhook verification requested")
    
    if hub_mode == "subscribe" and hub_verify_token == config.WEBHOOK_VERIFY_TOKEN:
        logger.info("✅ Webhook verification successful")
        return PlainTextResponse(content=hub_challenge)
    else:
        logger.error("❌ Webhook verification failed")
        raise HTTPException(status_code=403, detail="Verification failed")


async def _process_webhook_payload(app, data: dict):
    """Process webhook payload with the simplified flow"""
    start_time = time.time()
    logger.info("🚀 Starting simplified webhook payload processing")
    
    try:
        entries = data.get("entry", []) or []
        logger.info("📦 Processing webhook with %d entries", len(entries))
        
        for entry_idx, ent in enumerate(entries):
            changes = ent.get("changes", []) or []
            logger.debug("📋 Entry %d has %d changes", entry_idx + 1, len(changes))
            
            for change_idx, ch in enumerate(changes):
                value = ch.get("value", {}) or {}
                
                # Skip status-only webhooks
                if value.get("statuses") and not value.get("messages"):
                    logger.info("⏭️ Skipping webhook change because it contains only statuses")
                    continue
                
                messages = value.get("messages", []) or []
                logger.info("💬 Found %d messages to process in this change", len(messages))
                
                for idx, m in enumerate(messages):
                    message_id = m.get("id", "unknown")
                    
                    # Deduplication check
                    if message_id in _processed_messages or message_id in _processing_messages:
                        logger.info("💫 Skipping duplicate message ID: %s", message_id)
                        continue
                    
                    # Mark as being processed
                    _processing_messages.add(message_id)
                    
                    try:
                        # Extract message data
                        sender = m.get("from")
                        timestamp = m.get("timestamp")
                        
                        # Extract message text
                        text = None
                        if m.get("type") == "text":
                            text = (m.get("text") or {}).get("body")
                        
                        if not sender or not text:
                            logger.warning("⚠️ Skipping message with missing sender or text")
                            continue
                        
                        logger.info("🚀 Processing message from %s: %s", sender, text[:50] if text else "")
                        
                        # Normalize incoming message data
                        normalized_message = {
                            "phone_number": sender,
                            "message_text": text,
                            "message_id": message_id,
                            "timestamp": timestamp
                        }
                        
                        # Use simplified message processor
                        proc = SimplifiedMessageProcessor()
                        await proc.process(normalized_message)
                        
                        # Mark as successfully processed
                        _processed_messages[message_id] = time.time()
                        logger.info("✅ Successfully processed message ID: %s from %s", message_id, sender)
                        
                    except Exception as e:
                        logger.exception("❌ Error processing message ID %s: %s", message_id, e)
                    finally:
                        # Always remove from processing set
                        _processing_messages.discard(message_id)
                        
        processing_time = (time.time() - start_time) * 1000
        logger.info("🏁 Webhook payload processing complete in %.1fms", processing_time)
                        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.exception("❌ Webhook payload processing failed after %.1fms: %s", processing_time, e)


@router.post("/webhook")
async def webhook_receive(request: Request, background_tasks: BackgroundTasks, x_hub_signature_256: Optional[str] = Header(None, convert_underscored_headers=False)):
    """Main WhatsApp webhook handler with simplified processing"""
    start_time = time.time()
    logger.info("🔔 Simplified webhook handler called - POST /api/webhook")
    
    body = await request.body()
    
    # Verify signature
    valid = _verify_signature(body, x_hub_signature_256)
    if not valid:
        logger.warning("❌ Invalid webhook signature")
        raise HTTPException(status_code=403, detail="Invalid signature")
    
    try:
        data = await request.json()
        logger.debug("📋 JSON parsed successfully")
    except Exception as e:
        logger.exception("❌ Failed to parse webhook JSON: %s", e)
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    # Schedule background processing
    processing_time = (time.time() - start_time) * 1000
    logger.info("✅ Webhook acknowledged in %.1fms - scheduling background processing", processing_time)
    background_tasks.add_task(_process_webhook_payload, request.app, data)
    return JSONResponse({"status": "queued", "processing_time_ms": round(processing_time, 1)})