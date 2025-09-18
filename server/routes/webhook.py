from fastapi import APIRouter, Request, Query, HTTPException, BackgroundTasks, Header
from fastapi.responses import PlainTextResponse, JSONResponse
from ..config import config
from ..utils import get_logger
from ..services.processor_refactored import MessageProcessor
from ..ai import AdvancedAIHandler
from ..clients import EnhancedWhatsAppClient
from ..cache import cache_manager
import hmac
import hashlib
import os
import asyncio
from typing import Optional
import time

router = APIRouter()
logger = get_logger(__name__)

# In-memory processed message tracking for immediate deduplication
_processed_messages = {}
_processing_messages = set()
_last_cleanup = time.time()

# Webhook event deduplication - prevents multiple processing of same event
_processed_webhook_events = {}
_webhook_event_cleanup_interval = 180  # 3 minutes
_webhook_event_ttl = 300  # 5 minutes
_last_webhook_cleanup = time.time()

# Statistics tracking
_stats = {
    "total_webhooks": 0,
    "total_messages": 0,
    "duplicates_blocked": 0,
    "processing_blocked": 0,
    "successfully_processed": 0,
    "failed_processing": 0,
    "webhook_events_blocked": 0,  # New: Track webhook event deduplication
    "last_reset": time.time()
}

CLEANUP_INTERVAL = 300  # 5 minutes
MESSAGE_TTL = 3600      # 1 hour
STATS_LOG_INTERVAL = 600  # 10 minutes

def _log_stats():
    """Log current processing statistics with enhanced detail"""
    current_time = time.time()
    if current_time - _stats["last_reset"] > STATS_LOG_INTERVAL:
        uptime = current_time - _stats["last_reset"]
        
        # Calculate success rate
        total_attempts = _stats["successfully_processed"] + _stats["failed_processing"]
        success_rate = (_stats["successfully_processed"] / total_attempts * 100) if total_attempts > 0 else 0
        
        # Calculate deduplication efficiency
        total_blocked = _stats["duplicates_blocked"] + _stats["processing_blocked"] + _stats["webhook_events_blocked"]
        total_received = _stats["total_messages"] + total_blocked
        dedup_rate = (total_blocked / total_received * 100) if total_received > 0 else 0
        
        logger.info(
            "📊 Webhook Processing Stats (last %.1f min): "
            "webhooks=%d, messages=%d, duplicates_blocked=%d, processing_blocked=%d, "
            "successful=%d, failed=%d, webhook_events_blocked=%d, in_memory_cache=%d | "
            "success_rate=%.1f%%, dedup_efficiency=%.1f%%",
            uptime / 60,
            _stats["total_webhooks"],
            _stats["total_messages"],
            _stats["duplicates_blocked"],
            _stats["processing_blocked"],
            _stats["successfully_processed"],
            _stats["failed_processing"],
            _stats["webhook_events_blocked"],
            len(_processed_messages),
            success_rate,
            dedup_rate
        )
        
        # Log cache status
        logger.debug("🗃️ Cache status: processed_messages=%d, processing_messages=%d, webhook_events=%d",
                    len(_processed_messages), len(_processing_messages), len(_processed_webhook_events))
        
        # Reset stats for next interval
        for key in _stats:
            if key != "last_reset":
                _stats[key] = 0
        _stats["last_reset"] = current_time

def _cleanup_processed_messages():
    """Clean up old processed message entries"""
    global _last_cleanup
    current_time = time.time()
    if current_time - _last_cleanup > CLEANUP_INTERVAL:
        cutoff_time = current_time - MESSAGE_TTL
        keys_to_remove = [k for k, v in _processed_messages.items() if v < cutoff_time]
        for key in keys_to_remove:
            _processed_messages.pop(key, None)
        _last_cleanup = current_time
        if keys_to_remove:
            logger.debug("🧹 Cleaned up %d old processed message entries (keeping %d active)", 
                        len(keys_to_remove), len(_processed_messages))

def _cleanup_webhook_events():
    """Clean up old webhook event entries to prevent memory growth"""
    global _last_webhook_cleanup
    current_time = time.time()
    if current_time - _last_webhook_cleanup > _webhook_event_cleanup_interval:
        cutoff_time = current_time - _webhook_event_ttl
        keys_to_remove = [k for k, v in _processed_webhook_events.items() if v < cutoff_time]
        for key in keys_to_remove:
            _processed_webhook_events.pop(key, None)
        _last_webhook_cleanup = current_time
        if keys_to_remove:
            logger.debug("🧹 Cleaned up %d old webhook event entries (keeping %d active)", 
                        len(keys_to_remove), len(_processed_webhook_events))


@router.get("/webhook")
async def webhook_verify(
    hub_mode: str = Query(alias="hub.mode"),
    hub_challenge: str = Query(alias="hub.challenge"), 
    hub_verify_token: str = Query(alias="hub.verify_token")
):
    """WhatsApp webhook verification with enhanced logging"""
    logger.info("🔍 Webhook verification requested")
    logger.info("Mode: %s, token_present=%s", hub_mode, bool(hub_verify_token))

    if hub_mode == "subscribe" and hub_verify_token == config.WEBHOOK_VERIFY_TOKEN:
        logger.info("✅ Webhook verification successful")
        return PlainTextResponse(content=hub_challenge)
    else:
        logger.error("❌ Webhook verification failed")
        raise HTTPException(status_code=403, detail="Verification failed")


async def _process_webhook_payload(app, data: dict):
    """Extract messages from the webhook payload and process them with deduplication."""
    start_time = time.time()
    logger.info("🚀 Starting background webhook payload processing")
    
    try:
        _cleanup_processed_messages()  # Periodic cleanup
        _cleanup_webhook_events()      # Clean webhook event duplicates
        _log_stats()  # Periodic stats logging
        
        _stats["total_webhooks"] += 1
        logger.debug("📈 Total webhooks processed: %d", _stats["total_webhooks"])
        
        # Generate webhook event hash for deduplication
        import hashlib
        import json
        webhook_hash = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        current_time = time.time()
        logger.debug("🔐 Generated webhook event hash: %s", webhook_hash[:12])
        
        # Check if this exact webhook event was already processed recently
        if webhook_hash in _processed_webhook_events:
            time_since_processed = current_time - _processed_webhook_events[webhook_hash]
            if time_since_processed < _webhook_event_ttl:
                logger.warning("🔁 Skipping duplicate webhook event (hash: %s, processed %.1fs ago)", 
                           webhook_hash[:12], time_since_processed)
                _stats["webhook_events_blocked"] += 1
                return
            else:
                logger.debug("🔄 Webhook event hash exists but outside TTL window (%.1fs ago)", time_since_processed)
        
        # Mark this webhook event as processed
        _processed_webhook_events[webhook_hash] = current_time
        logger.debug("✅ Webhook event hash registered: %s", webhook_hash[:12])
        
        entries = data.get("entry", []) or []
        logger.info("📦 Processing webhook with %d entries", len(entries))
        
        total_messages_found = 0
        total_statuses_found = 0
        
        for entry_idx, ent in enumerate(entries):
            entry_id = ent.get("id", "unknown")
            changes = ent.get("changes", []) or []
            logger.debug("📋 Entry %d (ID: %s) has %d changes", entry_idx + 1, entry_id, len(changes))
            
            for change_idx, ch in enumerate(changes):
                value = ch.get("value", {}) or {}
                field = ch.get("field", "unknown")
                evt_types = list(value.keys())
                msg_count = len(value.get("messages", []) or [])
                status_count = len(value.get("statuses", []) or [])
                
                total_messages_found += msg_count
                total_statuses_found += status_count
                
                logger.info("🔄 Processing webhook change %d/%d (field=%s, keys=%s, messages=%d, statuses=%d)", 
                           change_idx + 1, len(changes), field, evt_types, msg_count, status_count)
                
                # CRITICAL: Skip status-only webhooks to reduce noise and prevent unnecessary processing
                if value.get("statuses") and not value.get("messages"):
                    logger.info("⏭️ Skipping webhook change because it contains only statuses (delivery/read updates)")
                    logger.debug("📅 Status details: %d status updates found", len(value.get("statuses", [])))
                    continue
                
                # If the webhook metadata indicates the event is for our own phone number
                # (e.g. metadata.phone_number_id matches our configured id), skip to avoid
                # processing messages originating from the business account itself.
                meta = value.get("metadata") or {}
                if meta.get("phone_number_id") and meta.get("phone_number_id") == config.WHATSAPP_PHONE_NUMBER_ID:
                    # There may still be inbound messages for customers under that phone_number_id;
                    # only skip if there are no external messages to process in this change.
                    msgs_present = bool(value.get("messages"))
                    if not msgs_present:
                        logger.info("Skipping webhook change for our phone_number_id (no inbound messages present)")
                        continue
                
                messages = value.get("messages", []) or []
                _stats["total_messages"] += len(messages)
                
                logger.info("💬 Found %d messages to process in this change", len(messages))
                
                # Add small delay between processing multiple messages to prevent race conditions
                if len(messages) > 1:
                    logger.info("🕑 Processing %d messages with staggered timing to prevent race conditions", len(messages))
                
                for idx, m in enumerate(messages):
                    message_id = m.get("id", "unknown")
                    message_type = m.get("type", "unknown")
                    
                    # CRITICAL: Extract message_id for deduplication
                    message_id = m.get("id")
                    if not message_id:
                        logger.warning("⚠️ Skipping message without ID: %s", m)
                        continue
                    
                    logger.debug("🔍 Starting deduplication checks for message: %s", message_id)
                    
                    # Check persistent cache for processed messages (survives server restarts)
                    if await cache_manager.is_message_processed(message_id):
                        logger.info("💫 Skipping duplicate message ID: %s (found in persistent cache)", message_id)
                        _stats["duplicates_blocked"] += 1
                        continue
                    
                    # Check if this message is currently being processed (in cache)
                    if await cache_manager.is_message_processing(message_id):
                        logger.info("⏳ Skipping message ID: %s (currently being processed - from cache)", message_id)
                        _stats["processing_blocked"] += 1
                        continue
                    
                    # Double-check with in-memory tracking for immediate deduplication
                    current_time = time.time()
                    if message_id in _processed_messages:
                        logger.info("💫 Skipping duplicate message ID: %s (already processed at %s)", 
                                   message_id, _processed_messages[message_id])
                        _stats["duplicates_blocked"] += 1
                        continue
                    
                    # Check if this message is currently being processed (in-memory)
                    if message_id in _processing_messages:
                        logger.info("⏳ Skipping message ID: %s (currently being processed - in memory)", message_id)
                        _stats["processing_blocked"] += 1
                        continue
                    
                    logger.info("✅ Message passed all deduplication checks - proceeding with processing")
                    
                    # Mark as being processed in both cache and memory
                    await cache_manager.mark_message_processing(message_id, ttl=300)  # 5 minutes
                    _processing_messages.add(message_id)
                    logger.debug("🔒 Message marked as processing in both cache and memory")
                    
                    sender = None  # Initialize sender variable
                    message_processing_start = time.time()
                    
                    try:
                        sender = m.get("from")
                        logger.debug("📱 Message sender: %s", sender)
                        
                        # Skip owner messages from being processed through consent workflow
                        if sender == os.getenv('PA_OWNER_NUMBER'):
                            logger.info("⏭️ Owner message detected - skipping standard processing")
                            continue
                        
                        # Some webhook payloads include an 'author' field for messages sent via templates or system
                        # Check that too and treat it as the originator when present.
                        author = m.get("author")
                        origin = author or sender
                        logger.debug("🔍 Message origin analysis: sender=%s, author=%s, origin=%s", sender, author, origin)
                        
                        # Ignore messages that originate from our own WhatsApp phone number
                        # to avoid processing outgoing messages that the API can send back
                        # to our webhook and create reply loops.
                        if origin and origin == config.WHATSAPP_PHONE_NUMBER_ID:
                            logger.info("🙅‍♂️ Ignoring message from our own phone_number_id/author: %s", origin)
                            continue

                        # handle text messages
                        text = None
                        if m.get("type") == "text":
                            text = (m.get("text") or {}).get("body")
                            logger.debug("💬 Text message content length: %d characters", len(text) if text else 0)
                        else:
                            # try to extract a sensible fallback
                            text = (m.get("text") or {}).get("body") if isinstance(m.get("text"), dict) else None
                            logger.debug("📋 Non-text message type: %s, extracted text: %s", m.get("type"), bool(text))

                        if not sender or not text:
                            logger.warning("⚠️ Skipping message with missing sender or text: sender=%s, text=%s", sender, bool(text))
                            continue
                        
                        logger.info("🚀 Starting message processing: from=%s, message_id=%s, text_length=%d", 
                                   sender, message_id, len(text))

                        # Use app.state.message_processor if available (set during startup), else create local instances
                        proc = getattr(app.state, "message_processor", None)
                        if proc:
                            logger.debug("🛠️ Using global MessageProcessor with memory system")
                        else:
                            # Fallback: create local instances
                            ai = getattr(app.state, "ai", None) or AdvancedAIHandler()
                            wa = getattr(app.state, "whatsapp", None) or EnhancedWhatsAppClient()
                            proc = MessageProcessor(ai=ai, whatsapp=wa)
                            
                            # Initialize memory system with the processor
                            from ..routes.memory import set_message_processor
                            set_message_processor(proc)
                            
                            logger.debug("🛠️ Created fallback MessageProcessor with memory system")
                        
                        # Process the message
                        logger.info("💬 Processing message content with AI...")                        
                        await proc.process(sender, text, message_id=message_id)
                        
                        processing_time = (time.time() - message_processing_start) * 1000
                        logger.info("✅ Message processing completed in %.1fms", processing_time)
                        
                        # Mark as successfully processed in both cache and memory
                        await cache_manager.mark_message_processed(message_id, ttl=3600)  # 1 hour
                        _processed_messages[message_id] = current_time
                        _stats["successfully_processed"] += 1
                        logger.info("✅ Successfully processed message ID: %s from %s (total processing time: %.1fms)", 
                                   message_id, sender, processing_time)
                        
                    except Exception as e:
                        processing_time = (time.time() - message_processing_start) * 1000
                        _stats["failed_processing"] += 1
                        logger.exception("❌ Error processing message ID %s from %s (failed after %.1fms): %s", 
                                       message_id, sender or "unknown", processing_time, e)
                    finally:
                        # Always remove from processing sets (both cache and memory)
                        await cache_manager.unmark_message_processing(message_id)
                        _processing_messages.discard(message_id)
                        logger.debug("🔓 Removed message %s from processing sets", message_id)
                        
        # Log summary of webhook processing
        processing_time = (time.time() - start_time) * 1000
        logger.info("🏁 Webhook payload processing complete: total_time=%.1fms, messages_found=%d, statuses_found=%d, entries=%d", 
                   processing_time, total_messages_found, total_statuses_found, len(entries))
                        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.exception("❌ Webhook payload processing failed after %.1fms: %s", processing_time, e)


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


@router.get("/webhook/stats")
async def webhook_stats():
    """Get webhook processing statistics"""
    current_time = time.time()
    uptime = current_time - _stats["last_reset"]
    
    # Get cache stats if available
    cache_stats = {}
    try:
        # Try to get some basic cache info
        cache_stats["cache_type"] = "redis" if cache_manager._redis else "memory"
        cache_stats["cache_connected"] = cache_manager._redis is not None
    except:
        cache_stats["cache_type"] = "unknown"
    
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "uptime_minutes": uptime / 60,
        "webhook_stats": _stats.copy(),
        "deduplication": {
            "in_memory_processed": len(_processed_messages),
            "currently_processing": len(_processing_messages),
            "webhook_events_processed": len(_processed_webhook_events),
            "cleanup_interval_seconds": CLEANUP_INTERVAL,
            "message_ttl_seconds": MESSAGE_TTL,
            "webhook_event_ttl_seconds": _webhook_event_ttl
        },
        "cache": cache_stats,
        "timestamp": current_time
    }


@router.post("/webhook")
async def webhook_receive(request: Request, background_tasks: BackgroundTasks, x_hub_signature_256: Optional[str] = Header(None, convert_underscored_headers=False)):
    """Main WhatsApp webhook handler with signature verification and deduplication"""
    start_time = time.time()
    logger.info("🔔 Main webhook handler called - POST /api/webhook")
    logger.debug("🔍 Request headers: %s", dict(request.headers))
    
    body = await request.body()
    body_size = len(body)
    logger.debug("📄 Request body size: %d bytes", body_size)
    
    # CRITICAL: Add immediate request-level deduplication
    import hashlib
    request_hash = hashlib.md5(body).hexdigest()
    current_time = time.time()
    logger.debug("🔐 Generated request hash: %s", request_hash[:12])
    
    # Check if this exact request was processed very recently (within 10 seconds)
    if request_hash in _processed_webhook_events:
        time_since_processed = current_time - _processed_webhook_events[request_hash]
        if time_since_processed < 10:  # 10 second window for immediate duplicates
            logger.warning("🚫 BLOCKED: Duplicate webhook request (hash: %s, %.1fs ago) - immediate deduplication", 
                           request_hash[:12], time_since_processed)
            return JSONResponse({"status": "duplicate_blocked"})
        else:
            logger.debug("🔄 Request hash exists but outside deduplication window (%.1fs ago)", time_since_processed)
    
    # Mark this request as processed immediately
    _processed_webhook_events[request_hash] = current_time
    logger.debug("✅ Request hash registered: %s", request_hash[:12])
    
    # Verify signature
    valid = _verify_signature(body, x_hub_signature_256)
    if not valid:
        logger.warning("❌ Invalid webhook signature; header=%s", x_hub_signature_256)
        raise HTTPException(status_code=403, detail="Invalid signature")
    else:
        logger.debug("✅ Webhook signature validated successfully")
    
    try:
        data = await request.json()
        logger.debug("📋 JSON parsed successfully - object type: %s", data.get('object', 'unknown'))
    except Exception as e:
        logger.exception("❌ Failed to parse webhook JSON: %s", e)
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    # Quick debug log (redact sensitive fields)
    try:
        import re
        summary = str(data)[:1000]
        summary = re.sub(r"\+?\d[\d\s\-()]{6,}\d", "<redacted:phone>", summary)
        summary = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<redacted:email>", summary)
        logger.info("📦 Received webhook event: %s", summary)
        
        # Additional debug info about the webhook
        entries = data.get("entry", [])
        total_changes = sum(len(entry.get("changes", [])) for entry in entries)
        logger.debug("📈 Webhook contains %d entries with %d total changes", len(entries), total_changes)
        
    except Exception as e:
        logger.warning("⚠️ Unable to summarize webhook event: %s", e)
        logger.info("Received webhook event (unable to summarize)")
    
    # Schedule background processing so we acknowledge quickly to WhatsApp
    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    logger.info("✅ Webhook acknowledged in %.1fms - scheduling background processing", processing_time)
    background_tasks.add_task(_process_webhook_payload, request.app, data)
    return JSONResponse({"status": "queued", "processing_time_ms": round(processing_time, 1)})


@router.get("/webhook/debug")
async def webhook_debug():
    """Debug endpoint to check webhook processing state"""
    current_time = time.time()
    return {
        "webhook_handler": "active",
        "current_time": current_time,
        "stats": _stats.copy(),
        "in_memory_state": {
            "processed_messages": len(_processed_messages),
            "processing_messages": len(_processing_messages),
            "webhook_events": len(_processed_webhook_events),
            "sample_processed_messages": list(_processed_messages.keys())[:5],
            "sample_webhook_events": list(_processed_webhook_events.keys())[:3]
        },
        "config": {
            "cleanup_interval": CLEANUP_INTERVAL,
            "message_ttl": MESSAGE_TTL,
            "webhook_event_ttl": _webhook_event_ttl,
            "stats_log_interval": STATS_LOG_INTERVAL
        }
    }
