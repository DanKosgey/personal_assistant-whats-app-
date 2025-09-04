from fastapi import APIRouter, Request, Query, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse, JSONResponse
from ..config import config
from ..utils import get_logger
from ..services.processor import MessageProcessor
from ..ai import AdvancedAIHandler
from ..clients import EnhancedWhatsAppClient

router = APIRouter()
logger = get_logger(__name__)


@router.get("/webhook")
async def webhook_verify(
    hub_mode: str = Query(alias="hub.mode"),
    hub_challenge: str = Query(alias="hub.challenge"), 
    hub_verify_token: str = Query(alias="hub.verify_token")
):
    """WhatsApp webhook verification with enhanced logging"""
    logger.info("🔍 Webhook verification requested")
    logger.info(f"Mode: {hub_mode}, Token: {hub_verify_token[:10]}...")

    if hub_mode == "subscribe" and hub_verify_token == config.WEBHOOK_VERIFY_TOKEN:
        logger.info("✅ Webhook verification successful")
        return PlainTextResponse(content=hub_challenge)
    else:
        logger.error("❌ Webhook verification failed")
        raise HTTPException(status_code=403, detail="Verification failed")


async def _process_webhook_payload(app, data: dict):
    """Extract messages from the webhook payload and process them."""
    try:
        entries = data.get("entry", []) or []
        for ent in entries:
            changes = ent.get("changes", []) or []
            for ch in changes:
                value = ch.get("value", {}) or {}
                logger.info("Processing webhook change with value: %s", value)
                # If this change contains only delivery/status updates, skip it.
                if value.get("statuses") and not value.get("messages"):
                    logger.info("Skipping webhook change because it contains only statuses (delivery/read updates)")
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
                for m in messages:
                    sender = m.get("from") or m.get("from")
                    # Some webhook payloads include an 'author' field for messages sent via templates or system
                    # Check that too and treat it as the originator when present.
                    author = m.get("author")
                    origin = author or sender
                    # Ignore messages that originate from our own WhatsApp phone number
                    # to avoid processing outgoing messages that the API can send back
                    # to our webhook and create reply loops.
                    if origin and origin == config.WHATSAPP_PHONE_NUMBER_ID:
                        logger.info("Ignoring message from our own phone_number_id/author: %s", origin)
                        continue
                    # handle text messages
                    text = None
                    if m.get("type") == "text":
                        text = (m.get("text") or {}).get("body")
                    else:
                        # try to extract a sensible fallback
                        text = (m.get("text") or {}).get("body") if isinstance(m.get("text"), dict) else None

                    if not sender or not text:
                        continue

                    # Use app.state.ai / whatsapp if available (set during startup), else create local instances
                    ai = getattr(app.state, "ai", None) or AdvancedAIHandler()
                    wa = getattr(app.state, "whatsapp", None) or EnhancedWhatsAppClient()

                    proc = MessageProcessor(ai=ai, whatsapp=wa)
                    try:
                        await proc.process(sender, text)
                    except Exception as e:
                        logger.exception("Error processing message from %s: %s", sender, e)
    except Exception as e:
        logger.exception("Webhook payload processing failed: %s", e)


@router.post("/webhook")
async def webhook_receive(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    # Schedule background processing so we acknowledge quickly to WhatsApp
    background_tasks.add_task(_process_webhook_payload, request.app, data)
    return JSONResponse({"status": "queued"})
