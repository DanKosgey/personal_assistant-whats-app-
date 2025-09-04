from fastapi import APIRouter, HTTPException, Request
from ..ai import AdvancedAIHandler
from ..clients import EnhancedWhatsAppClient
from ..services.processor import MessageProcessor

router = APIRouter()


@router.post("/messages")
async def post_message(request: Request):
    payload = await request.json()
    sender = payload.get("from")
    text = payload.get("text")
    if not sender or not text:
        raise HTTPException(status_code=400, detail="from and text required")

    ai = AdvancedAIHandler()
    wa = EnhancedWhatsAppClient()
    proc = MessageProcessor(ai=ai, whatsapp=wa)
    result = await proc.process(sender, text)
    return result
