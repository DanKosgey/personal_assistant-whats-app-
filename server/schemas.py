from pydantic import BaseModel
from typing import Optional, Dict, Any


class ContactInfo(BaseModel):
    phone: str
    name: Optional[str]


class EnhancedMessage(BaseModel):
    id: Optional[str]
    text: str
    metadata: Optional[Dict[str, Any]]


class ConversationSummary(BaseModel):
    conversation_id: str
    summary: Optional[str]
