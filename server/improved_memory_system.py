"""
Improved Memory and Conversation Summary System for WhatsApp AI Agent

This module provides enhanced memory management and conversation summarization
with better structure, context preservation, and actionable insights.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memories stored in the system"""
    PROFILE = "profile"                    # User profile information
    PREFERENCE = "preference"              # User preferences and settings
    CONVERSATION_SUMMARY = "conversation_summary"  # Summarized conversations
    ENTITY = "entity"                      # Extracted entities (contacts, etc.)
    FACT = "fact"                         # Important facts about the user
    INTERACTION = "interaction"            # Significant interactions
    CONTEXT = "context"                   # Cross-session context

@dataclass
class MemoryEntry:
    """Individual memory entry with metadata"""
    id: str
    user_id: str
    memory_type: MemoryType
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    importance: float  # 0.0 - 1.0 relevance score
    source_ref: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "source_ref": self.source_ref
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            memory_type=MemoryType(data["memory_type"]),
            content=data["content"],
            metadata=data["metadata"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            importance=data["importance"],
            source_ref=data.get("source_ref")
        )

class ImprovedMemoryManager:
    """Enhanced memory manager with better context preservation and summarization"""
    
    def __init__(self):
        # In-memory caches
        self.short_term_memory: Dict[str, List[Dict[str, Any]]] = {}
        self.long_term_memory: Dict[str, List[MemoryEntry]] = {}
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Enhanced intent keywords for better topic identification
        self.topic_keywords = {
            "Introduction": ["my name is", "i am", "call me", "this is"],
            "Scheduling": ["schedule", "meeting", "appointment", "calendar", "book", "reschedule", "cancel", "plan", "tomorrow", "8pm"],
            "Payment": ["payment", "pay", "invoice", "bill", "cost", "price", "fee", "charge"],
            "Support": ["help", "issue", "problem", "broken", "fix", "support", "trouble"],
            "Document": ["sign", "contract", "document", "file", "report", "paper", "agreement"],
            "Order": ["order", "purchase", "buy", "product", "item", "goods"],
            "Feedback": ["feedback", "complaint", "suggestion", "review", "opinion"],
            "Project": ["project", "task", "work", "job", "assignment"],
            "Inquiry": ["what", "how", "when", "where", "why", "which", "can you", "could you"],
            "Confirmation": ["confirm", "verify", "check", "validate", "approval"],
            "Delivery": ["delivery", "shipping", "send", "mail", "post", "package"],
            "Cancellation": ["cancel", "refund", "return", "terminate", "stop"]
        }
        
        # Action item patterns for better action extraction
        self.action_patterns = {
            "Schedule meeting": [r"schedule.*meeting", r"set.*meeting", r"book.*meeting"],
            "Process payment": [r"payment", r"pay", r"invoice", r"bill"],
            "Provide information": [r"information", r"details", r"info", r"question", r"clarify"],
            "Fix issue": [r"fix", r"resolve", r"problem", r"issue", r"broken", r"error"],
            "Sign document": [r"sign", r"contract", r"agreement", r"document"],
            "Call back": [r"call", r"phone", r"contact.*phone"],
            "Review feedback": [r"feedback", r"review", r"evaluate"],
            "Process order": [r"order", r"purchase", r"buy", r"product"],
            "Resolve complaint": [r"complaint", r"dissatisfied", r"problem"],
            "Follow up": [r"follow.*up", r"opportunity", r"business"],
            "Review document": [r"document", r"file", r"report"],
            "Send email": [r"email", r"send", r"forward"],
            "Confirm details": [r"confirm", r"verify", r"check"],
            "Prepare proposal": [r"proposal", r"quote", r"estimate"]
        }
        
    async def get_or_create_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get or create user profile with default structure"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "user_id": user_id,
                "name": None,
                "language": "English",
                "timezone": "UTC",
                "preferences": {},
                "contact_info": {
                    "phone": user_id
                },
                "created_at": datetime.now().astimezone().isoformat(),
                "last_interaction": None,
                "interaction_count": 0,
                "conversation_style": "professional",
                "topics_of_interest": [],
                "context_summary": ""
            }
            logger.debug(f"Created new user profile for {user_id}")
        return self.user_profiles[user_id]
    
    async def add_short_term_memory(self, user_id: str, message: str, response: str, 
                                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add to short-term conversation memory"""
        if user_id not in self.short_term_memory:
            self.short_term_memory[user_id] = []
            
        entry = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "message": message,
            "response": response,
            "metadata": metadata or {}
        }
        
        self.short_term_memory[user_id].append(entry)
        logger.debug(f"Added short-term memory for user {user_id}: {len(self.short_term_memory[user_id])} messages")
    
    async def add_long_term_memory(self, user_id: str, memory_type: MemoryType, 
                                 content: str, importance: float = 0.5,
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add to long-term persistent memory"""
        from uuid import uuid4
        memory_id = str(uuid4())
        
        memory_entry = MemoryEntry(
            id=memory_id,
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
            timestamp=datetime.now().astimezone(),
            importance=importance
        )
        
        if user_id not in self.long_term_memory:
            self.long_term_memory[user_id] = []
        
        self.long_term_memory[user_id].append(memory_entry)
        logger.info(f"Added long-term memory ({memory_type.value}) for user {user_id}: {content[:50]}...")
        return memory_id
    
    async def get_recent_context(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation context"""
        if user_id not in self.short_term_memory:
            return []
        messages = self.short_term_memory[user_id]
        return messages[-limit:] if messages else []
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text using keyword matching"""
        topics = []
        text_lower = text.lower()
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        return topics
    
    def _extract_action_items(self, text: str) -> List[str]:
        """Extract action items from text using pattern matching"""
        actions = []
        text_lower = text.lower()
        for action, patterns in self.action_patterns.items():
            if any(re.search(pattern, text_lower) for pattern in patterns):
                actions.append(action)
        return actions
    
    def _identify_urgency(self, text: str) -> Tuple[str, str]:
        """Identify urgency level and indicator"""
        text_lower = text.lower()
        urgency_keywords = ['urgent', 'asap', 'emergency', 'immediately', 'now', 'critical', 'important']
        if any(keyword in text_lower for keyword in urgency_keywords):
            return "High", "🔴"
        elif any(keyword in text_lower for keyword in ['soon', 'tomorrow', 'next week', 'whenever you can']):
            return "Medium", "🟡"
        else:
            return "Low", "🟢"
    
    async def create_enhanced_summary(self, user_id: str) -> Dict[str, Any]:
        """Create an enhanced structured summary of the conversation"""
        # Get recent context
        recent_context = await self.get_recent_context(user_id, limit=10)
        if not recent_context:
            return {"summary": "No conversation history available"}
        
        # Get user profile
        profile = await self.get_or_create_user_profile(user_id)
        user_name = profile.get("name", "Unknown User")
        
        # Build conversation transcript
        transcript_lines = []
        all_messages = []
        for ctx in recent_context:
            message = ctx.get('message', '') or ''
            response = ctx.get('response', '') or ''
            transcript_lines.append(f"User: {message}")
            transcript_lines.append(f"Assistant: {response}")
            all_messages.append(message)
            all_messages.append(response)
        
        full_transcript = " ".join(all_messages)
        
        # Extract key elements
        topics = self._extract_topics(full_transcript)
        actions = self._extract_action_items(full_transcript)
        urgency_level, urgency_indicator = self._identify_urgency(full_transcript)
        
        # Identify key points from the conversation
        key_points = []
        for ctx in recent_context[-3:]:  # Last 3 exchanges
            message = ctx.get('message', '')
            if len(message) > 10:  # Only include substantial messages
                # Extract first sentence or first 50 characters
                first_sentence = message.split('.')[0] if '.' in message else message[:50]
                key_points.append(first_sentence + ("..." if len(message) > 50 else ""))
        
        # Build structured summary
        summary_data = {
            "user": {
                "name": user_name,
                "phone": user_id,
                "known_since": profile.get("created_at", "Unknown")
            },
            "conversation": {
                "topic": topics[0] if topics else "General Discussion",
                "topics": topics,
                "exchanges": len(recent_context),
                "urgency_level": urgency_level,
                "urgency_indicator": urgency_indicator
            },
            "key_points": key_points,
            "action_items": actions,
            "status": "Completed" if len(recent_context) > 2 else "Ongoing"
        }
        
        return summary_data
    
    async def generate_structured_summary_text(self, user_id: str) -> str:
        """Generate a human-readable structured summary"""
        summary_data = await self.create_enhanced_summary(user_id)
        
        if "summary" in summary_data:
            return summary_data["summary"]
        
        # Format the structured summary into readable text
        user_info = summary_data["user"]
        conv_info = summary_data["conversation"]
        key_points = summary_data["key_points"]
        actions = summary_data["action_items"]
        
        # Create the structured summary text
        summary_lines = []
        summary_lines.append(f"User: {user_info['name']} ({user_info['phone']})")
        summary_lines.append(f"Topic: {conv_info['topic']}")
        summary_lines.append(f"Key Points:")
        for i, point in enumerate(key_points, 1):
            summary_lines.append(f"  {i}. {point}")
        
        if actions:
            summary_lines.append(f"Actions: {', '.join(actions[:3])}")
        else:
            summary_lines.append("Actions: None identified")
            
        summary_lines.append(f"Status: {summary_data['status']}")
        summary_lines.append(f"Urgency: {conv_info['urgency_indicator']} {conv_info['urgency_level']}")
        
        return "\n".join(summary_lines)