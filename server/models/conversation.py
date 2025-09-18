from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime

class ConversationState(Enum):
    """Represents the current state of a conversation"""
    GREETING = "greeting"
    ACTIVE = "active"
    TASK_FOCUSED = "task_focused"
    CLOSING = "closing"
    ENDED = "ended"

class IntentType(Enum):
    """Classification of user intents"""
    GREETING = "greeting"
    QUESTION = "question"
    REQUEST = "request"
    SCHEDULING = "scheduling"
    INFORMATION = "information"
    ACTION = "action"
    ESCALATION = "escalation"
    TIME_QUERY = "time_query"
    IDENTITY = "identity"
    CLOSING = "closing"
    UNKNOWN = "unknown"

@dataclass
class ConversationContext:
    """Enhanced conversation context with state tracking"""
    user_id: str
    current_state: ConversationState
    intent: IntentType
    intent_confidence: float
    last_interaction: datetime
    interaction_count: int
    engagement_score: float
    session_start: datetime
    recent_messages: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    
@dataclass
class UserProfile:
    """Rich user profile with preferences and communication style"""
    name: str
    phone: str
    timezone: str
    language: str
    communication_style: str
    preferences: Dict[str, Any]
    relationship: str
    tags: List[str]
    last_seen: datetime
    engagement_history: List[Dict[str, Any]]
    
@dataclass
class IntentClassification:
    """Result of intent classification with scoring"""
    intent: IntentType
    confidence: float
    keywords: List[str]
    patterns_matched: List[str]
    reasoning: str