"""
Feedback Collection System for WhatsApp AI Agent
Implements owner feedback collection for EOC summaries and conversation analysis
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of feedback that can be collected"""
    USEFUL = "useful"
    NOT_USEFUL = "not_useful"
    REOPEN = "reopen"
    CORRECT = "correct"
    INCORRECT = "incorrect"
    SUGGESTION = "suggestion"

class FeedbackSource(Enum):
    """Sources of feedback"""
    OWNER_MESSAGE = "owner_message"
    WEB_INTERFACE = "web_interface"
    API_CALL = "api_call"

@dataclass
class FeedbackEntry:
    """Individual feedback entry"""
    id: str
    conversation_id: str
    feedback_type: FeedbackType
    feedback_source: FeedbackSource
    feedback_text: Optional[str] = None
    owner_number: Optional[str] = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

class FeedbackManager:
    """Manager for collecting and storing feedback"""
    
    def __init__(self):
        # In-memory storage (would use database in production)
        self.feedback_storage: Dict[str, FeedbackEntry] = {}
        logger.info("Feedback manager initialized")
    
    def collect_feedback(self, conversation_id: str, feedback_type: FeedbackType, 
                        feedback_source: FeedbackSource, feedback_text: Optional[str] = None,
                        owner_number: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Collect feedback for a conversation
        
        Args:
            conversation_id: ID of the conversation
            feedback_type: Type of feedback
            feedback_source: Source of feedback
            feedback_text: Optional text feedback
            owner_number: Phone number of owner providing feedback
            metadata: Additional metadata
            
        Returns:
            ID of the created feedback entry
        """
        try:
            feedback_id = str(uuid.uuid4())
            
            feedback_entry = FeedbackEntry(
                id=feedback_id,
                conversation_id=conversation_id,
                feedback_type=feedback_type,
                feedback_source=feedback_source,
                feedback_text=feedback_text,
                owner_number=owner_number,
                metadata=metadata or {}
            )
            
            self.feedback_storage[feedback_id] = feedback_entry
            
            logger.info(f"Collected feedback {feedback_id} for conversation {conversation_id}: {feedback_type.value}")
            
            return feedback_id
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
            raise
    
    def get_feedback_for_conversation(self, conversation_id: str) -> List[FeedbackEntry]:
        """
        Get all feedback entries for a conversation
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            List of feedback entries
        """
        feedback_entries = [
            feedback for feedback in self.feedback_storage.values()
            if feedback.conversation_id == conversation_id
        ]
        return feedback_entries
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about collected feedback
        
        Returns:
            Dictionary with feedback statistics
        """
        if not self.feedback_storage:
            return {"total_feedback": 0}
        
        feedback_types = {}
        feedback_sources = {}
        
        for feedback in self.feedback_storage.values():
            # Count feedback types
            feedback_type = feedback.feedback_type.value
            feedback_types[feedback_type] = feedback_types.get(feedback_type, 0) + 1
            
            # Count feedback sources
            feedback_source = feedback.feedback_source.value
            feedback_sources[feedback_source] = feedback_sources.get(feedback_source, 0) + 1
        
        return {
            "total_feedback": len(self.feedback_storage),
            "feedback_types": feedback_types,
            "feedback_sources": feedback_sources
        }
    
    def process_owner_feedback_message(self, owner_number: str, message: str) -> Optional[Dict[str, Any]]:
        """
        Process feedback message from owner
        
        Args:
            owner_number: Phone number of owner
            message: Feedback message
            
        Returns:
            Dictionary with feedback details or None if not a feedback message
        """
        lower_message = message.lower().strip()
        
        # Check for feedback keywords
        feedback_mapping = {
            "useful": FeedbackType.USEFUL,
            "helpful": FeedbackType.USEFUL,
            "good": FeedbackType.USEFUL,
            "not useful": FeedbackType.NOT_USEFUL,
            "bad": FeedbackType.NOT_USEFUL,
            "incorrect": FeedbackType.INCORRECT,
            "wrong": FeedbackType.INCORRECT,
            "reopen": FeedbackType.REOPEN,
            "suggestion": FeedbackType.SUGGESTION,
            "suggest": FeedbackType.SUGGESTION
        }
        
        for keyword, feedback_type in feedback_mapping.items():
            if keyword in lower_message:
                # Extract any additional text after the keyword
                feedback_text = None
                if ":" in lower_message:
                    feedback_text = lower_message.split(":", 1)[1].strip()
                elif len(lower_message) > len(keyword) + 1:
                    feedback_text = lower_message[len(keyword):].strip()
                
                return {
                    "feedback_type": feedback_type,
                    "feedback_text": feedback_text,
                    "feedback_source": FeedbackSource.OWNER_MESSAGE
                }
        
        return None

# Global instance
feedback_manager = FeedbackManager()

def collect_conversation_feedback(conversation_id: str, feedback_type: FeedbackType,
                                 feedback_source: FeedbackSource, feedback_text: Optional[str] = None,
                                 owner_number: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Collect feedback for a conversation
    
    Args:
        conversation_id: ID of the conversation
        feedback_type: Type of feedback
        feedback_source: Source of feedback
        feedback_text: Optional text feedback
        owner_number: Phone number of owner providing feedback
        metadata: Additional metadata
        
    Returns:
        ID of the created feedback entry
    """
    return feedback_manager.collect_feedback(
        conversation_id, feedback_type, feedback_source, feedback_text, owner_number, metadata
    )

def process_owner_message_for_feedback(owner_number: str, message: str) -> Optional[Dict[str, Any]]:
    """
    Process owner message to detect and extract feedback
    
    Args:
        owner_number: Phone number of owner
        message: Message to process
        
    Returns:
        Dictionary with feedback details or None if not a feedback message
    """
    return feedback_manager.process_owner_feedback_message(owner_number, message)