import logging
import os
from typing import Dict, Optional, Any
from server.services.feedback import process_owner_message_for_feedback, collect_conversation_feedback, FeedbackType, FeedbackSource

logger = logging.getLogger(__name__)

# Import monitoring if available
try:
    from server.services.monitoring import record_feedback
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    logger.warning("Monitoring not available for feedback manager")

class FeedbackManager:
    def __init__(self, whatsapp_client, db_manager):
        self.whatsapp = whatsapp_client
        self.db_manager = db_manager
        self.owner_number = os.getenv('PA_OWNER_NUMBER')
        
    def process_owner_feedback(self, sender: str, text: str) -> Optional[Dict[str, Any]]:
        """Process feedback from the owner number"""
        if not self.owner_number or sender != self.owner_number:
            return None
            
        feedback_result = process_owner_message_for_feedback(sender, text)
        if not feedback_result:
            return None
            
        logger.info(f"Processing feedback from owner: {feedback_result}")
        return feedback_result
        
    def handle_feedback(self, sender: str, feedback_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the feedback collection and recording"""
        # Try to associate feedback with the most recent conversation
        conversation_id = self._get_most_recent_conversation_id(sender)
        
        # Collect the feedback
        feedback_type = feedback_result["feedback_type"]
        feedback_text = feedback_result["feedback_text"]
        feedback_source = feedback_result["feedback_source"]
        
        feedback_id = collect_conversation_feedback(
            conversation_id=conversation_id or "unknown",
            feedback_type=feedback_type,
            feedback_source=feedback_source,
            feedback_text=feedback_text,
            owner_number=sender
        )
        
        # Record feedback metric
        if MONITORING_AVAILABLE:
            record_feedback(feedback_type.value)
            
        return {
            "analysis": {"type": "feedback_received"},
            "generated": {"text": "Thank you for your feedback!", "source": "feedback_system"},
            "send": {"status": "success", "source": "feedback_response"},
            "context_stored": False
        }
        
    def _get_most_recent_conversation_id(self, sender: str) -> Optional[str]:
        """Get the most recent conversation ID for a sender"""
        try:
            get_col = getattr(self.db_manager, 'get_collection', None)
            if callable(get_col):
                convs = get_col('conversations')
                if convs is not None:
                    def _find_recent_conv():
                        return convs.find_one(
                            {"phone_number": sender}, 
                            sort=[("last_activity", -1)]
                        )
                    
                    # This is a simplified version - in practice, you'd need to handle async properly
                    recent_conv = _find_recent_conv()
                    
                    if recent_conv and '_id' in recent_conv:
                        return str(recent_conv['_id'])
        except Exception as e:
            logger.debug("Failed to get recent conversation ID: %s", e)
        return None