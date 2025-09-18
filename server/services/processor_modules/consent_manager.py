import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class ConsentManager:
    def __init__(self, consent_workflow):
        self.consent_workflow = consent_workflow
        
    async def check_consent_and_prompt(self, sender: str) -> Optional[str]:
        """Check if user has given consent and prompt if needed using the consent workflow"""
        try:
            return await self.consent_workflow.check_consent_needed(sender)
        except Exception as e:
            logger.error(f"Error checking consent for {sender}: {e}")
            return None
    
    async def handle_consent_response(self, sender: str, text: str) -> Optional[Dict[str, Any]]:
        """Handle user's consent response using the consent workflow"""
        try:
            response_text = await self.consent_workflow.handle_consent_response(sender, text)
            if response_text:
                return {
                    "analysis": {"type": "consent_response"},
                    "generated": {"text": response_text, "source": "consent_workflow"},
                    "send": {"status": "success", "source": "consent_response"},
                    "context_stored": True
                }
            return None
        except Exception as e:
            logger.error(f"Error handling consent response from {sender}: {e}")
            return None