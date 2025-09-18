"""
Consent Collection Workflow for GDPR Compliance
Handles automatic consent prompting, collection, and management
"""

import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum

from .profile_service import ProfileService
from ..models.profiles import ConsentType, ConsentMethod, ProfileConsent
from ..clients.whatsapp import EnhancedWhatsAppClient

logger = logging.getLogger(__name__)

class ConsentFlowState(str, Enum):
    """States in the consent collection flow"""
    INITIAL_PROMPT = "initial_prompt"
    CLARIFICATION = "clarification"
    SPECIFIC_CONSENT = "specific_consent"
    CONFIRMATION = "confirmation"
    COMPLETED = "completed"

class ConsentCollectionWorkflow:
    """Manages the consent collection workflow for users"""
    
    def __init__(self, profile_service: ProfileService, whatsapp_client: EnhancedWhatsAppClient):
        self.profile_service = profile_service
        self.whatsapp_client = whatsapp_client
        
        # Track active consent flows
        self.active_flows: Dict[str, Dict[str, Any]] = {}
        
        # Consent messages templates
        self.consent_messages = {
            "initial_prompt": (
                "🔒 **Privacy & Data Storage**\n\n"
                "To provide you with a personalized experience and remember our conversations, "
                "I'd like to store some information about you, such as:\n\n"
                "• Your name and contact preferences\n"
                "• Our conversation history\n"
                "• Your interests and preferences\n\n"
                "This helps me provide better assistance and remember what we've discussed. "
                "Your data is kept secure and private.\n\n"
                "**Do you consent to this data storage?**\n"
                "Reply 'YES' to allow or 'NO' to decline."
            ),
            "clarification": (
                "I understand you may have questions about data storage. Let me clarify:\n\n"
                "✅ **What I store**: Your name, preferences, and conversation summaries\n"
                "✅ **Why**: To provide personalized assistance and remember context\n"
                "✅ **Security**: Your data is encrypted and kept private\n"
                "✅ **Your rights**: You can request deletion or changes anytime\n\n"
                "Would you like to proceed with data storage? Reply 'YES' or 'NO'."
            ),
            "specific_consent_memory": (
                "📝 **Memory Storage Consent**\n\n"
                "I'd like permission to remember our conversations and store:\n"
                "• Chat history and context\n"
                "• Your preferences and settings\n"
                "• Important information you share\n\n"
                "This allows me to provide continuity across our conversations.\n"
                "Reply 'ALLOW' to grant permission or 'DENY' to decline."
            ),
            "specific_consent_analytics": (
                "📊 **Analytics Consent** (Optional)\n\n"
                "May I collect anonymous usage statistics to improve the service?\n"
                "This includes:\n"
                "• Message frequency and response times\n"
                "• Feature usage patterns\n"
                "• General interaction statistics\n\n"
                "No personal content is included in analytics.\n"
                "Reply 'ALLOW' to consent or 'DENY' to decline."
            ),
            "confirmation_granted": (
                "✅ **Thank you!**\n\n"
                "Your consent has been recorded. I can now:\n"
                "• Remember our conversations\n"
                "• Learn your preferences\n"
                "• Provide personalized assistance\n\n"
                "You can withdraw consent anytime by saying 'forget me' or 'delete my data'."
            ),
            "confirmation_denied": (
                "✅ **Understood**\n\n"
                "I won't store your personal information or conversation history. "
                "Each conversation will be treated independently.\n\n"
                "You can change this decision anytime by saying 'remember me' or 'store my data'."
            ),
            "withdrawal_confirmation": (
                "🗑️ **Data Deletion Requested**\n\n"
                "I'll delete all stored information about you, including:\n"
                "• Your profile and preferences\n"
                "• Conversation history\n"
                "• Any stored personal data\n\n"
                "This action cannot be undone. Reply 'CONFIRM DELETE' to proceed or 'CANCEL' to keep your data."
            ),
            "withdrawal_completed": (
                "✅ **Data Deleted**\n\n"
                "All your stored information has been permanently deleted. "
                "Future conversations will start fresh with no memory of previous interactions."
            )
        }
    
    async def check_consent_needed(self, phone: str) -> Optional[str]:
        """Check if user needs consent collection and return appropriate message"""
        try:
            # Skip consent check for owner messages
            if phone == os.getenv('PA_OWNER_NUMBER'):
                logger.info("⏭️ Owner message detected - skipping consent check")
                return None
                
            profile = await self.profile_service.get_or_create_profile(phone, auto_create=False)
            
            # If profile service is unavailable, skip consent flow
            if profile is None:
                logger.warning(f"Profile service unavailable for {phone} - skipping consent check")
                return None
            
            # No profile exists or consent not granted
            if not profile.consent:
                # Check if we're already in a consent flow
                if phone in self.active_flows:
                    return await self._continue_consent_flow(phone, None)
                
                # Start new consent flow
                await self._start_consent_flow(phone)
                return self.consent_messages["initial_prompt"]
            
            # Check if consent is expired (optional - for periodic re-confirmation)
            if profile.consent_date:
                expiry_days = 365  # 1 year consent validity
                consent_age = datetime.now().astimezone() - profile.consent_date
                if consent_age > timedelta(days=expiry_days):
                    await self._start_consent_flow(phone, flow_type="renewal")
                    return (
                        "🔄 **Annual Consent Renewal**\n\n"
                        "It's been a year since you last confirmed data storage consent. "
                        "Would you like to continue allowing me to remember our conversations?\n\n"
                        "Reply 'YES' to renew consent or 'NO' to stop data storage."
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking consent for {phone}: {e}")
            return None
    
    async def handle_consent_response(self, phone: str, message: str) -> Optional[str]:
        """Handle user's response to consent prompts"""
        try:
            message_lower = message.lower().strip()
            
            # Check for consent withdrawal requests
            withdrawal_triggers = [
                'forget me', 'delete my data', 'remove my information',
                'withdraw consent', 'stop storing', 'delete everything'
            ]
            
            if any(trigger in message_lower for trigger in withdrawal_triggers):
                return await self._handle_consent_withdrawal(phone)
            
            # Check for consent re-grant requests
            regrant_triggers = [
                'remember me', 'store my data', 'grant consent',
                'allow storage', 'save my information'
            ]
            
            if any(trigger in message_lower for trigger in regrant_triggers):
                return await self._handle_consent_regrant(phone)
            
            # Handle active consent flow
            if phone in self.active_flows:
                return await self._continue_consent_flow(phone, message)
            
            return None
            
        except Exception as e:
            logger.error(f"Error handling consent response from {phone}: {e}")
            return None
    
    async def _start_consent_flow(self, phone: str, flow_type: str = "initial") -> None:
        """Start a new consent collection flow"""
        self.active_flows[phone] = {
            "state": ConsentFlowState.INITIAL_PROMPT,
            "flow_type": flow_type,
            "started_at": datetime.now().astimezone(),
            "attempts": 0,
            "consents": {}  # Track individual consent types
        }
    
    async def _continue_consent_flow(self, phone: str, message: Optional[str]) -> Optional[str]:
        """Continue an active consent flow based on user response"""
        if phone not in self.active_flows:
            return None
        
        flow = self.active_flows[phone]
        current_state = flow["state"]
        flow["attempts"] += 1
        
        # Timeout check (abandon flow after 10 minutes of inactivity)
        if datetime.now() - flow["started_at"] > timedelta(minutes=10):
            del self.active_flows[phone]
            return "⏰ Consent collection has timed out. Please say 'remember me' if you'd like to grant data storage consent later."
        
        if not message:
            return None
        
        message_lower = message.lower().strip()
        
        # Parse common positive/negative responses
        positive_responses = ['yes', 'y', 'ok', 'okay', 'sure', 'fine', 'agree', 'accept', 'allow', 'grant']
        negative_responses = ['no', 'n', 'nope', 'refuse', 'decline', 'deny', 'disagree']
        
        if current_state == ConsentFlowState.INITIAL_PROMPT:
            if any(word in message_lower for word in positive_responses):
                # User agrees to data storage
                await self._grant_consent(phone, ConsentType.MEMORY_STORAGE, True)
                flow["state"] = ConsentFlowState.SPECIFIC_CONSENT
                return self.consent_messages["specific_consent_analytics"]
                
            elif any(word in message_lower for word in negative_responses):
                # User declines data storage
                await self._grant_consent(phone, ConsentType.MEMORY_STORAGE, False)
                del self.active_flows[phone]
                return self.consent_messages["confirmation_denied"]
                
            elif 'more info' in message_lower or 'tell me more' in message_lower or '?' in message:
                # User wants clarification
                flow["state"] = ConsentFlowState.CLARIFICATION
                return self.consent_messages["clarification"]
            
        elif current_state == ConsentFlowState.CLARIFICATION:
            if any(word in message_lower for word in positive_responses):
                await self._grant_consent(phone, ConsentType.MEMORY_STORAGE, True)
                flow["state"] = ConsentFlowState.SPECIFIC_CONSENT
                return self.consent_messages["specific_consent_analytics"]
                
            elif any(word in message_lower for word in negative_responses):
                await self._grant_consent(phone, ConsentType.MEMORY_STORAGE, False)
                del self.active_flows[phone]
                return self.consent_messages["confirmation_denied"]
                
        elif current_state == ConsentFlowState.SPECIFIC_CONSENT:
            if any(word in message_lower for word in positive_responses):
                # Grant analytics consent
                await self._grant_consent(phone, ConsentType.ANALYTICS, True)
                del self.active_flows[phone]
                return (
                    self.consent_messages["confirmation_granted"] + 
                    "\n\n📊 Analytics consent also granted - thank you for helping improve the service!"
                )
                
            elif any(word in message_lower for word in negative_responses):
                # Deny analytics but memory consent already granted
                await self._grant_consent(phone, ConsentType.ANALYTICS, False)
                del self.active_flows[phone]
                return self.consent_messages["confirmation_granted"]
        
        # Fallback for unclear responses
        if flow["attempts"] >= 3:
            # Too many unclear responses, abandon flow
            del self.active_flows[phone]
            return (
                "I'm having trouble understanding your response. "
                "Please say 'remember me' clearly if you'd like to grant consent later, "
                "or 'no data storage' if you prefer I don't store your information."
            )
        
        # Ask for clarification
        return (
            "I didn't quite understand that. Please reply with:\n"
            "• 'YES' to allow data storage\n"
            "• 'NO' to decline\n"
            "• 'MORE INFO' for details"
        )
    
    async def _grant_consent(self, phone: str, consent_type: ConsentType, granted: bool) -> bool:
        """Grant or deny specific consent type"""
        try:
            response = await self.profile_service.update_consent(
                phone=phone,
                consent_type=consent_type,
                granted=granted,
                method=ConsentMethod.WHATSAPP_MESSAGE,
                actor="consent_workflow"
            )
            
            logger.info(f"Consent {consent_type.value} {'granted' if granted else 'denied'} for {phone}")
            return response.success
            
        except Exception as e:
            logger.error(f"Error granting consent for {phone}: {e}")
            return False
    
    async def _handle_consent_withdrawal(self, phone: str) -> str:
        """Handle user request to withdraw consent and delete data"""
        try:
            # Start withdrawal confirmation flow
            self.active_flows[phone] = {
                "state": "withdrawal_confirmation",
                "started_at": datetime.now().astimezone(),
                "attempts": 0
            }
            
            return self.consent_messages["withdrawal_confirmation"]
            
        except Exception as e:
            logger.error(f"Error handling consent withdrawal for {phone}: {e}")
            return "I encountered an error processing your data deletion request. Please try again later."
    
    async def _handle_consent_regrant(self, phone: str) -> str:
        """Handle user request to re-grant consent"""
        try:
            # Check if user already has consent
            profile = await self.profile_service.get_or_create_profile(phone, auto_create=False)
            if profile and profile.consent:
                return "✅ You've already granted consent for data storage. I'm remembering our conversations!"
            
            # Start new consent flow
            await self._start_consent_flow(phone, flow_type="regrant")
            return (
                "🔄 **Re-granting Data Storage Consent**\n\n" +
                self.consent_messages["initial_prompt"]
            )
            
        except Exception as e:
            logger.error(f"Error handling consent re-grant for {phone}: {e}")
            return "I encountered an error processing your consent request. Please try again later."
    
    async def cleanup_expired_flows(self) -> int:
        """Clean up expired consent flows"""
        expired_count = 0
        current_time = datetime.now().astimezone()
        
        expired_phones = []
        for phone, flow in self.active_flows.items():
            if current_time - flow["started_at"] > timedelta(hours=1):  # 1 hour timeout
                expired_phones.append(phone)
        
        for phone in expired_phones:
            del self.active_flows[phone]
            expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired consent flows")
        
        return expired_count
    
    def get_flow_status(self, phone: str) -> Optional[Dict[str, Any]]:
        """Get current consent flow status for a user"""
        if phone not in self.active_flows:
            return None
        
        flow = self.active_flows[phone]
        return {
            "phone": phone,
            "state": flow["state"],
            "flow_type": flow.get("flow_type", "unknown"),
            "started_at": flow["started_at"].isoformat(),
            "attempts": flow["attempts"],
            "duration_minutes": (datetime.now() - flow["started_at"]).total_seconds() / 60
        }
    
    def get_all_active_flows(self) -> List[Dict[str, Any]]:
        """Get all active consent flows"""
        return [self.get_flow_status(phone) for phone in self.active_flows.keys()]