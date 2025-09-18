import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from server.models.conversation import ConversationState, IntentType

logger = logging.getLogger(__name__)

class ContextEnhancer:
    def __init__(self, persona_manager, time_tools=None, conversation_manager=None):
        self.persona_manager = persona_manager
        self.time_tools = time_tools
        self.conversation_manager = conversation_manager
        
    async def enhance_prompt(self, sender: str, text: str, memory_manager, contact_info: Optional[Dict[str, Any]] = None) -> str:
        """Streamlined prompt enhancement focused on efficiency"""
        
        # Get basic context
        conversation_context = await self.conversation_manager.get_conversation_context(sender, text) if self.conversation_manager else None
        current_persona = self.persona_manager.get_current_persona()
        base_persona_prompt = self.persona_manager.get_system_prompt()
        
        # Build core profile context
        profile_context = self._build_profile_context(contact_info, sender)
        
        # Build efficiency rules based on conversation state
        efficiency_rules = self._build_efficiency_rules(text, conversation_context, contact_info)
        
        # Combine contexts
        enhanced_prompt = f"{base_persona_prompt}\n\n{profile_context}\n\n{efficiency_rules}"
        
        # Add memory context if available
        if memory_manager and hasattr(memory_manager, 'advanced_memory'):
            enhanced_prompt = await memory_manager.advanced_memory.build_context_prompt(
                user_id=sender,
                current_message=text,
                base_prompt=enhanced_prompt,
                max_context_length=1500  # Reduced from 3000
            )
        
        return enhanced_prompt
    
    def _build_profile_context(self, contact_info: Optional[Dict[str, Any]], sender: str) -> str:
        """Build minimal profile context"""
        if not contact_info:
            return f"USER: {sender}"
        
        # Special handling for new contacts
        if contact_info.get("is_new_contact"):
            return f"NEW USER: {sender} - This appears to be the first interaction with this user. Approach naturally and get to know them."
        
        if not contact_info.get("name"):
            return f"USER: {sender} - Name unknown"
        
        context = f"USER PROFILE:\n- Name: {contact_info['name']}"
        
        if contact_info.get("relationship") and contact_info["relationship"] != "unknown":
            context += f"\n- Relationship: {contact_info['relationship']}"
            
        return context
    
    def _build_efficiency_rules(self, text: str, conversation_context, contact_info: Optional[Dict[str, Any]] = None) -> str:
        """Build context-aware efficiency rules"""
        lower_text = text.lower().strip()
        
        # Detect message type
        is_simple_greeting = lower_text in ["hi", "hello", "hey", "hi!", "hello!", "hey!"]
        is_short_response = len(text.split()) <= 3
        is_conversation_active = (conversation_context and 
                                conversation_context.interaction_count > 2 and
                                conversation_context.current_state == ConversationState.ACTIVE)
        
        # Check if this is a new contact
        is_new_contact = contact_info and contact_info.get("is_new_contact")
        
        rules = "RESPONSE RULES:\n"
        
        if is_new_contact:
            rules += "- NEW USER DETECTED: This is likely the first interaction with this user\n"
            rules += "- Approach naturally and get to know them in a human-like way\n"
            rules += "- Don't use scripted greetings, be conversational and authentic\n"
            rules += "- If appropriate, ask for their name in a natural way\n"
        elif is_simple_greeting:
            rules += "- Simple greeting detected: Respond with 'Hi [Name]' or 'Hello' ONLY\n"
            rules += "- DO NOT add questions, time greetings, or explanations\n"
        elif is_short_response and is_conversation_active:
            rules += "- Short response in active conversation: Match user's energy level\n"
            rules += "- Keep responses to 1 sentence maximum\n"
        elif is_conversation_active:
            rules += "- Ongoing conversation: NO time-based greetings\n"
            rules += "- Avoid repeating user's name excessively\n"
        else:
            rules += "- New conversation: Personalized greeting with time context is appropriate\n"
        
        # Universal efficiency rules
        rules += "\nEFFICIENCY REQUIREMENTS:\n"
        rules += "- Maximum 1-2 sentences per response\n"
        rules += "- NO meta-commentary or explanations of your process\n"
        rules += "- NO conversation summaries unless requested\n"
        rules += "- NO repetitive phrases like 'Whenever You Are Ready'\n"
        rules += "- Match user's communication style and energy\n"
        
        # Add conversation state guidance if available
        if conversation_context:
            rules += f"\nCONVERSATION STATE: {conversation_context.current_state.value}"
            if conversation_context.current_state == ConversationState.ACTIVE:
                rules += " - Continue natural flow, avoid formal greetings"
            elif conversation_context.current_state == ConversationState.GREETING:
                rules += " - Warm but concise welcome"
            elif conversation_context.current_state == ConversationState.CLOSING:
                rules += " - Brief, natural closure"
                
        return rules
    
    async def build_intent_specific_prompt(self, sender: str, text: str, persona_manager, memory_manager, contact_info: Optional[Dict[str, Any]] = None) -> str:
        """Simplified intent-specific prompting"""
        lower_text = text.lower()
        
        # Only handle specific intents that need special treatment
        if any(k in lower_text for k in ["time", "what time", "current time"]):
            time_info = self.time_tools.get_current_time() if self.time_tools else None
            if time_info:
                return f"Current time query - provide this information: {time_info['formatted']} ({time_info['timezone']})"
        
        # For most cases, use standard enhanced prompt
        return ""
    
    def _get_time_context(self) -> str:
        """Get minimal time context"""
        if not self.time_tools:
            return ""
            
        time_info = self.time_tools.get_current_time()
        return f"Current time: {time_info['formatted']} ({time_info['timezone']})"