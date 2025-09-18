from typing import Dict, Any, List, Optional
from ..ai import AdvancedAIHandler
from ..clients.whatsapp import EnhancedWhatsAppClient, WhatsAppAPIError
import logging
import re
from datetime import datetime
from .personas import (
    PERSONAL_ASSISTANT_PERSONA,
    TASK_ACKNOWLEDGMENTS,
    ERROR_MESSAGES,
    format_response
)
from .memory import MemoryManager
from .advanced_memory import AdvancedMemoryManager, MemoryType
from .super_memory import SuperMemoryManager
from .persistence import get_or_create_contact, update_contact
from ..database import db_manager
from ..cache import cache_manager
from .profile_service import ProfileService
from ..models.profiles import ConsentType, ConsentMethod, UpsertProfileRequest
from .consent_workflow import ConsentCollectionWorkflow
import os
import asyncio
from ..prompts import (
    build_agent_instruction_prompt,
    build_summary_prompt,
    build_scheduling_prompt,
    build_information_prompt,
    build_action_plan_prompt,
    build_escalation_prompt,
)
from ..persona_manager import PersonaManager
from ..models.profiles import ConversationState  # Import conversation state enum
from datetime import datetime, timedelta
import asyncio
import time
from ..config import config  # Import the config
from ..tools.time_tools import LLMTimeTools  # Import time tools

logger = logging.getLogger(__name__)

# Import EOC detector, classifier, feedback system, and monitoring
try:
    from .eoc_detector import detect_eoc_with_embedding, embed_text
    from .eoc_classifier import classify_eoc
    from .feedback import process_owner_message_for_feedback, collect_conversation_feedback, FeedbackType, FeedbackSource
    from .monitoring import record_eoc_detection, record_eoc_confirmation, record_summary_sent, record_feedback, record_conversation_ended, record_conversation_reopened, record_message_processed, record_ai_call
    EOC_DETECTOR_AVAILABLE = True
    FEEDBACK_SYSTEM_AVAILABLE = True
    MONITORING_AVAILABLE = True
except ImportError:
    EOC_DETECTOR_AVAILABLE = False
    FEEDBACK_SYSTEM_AVAILABLE = False
    MONITORING_AVAILABLE = False
    logger.warning("EOC detector, feedback system, or monitoring not available")
    
    # Define fallback functions
    def process_owner_message_for_feedback(*args, **kwargs):
        return None
    
    def collect_conversation_feedback(*args, **kwargs):
        return ""
    
    def record_eoc_detection(*args, **kwargs):
        pass
    
    def record_eoc_confirmation(*args, **kwargs):
        pass
    
    def record_summary_sent(*args, **kwargs):
        pass
    
    def record_feedback(*args, **kwargs):
        pass
    
    def record_conversation_ended(*args, **kwargs):
        pass
    
    def record_conversation_reopened(*args, **kwargs):
        pass
    
    def record_message_processed(*args, **kwargs):
        pass
    
    def record_ai_call(*args, **kwargs):
        pass
    
    def detect_eoc_with_embedding(*args, **kwargs):
        return False, 0.0, None
    
    def classify_eoc(*args, **kwargs):
        return False, 0.0

# In-memory cache for debounce logic
_last_summary_sent = {}  # phone_number -> timestamp
DEBOUNCE_WINDOW = timedelta(minutes=5)  # Don't send another summary for 5 minutes

logger = logging.getLogger(__name__)


class MessageProcessor:
    def __init__(self, ai: AdvancedAIHandler, whatsapp: EnhancedWhatsAppClient):
        self.ai = ai
        self.whatsapp = whatsapp
        self.memory = MemoryManager()  # Keep existing for compatibility
        
        # Get Google API key for embedding generation
        google_api_key = os.getenv("GEMINI_API_KEY")
        
        # Initialize memory system based on configuration
        if os.getenv("ENABLE_SUPER_MEMORY", "false").lower() in ("1", "true", "yes"):
            self.advanced_memory = SuperMemoryManager(google_api_key)  # Use super memory for premium users
        else:
            self.advanced_memory = AdvancedMemoryManager(google_api_key)  # Use standard advanced memory
        self.profile_service = ProfileService()  # New profile management system
        self.consent_workflow = ConsentCollectionWorkflow(self.profile_service, whatsapp)  # Consent workflow
        self.time_tools = LLMTimeTools()  # Time awareness tools
        
        # Initialize persona manager
        personas_dir = os.getenv("PERSONAS_DIR", "server/personas")
        default_personality = os.getenv("USER_PERSONALITY")
        self.persona_manager = PersonaManager(personas_dir=personas_dir, default_prompt=default_personality)
        
        logger.info(f"Initialized MessageProcessor with {len(self.persona_manager.list_personas())} personas, advanced memory system, profile management, and consent workflow")

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format recent context for AI prompt"""
        if not context:
            return ""
        
        context_str = "\nPrevious conversation context:\n"
        for ctx in context[-3:]:  # Last 3 interactions
            context_str += f"User: {ctx['message']}\n"
            context_str += f"Assistant: {ctx['response']}\n"
        return context_str

    async def _update_profile_from_message(self, sender: str, text: str, response: str) -> None:
        """Extract profile information from message and update profile"""
        try:
            # Get or create profile
            profile = await self.profile_service.get_or_create_profile(sender)
            
            # Extract information from text
            extracted_fields = {}
            lower_text = text.lower()
            
            # Extract name - but be very careful and avoid extracting nicknames or descriptive phrases
            # Only extract names when explicitly stated in common name introduction patterns
            name_patterns = [
                r"my name is\s+([A-Za-z]{2,20}\s+[A-Za-z]{2,20})",  # First Last
                r"my name is\s+([A-Za-z]{2,20})",  # First name only
                r"i am\s+([A-Za-z]{2,20}\s+[A-Za-z]{2,20})",  # First Last
                r"i am\s+([A-Za-z]{2,20})",  # First name only
                r"call me\s+([A-Za-z]{2,20})",  # Call me X
                r"i'm\s+([A-Za-z]{2,20}\s+[A-Za-z]{2,20})",  # First Last
                r"i'm\s+([A-Za-z]{2,20})"  # First name only
            ]
            
            # Only extract name if profile doesn't already have one
            if not getattr(profile, 'name', None):
                for pattern in name_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        name = match.group(1).strip().title()
                        # Enhanced validation to ensure the name makes sense
                        # Skip if the "name" contains words that suggest it's not actually a name
                        skip_words = [
                            'broke', 'poor', 'rich', 'busy', 'tired', 'hungry', 'thirsty', 'angry', 'happy', 'sad', 'excited', 
                            'loan', 'debt', 'money', 'cash', 'house', 'home', 'authorities', 'involved', 'problem', 'issue', 
                            'help', 'assistance', 'support', 'service', 'request', 'order', 'purchase', 'buy', 'sell', 
                            'developer', 'engineer', 'manager', 'director', 'doctor', 'lawyer', 'teacher', 'student', 
                            'professor', 'nurse', 'accountant', 'consultant', 'sales', 'marketing', 'customer', 'client',
                            'urgent', 'important', 'asap', 'emergency', 'immediate', 'critical', 'payment', 'price',
                            'cost', 'budget', 'quote', 'invoice', 'bill', 'charge', 'fee', 'amount', 'deposit', 'balance', 
                            'account', 'transaction', 'transfer', 'send', 'receive', 'withdraw', 'credit', 'debit', 'card', 
                            'mortgage', 'insurance', 'policy', 'claim', 'benefit', 'pension', 'retirement', 'investment', 
                            'stock', 'bond', 'fund', 'portfolio', 'meeting', 'appointment', 'schedule', 'calendar', 'event', 
                            'party', 'celebration', 'birthday', 'anniversary', 'wedding', 'graduation', 'conference', 
                            'seminar', 'workshop', 'training', 'course', 'class', 'question', 'answer', 'solution', 
                            'trouble', 'difficulty', 'challenge', 'obstacle', 'barrier', 'hurdle', 'impediment',
                            'hi', 'hello', 'hey', 'good', 'morning', 'afternoon', 'evening'  # Common greetings that are not names
                        ]
                        
                        # Additional check for common non-name patterns
                        non_name_patterns = [
                            r'\d',  # Numbers
                            r'[!@#$%^&*()_+=\[\]{}|;:,.<>?]',  # Special characters
                            r'\b(i|you|he|she|it|we|they|me|him|her|us|them)\b',  # Pronouns
                            r'\b(is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|could|should|may|might|must|can)\b',  # Verbs
                            r'\b(the|a|an|and|or|but|if|then|else|when|where|why|how|what|which|who|whom)\b'  # Articles and conjunctions
                        ]
                        
                        # Check if name contains skip words
                        contains_skip_word = any(skip_word in name.lower() for skip_word in skip_words)
                        
                        # Check if name matches non-name patterns
                        matches_non_name_pattern = any(re.search(pattern, name.lower()) for pattern in non_name_patterns)
                        
                        # Additional validation to ensure name makes sense
                        if (len(name) > 1 and len(name) <= 50 and name.isalpha() and 
                            not contains_skip_word and not matches_non_name_pattern):
                            # Additional check for common greeting words
                            if name.lower() in ['hi', 'hey', 'hello']:
                                return  # Skip greeting words
                            extracted_fields['name'] = name
                        break
            
            # Extract role/occupation
            role_patterns = [
                r"my role is\s+([A-Za-z0-9 ,.'-]{2,64})",
                r"i work as\s+([A-Za-z0-9 ,.'-]{2,64})",
                r"i am a\s+([A-Za-z0-9 ,.'-]{2,64})"
            ]
            
            for pattern in role_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    role = match.group(1).strip().title()
                    if len(role) > 1 and len(role) <= 64:
                        extracted_fields['persona'] = role[:64]
                        break
            
            # Extract timezone
            if 'timezone' in lower_text or 'time zone' in lower_text:
                tz_patterns = [
                    r"(UTC[+-]?\d{1,2})",
                    r"(GMT[+-]?\d{1,2})",
                    r"([A-Z]{3,4})",  # EST, PST, etc.
                    r"(Africa/[A-Za-z_]+)",
                    r"(America/[A-Za-z_]+)",
                    r"(Asia/[A-Za-z_]+)",
                    r"(Europe/[A-Za-z_]+)"
                ]
                
                for pattern in tz_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        extracted_fields['timezone'] = match.group(1)
                        break
            
            # Extract language preference
            lang_keywords = {
                'english': 'en', 'french': 'fr', 'spanish': 'es', 'german': 'de',
                'swahili': 'sw', 'arabic': 'ar', 'mandarin': 'zh', 'hindi': 'hi'
            }
            
            for lang_name, lang_code in lang_keywords.items():
                if lang_name in lower_text and ('speak' in lower_text or 'language' in lower_text):
                    extracted_fields['language'] = lang_code
                    break
            
            # Update profile if we extracted any fields
            if extracted_fields:
                upsert_request = UpsertProfileRequest(
                    phone=sender,
                    fields=extracted_fields,
                    reason=f"Extracted from message: {text[:50]}...",
                    expected_version=None,
                    session_id=None
                )
                
                profile_response = await self.profile_service.upsert_profile(
                    upsert_request, actor="message_processor"
                )
                
                if profile_response.success:
                    logger.info(f"Updated profile for {sender} with fields: {list(extracted_fields.keys())}")
                else:
                    logger.warning(f"Failed to update profile for {sender}: {profile_response.message}")
        except Exception as e:
            logger.error(f"Error updating profile from message: {e}")
    
    async def _check_consent_and_prompt(self, sender: str) -> Optional[str]:
        """Check if user has given consent and prompt if needed using the consent workflow"""
        try:
            return await self.consent_workflow.check_consent_needed(sender)
        except Exception as e:
            logger.error(f"Error checking consent for {sender}: {e}")
            return None
    
    async def _handle_consent_response(self, sender: str, text: str) -> Optional[str]:
        """Handle user's consent response using the consent workflow"""
        try:
            return await self.consent_workflow.handle_consent_response(sender, text)
        except Exception as e:
            logger.error(f"Error handling consent response from {sender}: {e}")
            return None
    
    async def _enhance_prompt(self, sender: str, text: str) -> str:
        """Enhance user prompt with persona, context, profile information, and advanced memory system"""
        # Get current persona information
        current_persona = self.persona_manager.get_current_persona()
        base_persona_prompt = self.persona_manager.get_system_prompt()
        
        # Get profile information
        try:
            profile = await self.profile_service.get_or_create_profile(sender, auto_create=False)
            if profile:
                # Add profile context to base prompt
                profile_context = f"\n\nUSER PROFILE:\n- Name: {profile.name or 'Unknown'}\n- Language: {profile.language}\n- Timezone: {profile.timezone or 'Unknown'}\n- Persona: {profile.persona or 'Unknown'}"
                if profile.description:
                    profile_context += f"\n- Background: {profile.description[:200]}..."
                
                if base_persona_prompt:
                    base_persona_prompt += profile_context
                else:
                    base_persona_prompt = f"You are a helpful AI assistant.{profile_context}"
        except Exception as e:
            logger.warning(f"Failed to get profile for prompt enhancement: {e}")
        
        if base_persona_prompt and current_persona:
            # Use advanced memory system to build comprehensive context
            enhanced_prompt = await self.advanced_memory.build_context_prompt(
                user_id=sender,
                current_message=text,
                base_prompt=base_persona_prompt,
                max_context_length=3000
            )
            
            # Add persona-specific context
            persona_name = current_persona.get('name', 'Assistant')
            persona_department = current_persona.get('department', 'General')
            persona_tone = current_persona.get('tone', 'professional')
            persona_style = current_persona.get('response_style', 'concise')
            
            # Append persona details to the enhanced prompt
            persona_context = f"\n\nPERSONA DETAILS:\n- Role: {persona_name} from {persona_department}\n- Tone: {persona_tone}\n- Style: {persona_style}"
            enhanced_prompt += persona_context
            
            return enhanced_prompt
        else:
            # Fallback: use advanced memory with basic prompt
            base_prompt = f"You are a helpful AI assistant. Respond appropriately to messages from {sender}."
            return await self.advanced_memory.build_context_prompt(
                user_id=sender,
                current_message=text,
                base_prompt=base_prompt
            )
    
    def get_persona_info(self) -> Dict[str, Any]:
        """Get current persona information"""
        return {
            "current_persona": self.persona_manager.get_current_persona(),
            "available_personas": self.persona_manager.list_personas(),
            "system_prompt": self.persona_manager.get_system_prompt(),
            "stats": self.persona_manager.get_persona_stats()
        }
    
    async def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive memory statistics for a user"""
        advanced_stats = await self.advanced_memory.get_memory_stats(user_id)
        legacy_memory = await self.memory.get_user_memory(user_id)
        
        # Get profile information
        profile_stats = {}
        try:
            profile = await self.profile_service.get_or_create_profile(user_id, auto_create=False)
            if profile:
                profile_stats = {
                    "profile_exists": True,
                    "profile_completeness": await self.profile_service.analyze_profile_completeness(profile),
                    "consent_granted": profile.consent,
                    "consent_date": profile.consent_date.isoformat() if profile.consent_date else None,
                    "last_seen": profile.last_seen.isoformat() if profile.last_seen else None,
                    "profile_version": profile.version,
                    "tags_count": len(profile.tags)
                }
            else:
                profile_stats = {"profile_exists": False}
        except Exception as e:
            logger.error(f"Error getting profile stats for {user_id}: {e}")
            profile_stats = {"profile_exists": False, "error": str(e)}
        
        return {
            "advanced_memory": advanced_stats,
            "legacy_memory": {
                "context_size": len(legacy_memory.session_context),
                "preferences_count": len(legacy_memory.preferences),
                "contacts_count": len(legacy_memory.contacts),
                "last_interaction": legacy_memory.last_interaction.isoformat() if legacy_memory.last_interaction else None
            },
            "profile_system": profile_stats,
            "memory_system": "integrated_advanced_with_profiles"
        }
    
    def reload_personas(self):
        """Reload personas from disk"""
        self.persona_manager.reload_personas()
        logger.info("Personas reloaded successfully")
    
    def switch_persona(self, persona_name: str) -> bool:
        """Switch to a different persona"""
        success = self.persona_manager.select_persona(persona_name)
        if success:
            logger.info(f"Switched to persona: {persona_name}")
        else:
            logger.warning(f"Failed to switch to persona: {persona_name}")
        return success
    
    def _get_most_recent_conversation_id(self, sender: str) -> Optional[str]:
        """Get the most recent conversation ID for a sender"""
        try:
            get_col = getattr(db_manager, 'get_collection', None)
            if callable(get_col):
                convs = get_col('conversations')
                # Check if convs has the find_one method and it's callable
                find_one = getattr(convs, 'find_one', None)
                if convs is not None and find_one is not None and callable(find_one):
                    def _find_recent_conv():
                        return find_one(
                            {"phone_number": sender}, 
                            sort=[("last_activity", -1)]
                        )
                    
                    recent_conv = asyncio.run_coroutine_threadsafe(
                        asyncio.to_thread(_find_recent_conv), 
                        asyncio.get_event_loop()
                    ).result()
                    
                    # Check if recent_conv is a dictionary and has '_id' key
                    if isinstance(recent_conv, dict) and '_id' in recent_conv:
                        return str(recent_conv['_id'])
        except Exception as e:
            logger.debug("Failed to get recent conversation ID: %s", e)
        return None

    async def process(self, sender: str, text: str, message_id: Optional[str] = None) -> Dict[str, Any]:
        logger.info("Processing message from %s: %s (ID: %s)", sender, text, message_id or "unknown")
        
        # Initialize send_result to ensure it's always defined
        send_result = {"status": "unknown"}
        
        # Record message processed metric
        if MONITORING_AVAILABLE:
            record_message_processed()
        
        # Check for feedback messages from owner
        owner_number = os.getenv('PA_OWNER_NUMBER')
        if FEEDBACK_SYSTEM_AVAILABLE and owner_number and sender == owner_number:
            feedback_result = process_owner_message_for_feedback(sender, text)
            if feedback_result:
                # This is a feedback message, process it
                logger.info(f"Processing feedback from owner: {feedback_result}")
                
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
        
        # Check for consent response first
        consent_response = await self._handle_consent_response(sender, text)
        if consent_response:
            await self.whatsapp.send_message(sender, consent_response)
            return {
                "analysis": {"type": "consent_response"},
                "generated": {"text": consent_response, "source": "consent_workflow"},
                "send": {"status": "success", "source": "consent_response"},
                "context_stored": True
            }
        
        # Check if consent is needed
        consent_prompt = await self._check_consent_and_prompt(sender)
        if consent_prompt:
            await self.whatsapp.send_message(sender, consent_prompt)
            return {
                "analysis": {"type": "consent_request"},
                "generated": {"text": consent_prompt, "source": "consent_checker"},
                "send": {"status": "success", "source": "consent_request"},
                "context_stored": False
            }
        
        # Check for contextual field inquiries (only if user has consent)
        # Temporarily disabled field inquiry system

        try:
            # First verify AI handler is ready
            if not self.ai.ai_available:
                logger.error("AI handler not available: %s", 
                    f"Next retry at {self.ai.next_retry_time}" if self.ai.next_retry_time else "No retry time set")
                # AUTO-REPLY: Only when AI is completely unavailable
                await self.whatsapp.send_message(sender, ERROR_MESSAGES["technical"])
                raise RuntimeError("AI service unavailable")

            # Analyze and generate a response
            try:
                analysis = await self.ai.analyze(text)
                logger.info("Message analysis: %s", analysis)
                # Record AI call metric
                if MONITORING_AVAILABLE:
                    record_ai_call()
            except Exception as e:
                logger.error("Analysis failed: %s", e)
                analysis = {"error": str(e)}

            # Generate AI response with enhanced prompt
            try:
                # Record AI call metric
                if MONITORING_AVAILABLE:
                    record_ai_call()
                # Basic intent routing to pick prompt variant
                lower_text = text.lower()
                recent_context = await self.memory.get_recent_context(sender)
                
                # Get persona information for dynamic prompts
                current_persona = self.persona_manager.get_current_persona()
                persona_name = current_persona.get('name', 'Assistant') if current_persona else 'Assistant'
                persona_department = current_persona.get('department', 'General') if current_persona else 'General'

                if any(k in lower_text for k in ["schedule", "meeting", "appoint", "calendar", "resched"]):
                    user_memory = await self.memory.get_user_memory(sender)
                    
                    # Handle both UserMemory object and dictionary cases
                    if hasattr(user_memory, 'preferences'):
                        # UserMemory object case
                        user_preferences = user_memory.preferences
                    elif isinstance(user_memory, dict):
                        # Dictionary case (from advanced memory)
                        user_preferences = user_memory.get('preferences', {})
                    else:
                        # Fallback case
                        user_preferences = {}
                    
                    enhanced_prompt = build_scheduling_prompt(
                        sender=sender,
                        text=text,
                        assistant_name=persona_name,
                        company_name=persona_department,
                        user_preferences=user_preferences,
                        recent_context=recent_context,
                    )
                elif any(k in lower_text for k in ["how", "what", "when", "where", "why"]) and len(text.split()) > 3:
                    enhanced_prompt = build_information_prompt(
                        sender=sender,
                        text=text,
                        assistant_name=persona_name,
                        company_name=persona_department,
                        knowledge_hints=None,
                        recent_context=recent_context,
                    )
                elif any(k in lower_text for k in ["plan", "steps", "todo", "task", "action"]):
                    enhanced_prompt = build_action_plan_prompt(
                        sender=sender,
                        text=text,
                        assistant_name=persona_name,
                        company_name=persona_department,
                        recent_context=recent_context,
                    )
                elif any(k in lower_text for k in ["escalate", "urgent", "important", "manager", "owner"]):
                    owner = current_persona.get('owner_name', 'Manager') if current_persona else 'Manager'
                    enhanced_prompt = build_escalation_prompt(
                        sender=sender,
                        text=text,
                        assistant_name=persona_name,
                        owner_name=owner,
                        reason=None,
                        recent_context=recent_context,
                    )
                else:
                    enhanced_prompt = await self._enhance_prompt(sender, text)

                gen = await self.ai.generate(enhanced_prompt)
                logger.info("Generated AI response: %s", gen)
            except Exception as e:
                logger.error("Generation failed: %s", e)
                
                # ENHANCED FALLBACK: Try to provide context-aware response using memory
                try:
                    # Check if this is a context-related question
                    lower_text = text.lower()
                    context_keywords = ["earlier", "before", "previous", "talked about", "discussed", "mentioned", "said"]
                    
                    if any(keyword in lower_text for keyword in context_keywords):
                        # Get recent conversation context from memory
                        recent_context = await self.memory.get_recent_context(sender)
                        
                        if recent_context:
                            # Build a context summary
                            context_summary = "Here's what we discussed recently:\n\n"
                            for i, ctx in enumerate(recent_context[-3:], 1):  # Last 3 conversations
                                context_summary += f"{i}. You: {ctx['message'][:100]}{'...' if len(ctx['message']) > 100 else ''}\n"
                                context_summary += f"   Me: {ctx['response'][:100]}{'...' if len(ctx['response']) > 100 else ''}\n\n"
                            
                            context_summary += "Is there something specific you'd like to continue discussing?"
                            
                            # Send context summary immediately
                            await self.whatsapp.send_message(sender, context_summary)
                            return {
                                "analysis": analysis,
                                "generated": {"text": context_summary, "source": "memory_fallback"},
                                "send": {"status": "success", "source": "memory_context"},
                                "context_stored": False
                            }
                
                except Exception as memory_error:
                    logger.error("Memory fallback also failed: %s", memory_error)
                
                # AUTO-REPLY: Only when AI generation fails and memory fallback fails
                error_msg = ERROR_MESSAGES["technical"]
                await self.whatsapp.send_message(sender, error_msg)
                return {"error": "ai_generation_failed", "analysis": analysis}
            
            if not gen or not gen.get("text"):
                # AUTO-REPLY: Only when AI returns empty response
                error_msg = ERROR_MESSAGES["technical"]
                await self.whatsapp.send_message(sender, error_msg)
                return {"error": "ai_generation_failed", "analysis": analysis}
                
            response_text = gen.get("text", "")
        except Exception as e:
            # AUTO-REPLY: Only when there's a system-level processing error
            error_msg = "I apologize, but there was an error processing your message. Please try again in a few moments."
            logger.error("Error processing message: %s", str(e))
            try:
                await self.whatsapp.send_message(sender, error_msg)
            except Exception as send_error:
                logger.error("Failed to send error message: %s", str(send_error))
            raise

        # Clean provider tags and special characters from response text
        if isinstance(response_text, str):
            # If the AI handler wrapped replies in square brackets with a provider prefix, strip it.
            # Examples handled:
            #  [DEV_SMOKE reply to: Hello]
            #  [Gemini reply to: Hello]
            import re

            # Strip any provider wrapper but keep the AI-generated response
            m = re.match(r"^\[(?:[A-Za-z0-9_\- ]+?) reply to: .*?\](.*?)$", response_text)
            if m and m.group(1):  # If we matched and have content after the wrapper
                response_text = m.group(1).strip()
            elif not response_text:  # Fallback if empty
                response_text = "I apologize, but I couldn't generate a response at the moment."
            
            # Remove special characters like <s> and </s> that shouldn't be in the final response
            response_text = re.sub(r'<s>\s*', '', response_text)
            response_text = re.sub(r'\s*</s>', '', response_text)
            response_text = re.sub(r'\[OUT\]', '', response_text).strip()
            
            # Enhanced cleaning for problematic AI responses
            # Remove whitespace and special characters that might cause API errors
            response_text = re.sub(r'^\s*<s>\s*', '', response_text)
            response_text = re.sub(r'\s*</s>\s*$', '', response_text)
            response_text = re.sub(r'\[OUT\]', '', response_text)
            response_text = response_text.strip()
            
            # Check if response is effectively empty after cleaning
            if not response_text or len(response_text.strip()) == 0:
                response_text = "I apologize, but I couldn't generate a response at the moment."
            
            # Additional validation to ensure response has meaningful content
            # If response is too short or contains only special characters, provide fallback
            if len(response_text) < 2 or re.match(r'^[^a-zA-Z0-9]+$', response_text):
                response_text = "I'm here to help! How can I assist you today?"
                
        else:
            response_text = "I apologize, but I couldn't generate a response at the moment."
            
        try:
            # Store message and response in both memory systems
            await self.memory.add_to_context(sender, text, response_text)  # Existing system
            await self.advanced_memory.add_short_term_memory(sender, text, response_text)  # New system
            
            # Update profile with extracted information from message
            await self._update_profile_from_message(sender, text, response_text)
            
            # Extract and store entities using advanced memory with AI
            entities = await self.advanced_memory.extract_and_store_entities_with_ai(sender, text, self.ai)
            if entities:
                logger.info(f"Extracted entities for {sender}: {entities}")

            # Simple entity extraction for legacy compatibility
            phones = re.findall(r"\+?\d[\d\s\-()]{6,}\d", text)
            emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)

            # Name heuristics: look for "my name is <Name>" or capitalized words near greeting
            name = None
            m = re.search(r"my name is\s+([A-Za-z ,.'-]{2,50})", text, flags=re.IGNORECASE)
            if m:
                name = m.group(1).strip().title()

            # Persist contact(s) — prefer phone numbers for dedupe; fallback to sender id
            contact_key = sender
            # If phone-like found, prefer the first one
            if phones:
                contact_key = re.sub(r"[^0-9+]", "", phones[0])

            contact_info = {
                "phone": contact_key,
                "name": name or None,
                "emails": emails,
                "source": "whatsapp",
                "first_contacted": datetime.now().astimezone().isoformat()
            }

            # Update in-memory contacts
            await self.memory.add_contact(sender, contact_info)

            # Try to persist to DB if available
            try:
                contact_doc = await get_or_create_contact(contact_key, db_manager, cache_manager)
                # Merge fields and update
                updates = {}
                if name and not contact_doc.get('name'):
                    updates['name'] = name
                if emails and not contact_doc.get('email'):
                    updates['email'] = emails[0]
                if updates:
                    await update_contact(contact_key, updates, db_manager, cache_manager)
            except Exception as e:
                logger.debug("Contact persistence skipped/failed: %s", e)

            # Update user preferences in both memory systems if explicitly mentioned
            lower = text.lower()
            if "my name is" in lower and name:
                await self.memory.set_preference(sender, "name", name)  # Legacy
                await self.advanced_memory.update_user_preference(sender, "name", name)  # Advanced

            role = None
            for marker in ["my role is", "i am a", "i work as"]:
                if marker in lower:
                    role = lower.split(marker)[-1].strip().split(".\n")[0][:50].title()
                    await self.memory.set_preference(sender, "role", role)
                    await self.advanced_memory.update_user_preference(sender, "role", role)
                    break

            if "timezone" in lower:
                if "utc" in text.upper() or "gmt" in text.upper():
                    await self.memory.set_preference(sender, "timezone", text)
                    await self.advanced_memory.update_user_preference(sender, "timezone", text)

            if "work hours" in lower or "working hours" in lower:
                if ":" in text and "-" in text:
                    await self.memory.set_preference(sender, "work_hours", text)
                    await self.advanced_memory.update_user_preference(sender, "work_hours", text)

            # End-of-conversation detection and state management
            # Traditional keyword-based detection
            eoc_keywords = [
                "thank you", "thanks", "thankyou", "bye", "goodbye", "see you", 
                "regards", "that's all", "that's it", "done", "all good", "all set",
                "i'm done", "we're done", "thats all", "thats it"
            ]
            is_eoc_keyword = any(k in lower for k in eoc_keywords)
            
            # Calculate keyword-based confidence
            eoc_matches = [k for k in eoc_keywords if k in lower]
            eoc_confidence_keyword = len(eoc_matches) / len(eoc_keywords) if eoc_keywords else 0.0
            
            # Enhanced embedding-based detection (if available)
            is_eoc_embedding = False
            eoc_confidence_embedding = 0.0
            eoc_example_id_embedding = None
            
            if EOC_DETECTOR_AVAILABLE:
                try:
                    is_eoc_embedding, eoc_confidence_embedding, eoc_example_id_embedding = detect_eoc_with_embedding(text)
                    logger.debug(f"Embedding-based EOC detection: is_eoc={is_eoc_embedding}, confidence={eoc_confidence_embedding:.3f}")
                except Exception as e:
                    logger.warning(f"Embedding-based EOC detection failed: {e}")
            
            # Combined classifier-based detection (if available)
            is_eoc_classified = False
            eoc_probability = 0.0
            
            if EOC_DETECTOR_AVAILABLE:
                try:
                    is_eoc_classified, eoc_probability = classify_eoc(text, eoc_confidence_keyword, eoc_confidence_embedding)
                    logger.debug(f"Classifier-based EOC detection: is_eoc={is_eoc_classified}, probability={eoc_probability:.3f}")
                except Exception as e:
                    logger.warning(f"Classifier-based EOC detection failed: {e}")
            
            # Determine final EOC decision using classifier if available, otherwise fallback
            if EOC_DETECTOR_AVAILABLE and is_eoc_classified:
                is_eoc = is_eoc_classified
                eoc_confidence = eoc_probability
                eoc_detected_by = "classifier"
                eoc_example_id = eoc_example_id_embedding  # Use embedding example ID if available
            elif is_eoc_embedding and eoc_confidence_embedding > 0.5:
                is_eoc = is_eoc_embedding
                eoc_confidence = eoc_confidence_embedding
                eoc_detected_by = "embedding_similarity"
                eoc_example_id = eoc_example_id_embedding
            else:
                is_eoc = is_eoc_keyword
                eoc_confidence = eoc_confidence_keyword
                eoc_detected_by = "keyword_matching"
                eoc_example_id = None
            
            # Record EOC detection metrics
            if MONITORING_AVAILABLE:
                record_eoc_detection(eoc_confidence, eoc_detected_by)
                if is_eoc:
                    record_eoc_confirmation()
            
            logger.info(f"EOC detection result: is_eoc={is_eoc}, confidence={eoc_confidence:.3f}, method={eoc_detected_by}")
            
            # Check if conversation is already in a specific state
            conversation_state = "active"  # Default state
            try:
                # Try to get current conversation state from DB
                get_col = getattr(db_manager, 'get_collection', None)
                if callable(get_col):
                    convs = get_col('conversations')
                    if convs is not None:
                        def _find_conv():
                            return convs.find_one({"phone_number": contact_key}, sort=[("last_activity", -1)])
                        
                        recent_conv = await asyncio.to_thread(_find_conv)
                        if recent_conv and 'state' in recent_conv:
                            conversation_state = recent_conv['state']
            except Exception:
                pass
            
            # Handle state transitions
            if conversation_state == "ended" or conversation_state == "reopened":
                # If conversation was ended and user sends a new message, reopen it
                if not is_eoc:  # Only reopen for non-EOC messages
                    # Update conversation state to reopened
                    try:
                        get_col = getattr(db_manager, 'get_collection', None)
                        convs = None
                        if callable(get_col):
                            convs = get_col('conversations')
                        if convs is not None:
                            upd = getattr(convs, 'update_one', None)
                            if callable(upd):
                                await asyncio.to_thread(
                                    upd,
                                    {"phone_number": contact_key, "state": "ended"},
                                    {
                                        "$set": {
                                            "state": "reopened", 
                                            "last_activity": datetime.utcnow().isoformat()
                                        },
                                        "$inc": {"message_count": 1, "turns_count": 1}
                                    },
                                    upsert=False,
                                )
                        logger.info(f"Reopened conversation for {contact_key}")
                        # Record conversation reopened metric
                        if MONITORING_AVAILABLE:
                            record_conversation_reopened()
                    except Exception as e:
                        logger.debug("Failed to reopen conversation: %s", e)
            
            elif is_eoc and conversation_state == "active":
                # Move from active to pending_end state
                try:
                    get_col = getattr(db_manager, 'get_collection', None)
                    convs = None
                    if callable(get_col):
                        convs = get_col('conversations')
                    if convs is not None:
                        upd = getattr(convs, 'update_one', None)
                        if callable(upd):
                            await asyncio.to_thread(
                                upd,
                                {"phone_number": contact_key, "state": "active"},
                                {
                                    "$set": {
                                        "state": "pending_end", 
                                        "eoc_detected_time": datetime.utcnow().isoformat(),
                                        "eoc_confidence": eoc_confidence,
                                        "eoc_detected_by": eoc_detected_by,
                                        "eoc_example_id": eoc_example_id
                                    }
                                },
                                upsert=False,
                            )
                    logger.info(f"Detected EOC for {contact_key}, moved to pending_end state with confidence {eoc_confidence}")
                except Exception as e:
                    logger.debug("Failed to update conversation state: %s", e)
                
                # Apply a timeout-based transition from pending_end to ended
                # In a production system, this would be handled by a background task
                # For now, we'll simulate it with a simple delay
                import asyncio
                await asyncio.sleep(0.1)  # Simulate processing time
                
                # Move to ended state immediately for simplicity
                # In a real implementation, this would be a separate process
                try:
                    # Summarize conversation for long-term memory
                    summary = await self.advanced_memory.summarize_conversation(sender)
                    if summary:
                        logger.info(f"Created conversation summary for {sender}: {summary}")
                    
                    get_col = getattr(db_manager, 'get_collection', None)
                    convs = None
                    if callable(get_col):
                        convs = get_col('conversations')
                    if convs is not None:
                        upd = getattr(convs, 'update_one', None)
                        if callable(upd):
                            await asyncio.to_thread(
                                upd,
                                {"phone_number": contact_key, "state": "pending_end"},
                                {
                                    "$set": {
                                        "state": "ended", 
                                        "end_time": datetime.utcnow().isoformat(),
                                        "eoc_confidence": eoc_confidence,
                                        "eoc_detected_by": eoc_detected_by,
                                        "eoc_example_id": eoc_example_id
                                    }
                                },
                                upsert=False,
                            )
                    logger.info(f"Moved conversation for {contact_key} to ended state")
                    # Record conversation ended metric
                    if MONITORING_AVAILABLE:
                        record_conversation_ended()
                except Exception as e:
                    logger.debug("Failed to end conversation: %s", e)
                
                # Optionally escalate to owner number — environment variable or memory preference
                owner_number = os.getenv('PA_OWNER_NUMBER') or (await self.memory.get_preference(sender, 'owner_number', None))
                if owner_number:
                    # Check if this conversation is important enough to notify the owner
                    is_important = await self._is_conversation_important(
                        sender, eoc_confidence, recent_ctx, response_text
                    )
                    
                    if is_important:
                        logger.info(f"Conversation with {contact_key} deemed important enough to notify owner (confidence: {eoc_confidence:.2f})")
                        # Prepare conversation summary and suggested agenda/actions
                        # Build a brief human prompt for summarization
                        try:
                            # Create a compact transcript
                            transcript = "\n".join([
                                f"U: {c['message']}\nA: {c['response']}" for c in recent_ctx[-10:]
                            ]) if recent_ctx else text

                            # Use hierarchical summarization for long conversations
                            if len(transcript) > 2000:  # If transcript is long, use hierarchical summarization
                                summary_text = await self.advanced_memory.hierarchical_summarize_conversation(sender)
                            else:
                                # Use regular summarization for shorter conversations
                                summary = await self.advanced_memory.summarize_conversation(sender)
                                summary_text = summary.content if hasattr(summary, 'content') else str(summary)

                            if not summary_text:
                                # Fallback summary
                                last_msgs = transcript if isinstance(transcript, str) else text
                                summary_text = (
                                    f"Conversation ended with {contact_info.get('name') or contact_key}. "
                                    f"Last messages: {last_msgs[:200]}. Suggested next steps: follow up to confirm details, propose meeting times, collect missing contact info."
                                )

                            # Get relevant memory snippets for enhanced summary
                            memory_snippets = ""
                            try:
                                relevant_memories = await self.advanced_memory.search_memories(
                                    sender, 
                                    summary_text, 
                                    limit=3
                                )
                                if relevant_memories:
                                    memory_snippets = "\n\nRelevant Context:\n"
                                    for i, memory in enumerate(relevant_memories, 1):
                                        memory_snippets += f"{i}. {memory.content}\n"
                            except Exception as e:
                                logger.debug("Failed to retrieve memory snippets: %s", e)

                            # Get contact profile for enhanced summary
                            contact_name = contact_info.get('name') or 'Unknown'
                            try:
                                profile = await self.profile_service.get_or_create_profile(sender, auto_create=False)
                                if profile and profile.tags:
                                    contact_tags = ", ".join(profile.tags)
                                else:
                                    contact_tags = "None"
                            except Exception:
                                contact_tags = "Unknown"

                            # Create structured summary matching reference format
                            owner_message = self._create_structured_summary(
                                contact_name=contact_name,
                                contact_phone=contact_key,
                                contact_tags=contact_tags,
                                transcript=transcript,
                                summary_text=summary_text,
                                eoc_confidence=eoc_confidence,
                                eoc_detected_by=eoc_detected_by,
                                recent_context=recent_ctx
                            )

                            # Send summary to owner
                            try:
                                await self.whatsapp.send_message(owner_number, owner_message)
                                # Record summary sent metric
                                if MONITORING_AVAILABLE:
                                    record_summary_sent()
                            except Exception as e:
                                logger.debug("Failed to notify owner %s: %s", owner_number, e)

                            # Persist summary in conversations collection if available
                            try:
                                get_col = getattr(db_manager, 'get_collection', None)
                                convs = None
                                if callable(get_col):
                                    convs = get_col('conversations')
                                if convs is None:
                                    db_attr = getattr(db_manager, 'db', None)
                                    if db_attr is not None:
                                        convs = getattr(db_attr, 'conversations', None) or db_attr['conversations']
                                if convs is not None:
                                    upd = getattr(convs, 'update_one', None)
                                    if callable(upd):
                                        await asyncio.to_thread(
                                            upd,
                                            {"phone_number": contact_key, "state": "ended"},
                                            {
                                                "$set": {
                                                    "summary": summary_text, 
                                                    "state": "ended", 
                                                    "end_time": datetime.utcnow().isoformat(),
                                                    "eoc_confidence": eoc_confidence,
                                                    "eoc_detected_by": eoc_detected_by,
                                                    "eoc_example_id": eoc_example_id
                                                }
                                            },
                                            upsert=False,
                                        )
                            except Exception:
                                pass
                        except Exception as e:
                            logger.debug("Failed to prepare owner summary: %s", e)
                            # Don't send the response here to avoid double texting
                    else:
                        logger.info(f"Conversation with {contact_key} deemed not important enough to notify owner (confidence: {eoc_confidence:.2f})")
                        
                        # Even if not important enough to notify, store embedding if it meets the threshold
                        try:
                            # Calculate importance score for embedding storage
                            importance_score = eoc_confidence * 100  # Convert to 0-100 scale
                            
                            if importance_score >= config.EMBED_SAVE_THRESHOLD:
                                # Create conversation text from recent context
                                conversation_text = "\n".join([
                                    f"User: {ctx.get('message', '')}\nAssistant: {ctx.get('response', '')}" 
                                    for ctx in recent_ctx[-10:]  # Last 10 exchanges
                                ]) if recent_ctx else text
                                
                                # Store conversation embedding
                                embedding_id = await self.advanced_memory.store_conversation_embedding(
                                    user_id=sender,
                                    conversation_text=conversation_text,
                                    importance_score=importance_score,
                                    metadata={
                                        "eoc_confidence": eoc_confidence,
                                        "eoc_detected_by": eoc_detected_by,
                                        "contact_name": contact_info.get('name') or 'Unknown',
                                        "contact_phone": contact_key
                                    }
                                )
                                
                                if embedding_id:
                                    logger.info(f"Stored conversation embedding for {contact_key} with ID {embedding_id}")
                        except Exception as e:
                            logger.debug("Failed to store conversation embedding: %s", e)
                # Send the response to the user (only once)
                send_result = await self.whatsapp.send_message(sender, response_text)
        except WhatsAppAPIError as e:
            logger.error("WhatsApp send failed: %s", e)
            send_result = {"status": "failed", "error": str(e)}
        except Exception as e:
            logger.error("Unexpected error in processing: %s", e)
            # Try to send an error message to the user
            try:
                error_msg = "I apologize, but there was an unexpected error processing your message."
                send_result = await self.whatsapp.send_message(sender, error_msg)
            except Exception as send_error:
                logger.error("Failed to send error message: %s", str(send_error))
                send_result = {"status": "failed", "error": str(send_error)}

        return {
            "analysis": analysis,
            "generated": gen,
            "send": send_result,
            "context_stored": True
        }

    def _get_time_aware_greeting(self) -> str:
        """Generate a time-aware greeting based on current time"""
        time_info = self.time_tools.get_current_time()
        hour = time_info["hour"]
        
        if 5 <= hour < 12:
            return "Good morning"
        elif 12 <= hour < 17:
            return "Good afternoon"
        elif 17 <= hour < 21:
            return "Good evening"
        else:
            return "Hello"
    
    async def _generate_initial_greeting(self, sender: str) -> str:
        """Generate an initial greeting with time awareness and name handling"""
        greeting = self._get_time_aware_greeting()
        
        # Check if we already know the user's name
        name = None
        try:
            profile = await self.profile_service.get_or_create_profile(sender, auto_create=False)
            if profile and profile.name:
                name = profile.name
        except Exception:
            pass
        
        if name:
            return f"{greeting} {name} — I'm Gatekeeper, your assistant. How can I help you today?"
        else:
            return f"{greeting} — I'm Gatekeeper, your assistant. May I have your full name so I can address you properly?"
    
    async def _generate_follow_up_for_name(self, sender: str) -> str:
        """Generate a follow-up message to collect the user's name if not already known"""
        try:
            profile = await self.profile_service.get_or_create_profile(sender, auto_create=False)
            if profile and profile.name:
                # Name already known, no need to ask
                return None
        except Exception:
            pass
        
        # Name not known, ask for it
        return "May I have your full name so I can address you properly?"
    
    async def _is_conversation_important(self, sender: str, eoc_confidence: float, recent_ctx: List[Dict[str, Any]], response_text: str) -> bool:
        """Determine if a conversation is important enough to notify the owner"""
        # Calculate importance score based on multiple factors
        importance_score = 0.0
        
        # 1. EOC confidence (0-30 points)
        importance_score += eoc_confidence * 30
        
        # 2. Conversation length (0-20 points)
        if recent_ctx:
            conv_length_score = min(len(recent_ctx) / 10.0, 1.0) * 20
            importance_score += conv_length_score
        
        # 3. Key action words in response (0-20 points)
        action_keywords = [
            "schedule", "meeting", "appointment", "call", "discuss", "review",
            "approve", "confirm", "verify", "follow up", "follow-up", "urgent",
            "asap", "important", "critical", "priority", "deadline", "sign"
        ]
        response_lower = response_text.lower()
        action_score = sum(2 for keyword in action_keywords if keyword in response_lower)
        importance_score += min(action_score, 20)
        
        # 4. Contact priority (0-15 points) - Check if contact is VIP
        priority_score = 0
        try:
            profile = await self.profile_service.get_or_create_profile(sender, auto_create=False)
            if profile and profile.tags:
                # Check for VIP tags
                vip_tags = ["vip", "important", "priority", "executive", "manager"]
                customer_tags = ["customer", "client", "partner", "vendor"]
                
                if any(tag.lower() in vip_tags for tag in profile.tags):
                    priority_score = 15
                elif any(tag.lower() in customer_tags for tag in profile.tags):
                    priority_score = 10
                # Check for high priority level
                elif getattr(profile, 'priority_level', '').upper() in ['HIGH', 'URGENT']:
                    priority_score = 15
        except Exception as e:
            logger.debug(f"Error checking contact priority: {e}")
        importance_score += priority_score
        
        # 5. Financial keywords (0-15 points)
        financial_keywords = [
            "payment", "price", "cost", "budget", "quote", "invoice", "bill",
            "charge", "fee", "amount", "deposit", "balance", "account"
        ]
        financial_score = sum(1 for keyword in financial_keywords if keyword in response_lower)
        importance_score += min(financial_score, 15)
        
        # Normalize to 0-1 range (0-100 -> 0-1)
        normalized_score = min(importance_score / 100.0, 1.0)
        
        # Consider important if score is above threshold (0.3 = 30%)
        return normalized_score > 0.3

    def _create_structured_summary(self, contact_name: str, contact_phone: str, contact_tags: str,
                                 transcript: str, summary_text: str, eoc_confidence: float, 
                                 eoc_detected_by: str, recent_ctx: List[Dict[str, Any]]) -> str:
        """Create a structured summary matching the reference format"""
        # Extract key information from the summary
        core_intent = summary_text.split('\n')[0] if '\n' in summary_text else summary_text[:100]
        
        # Extract key details from the conversation
        context_text = " ".join([
            f"{ctx.get('message', '')} {ctx.get('response', '')}" 
            for ctx in recent_ctx[-5:]  # Last 5 exchanges
        ]).lower()
        
        # Privacy-conscious filtering - remove sensitive information
        # Remove potential sensitive data patterns
        import re
        # Remove phone numbers
        context_text_filtered = re.sub(r'\+?\d[\d\s\-()]{6,}\d', '[PHONE]', context_text)
        # Remove email addresses
        context_text_filtered = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', context_text_filtered)
        # Remove potential ID numbers
        context_text_filtered = re.sub(r'\b\d{4,}\b', '[ID]', context_text_filtered)
        
        # Extract potential key details (privacy-conscious)
        key_details = []
        if "meeting" in context_text_filtered or "appointment" in context_text_filtered:
            key_details.append("Meeting/appointment requested")
        if "payment" in context_text_filtered or "price" in context_text_filtered or "cost" in context_text_filtered:
            key_details.append("Financial discussion")
        if "urgent" in context_text_filtered or "asap" in context_text_filtered or "emergency" in context_text_filtered:
            key_details.append("Urgent matter")
        if "question" in context_text_filtered or "help" in context_text_filtered:
            key_details.append("Request for assistance")
        if "follow up" in context_text_filtered or "follow-up" in context_text_filtered:
            key_details.append("Follow-up required")
            
        # Extract potential action items
        actionable_items = []
        if "meeting" in context_text_filtered or "appointment" in context_text_filtered:
            actionable_items.append("Confirm meeting time")
        if "payment" in context_text_filtered:
            actionable_items.append("Process payment request")
        if "question" in context_text_filtered:
            actionable_items.append("Provide requested information")
        if "follow up" in context_text_filtered:
            actionable_items.append("Schedule follow-up")
        if "approval" in context_text_filtered or "approve" in context_text_filtered:
            actionable_items.append("Review and approve request")
        if "sign" in context_text_filtered:
            actionable_items.append("Sign document")
            
        # Determine urgency level
        if "urgent" in context_text_filtered or "asap" in context_text_filtered or "emergency" in context_text_filtered or "deadline" in context_text_filtered:
            urgency_level = "High"
        elif "soon" in context_text_filtered or "tomorrow" in context_text_filtered or "next week" in context_text_filtered:
            urgency_level = "Medium"
        else:
            urgency_level = "Low"
            
        # Calculate meaningfulness score for display
        meaningfulness_score = int(eoc_confidence * 100)
        
        # Create a structured summary matching the reference format
        structured_summary = (
            f"WHATSAPP SUMMARY — {contact_name} — {core_intent[:50]}... — Score: {meaningfulness_score}\n\n"
            f"- Contact: {contact_name} ({contact_phone})\n"
            f"- Tags: {contact_tags}\n"
        )
        
        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        structured_summary += f"- Timestamp: {timestamp} EAT\n"
        
        # Add core intent
        structured_summary += f"- Core intent: {core_intent}\n"
        
        # Add key details
        if key_details:
            structured_summary += f"- Key details: {', '.join(key_details[:3])}\n"
        else:
            structured_summary += "- Key details: General conversation\n"
            
        # Add action items
        if actionable_items:
            structured_summary += "- Action items for Owner:\n"
            for i, item in enumerate(actionable_items[:5], 1):
                structured_summary += f"  {i}. {item}\n"
        else:
            structured_summary += "- Action items for Owner: Review conversation\n"
            
        # Add urgency and confidence
        structured_summary += (
            f"- Urgency: {urgency_level}\n"
            f"- Confidence: {meaningfulness_score}%\n"
        )
        
        # Add suggested reply
        structured_summary += "- Suggested reply: Based on the conversation context\n"
        
        # Add transcript excerpt (privacy-conscious)
        transcript_excerpt = transcript[:300] + "..." if len(transcript) > 300 else transcript
        # Apply privacy filtering to excerpt
        transcript_excerpt_filtered = re.sub(r'\+?\d[\d\s\-()]{6,}\d', '[PHONE]', transcript_excerpt)
        transcript_excerpt_filtered = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', transcript_excerpt_filtered)
        transcript_excerpt_filtered = re.sub(r'\b\d{4,}\b', '[ID]', transcript_excerpt_filtered)
        structured_summary += f"- Excerpt: {transcript_excerpt_filtered}\n"
        
        # Add metadata
        structured_summary += f"- Metadata: {{confidence:{eoc_confidence:.2f}, method:{eoc_detected_by}}}\n\n"
        
        # Add full summary (privacy-conscious)
        summary_text_filtered = re.sub(r'\+?\d[\d\s\-()]{6,}\d', '[PHONE]', summary_text)
        summary_text_filtered = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', summary_text_filtered)
        summary_text_filtered = re.sub(r'\b\d{4,}\b', '[ID]', summary_text_filtered)
        structured_summary += f"Full Summary:\n{summary_text_filtered}\n\n"
        
        # Add feedback prompt
        structured_summary += "Was this summary helpful? Reply with: Useful / Not Useful / Reopen"
        
        return structured_summary

