from typing import Dict, Any, List, Optional
from ..ai import AdvancedAIHandler
from ..clients.whatsapp import EnhancedWhatsAppClient, WhatsAppAPIError
import logging
import re
from datetime import datetime
from .memory import MemoryManager
from .advanced_memory import AdvancedMemoryManager, MemoryType
from .persistence import get_or_create_contact, update_contact
from ..database import db_manager
from ..cache import cache_manager
from .profile_service import ProfileService
from .consent_workflow import ConsentCollectionWorkflow
import os
import asyncio
from ..persona_manager import PersonaManager
from datetime import datetime, timedelta
import time
from ..config import config  # Import the config
from ..tools.time_tools import LLMTimeTools  # Import time tools

# Import our new modular components
from .processor_modules.feedback_manager import FeedbackManager
from .processor_modules.eoc_manager import EOCManager
from .processor_modules.autonomous_owner_notification_manager import handle_notification_decision
from .processor_modules.memory_manager import MemoryManager as ProcessorMemoryManager
from .processor_modules.profile_manager import ProfileManager
from .processor_modules.consent_manager import ConsentManager
from .processor_modules.context_enhancer import ContextEnhancer

# Import contact assistant
from .processor_modules.contact_assistant import ContactAssistant

# Import existing systems
try:
    from .monitoring import record_message_processed, record_ai_call
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    record_message_processed = lambda: None
    record_ai_call = lambda: None
    logger = logging.getLogger(__name__)
    logger.warning("Monitoring not available")

# Error messages
from .personas import ERROR_MESSAGES

# In-memory cache for debounce logic
_last_summary_sent = {}  # phone_number -> timestamp
DEBOUNCE_WINDOW = timedelta(minutes=5)  # Don't send another summary for 5 minutes

logger = logging.getLogger(__name__)


class MessageProcessor:
    def __init__(self, ai: AdvancedAIHandler, whatsapp: EnhancedWhatsAppClient):
        self.ai = ai
        self.whatsapp = whatsapp
        
        # Initialize core services
        self.memory = MemoryManager()  # Keep existing for compatibility
        self.advanced_memory = AdvancedMemoryManager()  # New advanced memory system
        self.profile_service = ProfileService()  # New profile management system
        self.consent_workflow = ConsentCollectionWorkflow(self.profile_service, whatsapp)  # Consent workflow
        self.time_tools = LLMTimeTools()  # Time awareness tools
        
        # Initialize persona manager
        personas_dir = os.getenv("PERSONAS_DIR", "server/personas")
        default_personality = os.getenv("USER_PERSONALITY")
        self.persona_manager = PersonaManager(personas_dir=personas_dir, default_prompt=default_personality)
        
        # Initialize contact assistant
        self.contact_assistant = ContactAssistant(self.profile_service, self.whatsapp)
        
        # Initialize conversation manager to be shared across components
        from .conversation_manager import ConversationManager
        self.conversation_manager = ConversationManager()
        
        # Initialize modular components with shared conversation manager
        self.feedback_manager = FeedbackManager(whatsapp, db_manager)
        self.eoc_manager = EOCManager(db_manager)
        # Set config for EOC manager
        self.eoc_manager.config = config
        # Add the autonomous notification manager
        self.whatsapp_client = whatsapp
        self.processor_memory_manager = ProcessorMemoryManager(self.memory, self.advanced_memory)
        self.profile_manager = ProfileManager(self.profile_service)
        self.consent_manager = ConsentManager(self.consent_workflow)
        self.context_enhancer = ContextEnhancer(self.persona_manager, self.time_tools, self.conversation_manager)
        # Note: We'll use the existing _enhance_prompt method for now to avoid circular imports
        
        logger.info(f"Initialized MessageProcessor with {len(self.persona_manager.list_personas())} personas, advanced memory system, profile management, and consent workflow")

    async def process(self, sender: str, text: str, message_id: Optional[str] = None) -> Dict[str, Any]:
        logger.info("Processing message from %s: %s (ID: %s)", sender, text, message_id or "unknown")
        
        # Initialize send_result to ensure it's always defined
        send_result = {"status": "unknown"}
        
        # Record message processed metric
        if MONITORING_AVAILABLE:
            try:
                record_message_processed()
            except Exception:
                pass
        
        # Check for feedback messages from owner
        feedback_result = self.feedback_manager.process_owner_feedback(sender, text)
        if feedback_result:
            # This is a feedback message, process it
            return self.feedback_manager.handle_feedback(sender, feedback_result)
        
        # Check for consent response first
        consent_response = await self.consent_manager.handle_consent_response(sender, text)
        if consent_response:
            await self.whatsapp.send_message(sender, consent_response["generated"]["text"])
            return consent_response
        
        # Process contact information first
        should_continue, contact_info, contact_response = await self.contact_assistant.process_contact(
            sender, text, message_id or ""
        )
        
        # If contact assistant wants to send a message (e.g., asking for name), send it and return
        if contact_response:
            await self.whatsapp.send_message(sender, contact_response)
            return {
                "analysis": {"type": "contact_info_request"},
                "generated": {"text": contact_response, "source": "contact_assistant"},
                "send": {"status": "success", "source": "contact_assistant"},
                "context_stored": False
            }
        
        # If contact assistant indicates we shouldn't continue processing, return
        if not should_continue:
            return {
                "analysis": {"type": "contact_info_pending"},
                "generated": {"text": "", "source": "contact_assistant"},
                "send": {"status": "success", "source": "contact_assistant"},
                "context_stored": False
            }
        
        # Check if sender is the owner - if so, use special processing
        owner_number = os.getenv('PA_OWNER_NUMBER')
        if owner_number and sender == owner_number:
            logger.info("👑 Processing message from owner")
            return await self._process_owner_message(sender, text, message_id)
        
        try:
            # First verify AI handler is ready
            if not self.ai.ai_available:
                logger.error("AI handler not available: %s", 
                    f"Next retry at {self.ai.next_retry_time}" if self.ai.next_retry_time else "No retry time set")
                
                # Check if this is due to missing API keys
                if (len(self.ai.gemini_keys) == 0 and 
                    (not self.ai.enable_openrouter_fallback or len(self.ai.openrouter_keys) == 0)):
                    error_msg = "I apologize, but the AI service is not properly configured. No API keys have been added. Please contact the system administrator."
                else:
                    error_msg = ERROR_MESSAGES["technical"]
                
                # AUTO-REPLY: Only when AI is completely unavailable
                await self.whatsapp.send_message(sender, error_msg)
                raise RuntimeError("AI service unavailable")

            # Analyze and generate a response
            try:
                analysis = await self.ai.analyze(text)
                logger.info("Message analysis: %s", analysis)
                # Record AI call metric
                if MONITORING_AVAILABLE:
                    try:
                        record_ai_call()
                    except Exception:
                        pass
            except Exception as e:
                logger.error("Analysis failed: %s", e)
                analysis = {"error": str(e)}

            # Generate AI response with enhanced prompt
            try:
                # Record AI call metric
                if MONITORING_AVAILABLE:
                    try:
                        record_ai_call()
                    except Exception:
                        pass
                    
                # Try to build intent-specific prompt first
                enhanced_prompt = await self.context_enhancer.build_intent_specific_prompt(
                    sender, text, self.persona_manager, self.processor_memory_manager, contact_info
                )
                
                # If no intent-specific prompt, use general enhancement
                if not enhanced_prompt:
                    enhanced_prompt = await self.context_enhancer.enhance_prompt(
                        sender, text, self.processor_memory_manager, contact_info
                    )

                logger.info("PROCESSOR: Generating AI response with prompt length: %d", len(enhanced_prompt) if enhanced_prompt else 0)
                
                # Use reasoning loop for complex tasks
                if self._requires_reasoning(text):
                    gen = await self.ai.generate_with_reasoning(enhanced_prompt)
                else:
                    gen = await self.ai.generate(enhanced_prompt)
                    
                logger.info("PROCESSOR: Generated AI response: %s", gen)
                logger.info("PROCESSOR: AI response details - Provider: %s, Text length: %d", 
                           gen.get("provider", "unknown") if gen else "unknown",
                           len(gen.get("text", "")) if gen else 0)
            except Exception as e:
                logger.error("Generation failed: %s", e)
                return await self._handle_generation_error(sender, text)
            
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

        # Clean provider tags added for debugging/dev (e.g. "[DEV_SMOKE reply to: ...]" or "[Gemini reply to: ...]")
        response_text = self._clean_response_text(response_text)

        try:
            # Store message and response in both memory systems
            await self.processor_memory_manager.store_context(sender, text, response_text, {})
            
            # Process conversation chunk for rolling summaries
            await self.advanced_memory.process_conversation_chunk(
                sender, 
                [{"message": text, "response": response_text, "timestamp": datetime.now().astimezone().isoformat()}]
            )
            
            # Extract and store entities using AI
            if self.ai and hasattr(self.ai, 'generate'):
                await self.advanced_memory.extract_and_store_entities_with_ai(sender, text, self.ai)
            
            # Update the conversation context with the AI's response
            self.conversation_manager.update_response_in_context(sender, response_text)
            
            # Update profile with extracted information from message
            await self.profile_manager.update_from_message(sender, text, response_text)
            
            # Get recent context for EOC detection
            recent_context = await self.processor_memory_manager.get_recent_context(sender)
            
            # End-of-conversation detection
            is_eoc, eoc_confidence, eoc_detected_by, eoc_example_id = self.eoc_manager.detect_eoc(
                text, recent_context
            )
            
            # Extract contact key for conversation state transitions and owner notification
            contact_info, contact_key = self._extract_contact_info(sender, text)
            
            # Handle conversation state transitions
            conversation_state = await self.eoc_manager.handle_state_transition(
                sender, is_eoc, eoc_confidence, eoc_detected_by, eoc_example_id, 
                contact_key, recent_context, response_text
            )
            
            # Notify owner if needed (only when EOC is detected and confidence is high enough)
            if is_eoc and eoc_confidence >= 0.7:  # Use a reasonable threshold
                try:
                    # Use the autonomous notification system
                    owner_number = os.getenv('PA_OWNER_NUMBER')
                    if owner_number:
                        # Prepare features for the autonomous notification system
                        features = {
                            "conversation_length": len(recent_context) if recent_context else 0,
                            "eoc_confidence": eoc_confidence,
                            "response_length": len(response_text) if response_text else 0
                        }
                        
                        # Get owner contact info
                        owner_contact_info = {
                            "whatsapp": owner_number,
                            "channels": ["whatsapp"]
                        }
                        
                        # Extract contact name if available
                        contact_name = "Unknown"
                        if contact_info and isinstance(contact_info, dict):
                            contact_name = contact_info.get("name", "Unknown")
                            
                        # Calculate a more accurate importance score
                        importance_score = self._calculate_importance_score(
                            recent_context, response_text, contact_info
                        )
                        
                        # Call the autonomous notification system
                        notification_result = handle_notification_decision(
                            self.whatsapp_client,
                            owner_number,  # owner_id
                            contact_key,   # conversation_id
                            response_text, # summary_text
                            eoc_confidence,
                            importance_score,
                            features,
                            message_id,    # eoc_event_id
                            "Conversation ended with high confidence",  # model_reasoning
                            owner_contact_info,
                            contact_name,
                            contact_key
                        )
                        logger.info(f"Autonomous notification result: {notification_result}")
                except Exception as e:
                    logger.warning(f"Failed to notify owner via autonomous system: {e}")
                    # No fallback to old system anymore
        
            # Send the response to the user
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
        
    async def _process_owner_message(self, sender: str, text: str, message_id: Optional[str] = None) -> Dict[str, Any]:
        """Process messages from the owner without consent checks."""
        logger.info("👑 Processing message from owner: %s", text)
        
        # Bypass consent checks and process directly
        # First check for feedback
        feedback_result = self.feedback_manager.process_owner_feedback(sender, text)
        if feedback_result:
            # This is a feedback message, process it
            return self.feedback_manager.handle_feedback(sender, feedback_result)

        # Check for contact information
        should_continue, contact_info, contact_response = await self.contact_assistant.process_contact(
            sender, text, message_id or ""
        )
        
        # If contact assistant wants to send a message, send it and return
        if contact_response:
            await self.whatsapp.send_message(sender, contact_response)
            return {
                "analysis": {"type": "contact_info_request"},
                "generated": {"text": contact_response, "source": "contact_assistant"},
                "send": {"status": "success", "source": "contact_assistant"},
                "context_stored": False
            }
        
        # If contact assistant indicates we shouldn't continue processing, return
        if not should_continue:
            return {
                "analysis": {"type": "contact_info_pending"},
                "generated": {"text": "", "source": "contact_assistant"},
                "send": {"status": "success", "source": "contact_assistant"},
                "context_stored": False
            }

        # Analyze and generate response
        try:
            # First verify AI handler is ready
            if not self.ai.ai_available:
                logger.error("PROCESSOR: AI handler not available: %s", 
                    f"Next retry at {self.ai.next_retry_time}" if self.ai.next_retry_time else "No retry time set")
                logger.error("PROCESSOR: AI handler status - Gemini keys: %d, OpenRouter keys: %d, OpenRouter fallback enabled: %s",
                           len(self.ai.gemini_keys), len(self.ai.openrouter_keys), self.ai.enable_openrouter_fallback)
                
                # Check if this is due to missing API keys
                if (len(self.ai.gemini_keys) == 0 and 
                    (not self.ai.enable_openrouter_fallback or len(self.ai.openrouter_keys) == 0)):
                    error_msg = "I apologize, but the AI service is not properly configured. No API keys have been added. Please add API keys using the manage_api_keys.py script."
                else:
                    error_msg = ERROR_MESSAGES["technical"]
                
                # AUTO-REPLY: Only when AI is completely unavailable
                await self.whatsapp.send_message(sender, error_msg)
                raise RuntimeError("AI service unavailable")

            # Analyze and generate response
            analysis = await self.ai.analyze(text)
            logger.info("Message analysis: %s", analysis)
            
            # Build enhanced prompt
            enhanced_prompt = await self.context_enhancer.build_intent_specific_prompt(
                sender, text, self.persona_manager, self.processor_memory_manager, contact_info
            )
            
            # If no intent-specific prompt, use general enhancement
            if not enhanced_prompt:
                enhanced_prompt = await self.context_enhancer.enhance_prompt(
                    sender, text, self.processor_memory_manager, contact_info
                )
            
            logger.info("PROCESSOR: Generating AI response with prompt length: %d", len(enhanced_prompt) if enhanced_prompt else 0)
            gen = await self.ai.generate(enhanced_prompt)
            logger.info("PROCESSOR: Generated AI response: %s", gen)
            logger.info("PROCESSOR: AI response details - Provider: %s, Text length: %d", 
                       gen.get("provider", "unknown") if gen else "unknown",
                       len(gen.get("text", "")) if gen else 0)
            
            response_text = gen.get("text", "")
            
        except Exception as e:
            # Handle generation errors
            logger.error("Generation failed: %s", e)
            return await self._handle_generation_error(sender, text)
            
        # Clean provider tags
        response_text = self._clean_response_text(response_text)
        
        # Store context
        await self.processor_memory_manager.store_context(sender, text, response_text, {})
        
        # Update the conversation context with the AI's response
        self.conversation_manager.update_response_in_context(sender, response_text)
        
        # Extract and store contact info
        contact_info, contact_key = self._extract_contact_info(sender, text)
        await self.processor_memory_manager.add_contact(sender, contact_info)
        
        # Get recent context for EOC detection
        recent_context = await self.processor_memory_manager.get_recent_context(sender)
        
        # End-of-conversation detection
        is_eoc, eoc_confidence, eoc_detected_by, eoc_example_id = self.eoc_manager.detect_eoc(
            text, recent_context
        )
        
        # Handle conversation state transitions
        conversation_state = await self.eoc_manager.handle_state_transition(
            sender, is_eoc, eoc_confidence, eoc_detected_by, eoc_example_id, 
            contact_key, recent_context, response_text
        )
        
        # Notify owner if needed (only when EOC is detected and confidence is high enough)
        if is_eoc and eoc_confidence >= 0.7:  # Use a reasonable threshold
            try:
                # Use the autonomous notification system
                owner_number = os.getenv('PA_OWNER_NUMBER')
                if owner_number:
                    # Prepare features for the autonomous notification system
                    features = {
                        "conversation_length": len(recent_context) if recent_context else 0,
                        "eoc_confidence": eoc_confidence,
                        "response_length": len(response_text) if response_text else 0
                    }
                    
                    # Get owner contact info
                    owner_contact_info = {
                        "whatsapp": owner_number,
                        "channels": ["whatsapp"]
                    }
                    
                    # Extract contact name if available
                    contact_name = "Unknown"
                    if contact_info and isinstance(contact_info, dict):
                        contact_name = contact_info.get("name", "Unknown")
                    
                    # Calculate a more accurate importance score
                    importance_score = self._calculate_importance_score(
                        recent_context, response_text, contact_info
                    )
                    
                    # Call the autonomous notification system
                    notification_result = handle_notification_decision(
                        self.whatsapp_client,
                        owner_number,  # owner_id
                        contact_key,   # conversation_id
                        response_text, # summary_text
                        eoc_confidence,
                        importance_score,
                        features,
                        message_id,    # eoc_event_id
                        "Conversation ended with high confidence",  # model_reasoning
                        owner_contact_info,
                        contact_name,
                        contact_key
                    )
                    logger.info(f"Autonomous notification result: {notification_result}")
            except Exception as e:
                logger.warning(f"Failed to notify owner via autonomous system: {e}")
                # No fallback to old system anymore
        
        # Update contact in database and cache
        try:
            contact_doc = await get_or_create_contact(contact_key, db_manager, cache_manager)
            # Merge fields and update
            updates = {}
            if contact_info.get('name') and not contact_doc.get('name'):
                updates['name'] = contact_info['name']
            if contact_info.get('emails') and not contact_doc.get('email'):
                updates['email'] = contact_info['emails'][0] if contact_info['emails'] else None
            if updates:
                await update_contact(contact_key, updates, db_manager, cache_manager)
        except Exception as e:
            logger.debug("Contact persistence skipped/failed: %s", e)
        
        # Update profile
        await self.profile_manager.update_from_message(sender, text, response_text)
        
        # Get recent context for EOC detection
        recent_context = await self.processor_memory_manager.get_recent_context(sender)
        
        # End-of-conversation detection
        is_eoc, eoc_confidence, eoc_detected_by, eoc_example_id = self.eoc_manager.detect_eoc(
            text, recent_context
        )
        
        # Handle conversation state transitions
        conversation_state = await self.eoc_manager.handle_state_transition(
            sender, is_eoc, eoc_confidence, eoc_detected_by, eoc_example_id, 
            contact_key, recent_context, response_text
        )
        
        # Send the response to the user
        send_result = await self.whatsapp.send_message(sender, response_text)
        
        return {
            "analysis": analysis,
            "generated": gen,
            "send": send_result,
            "context_stored": True
        }
        
    def _extract_contact_info(self, sender: str, text: str) -> tuple:
        """Extract contact information from text with improved name extraction"""
        import re
        from datetime import datetime
        
        # Simple entity extraction for legacy compatibility
        phones = re.findall(r"\+?\d[\d\s\-()]{6,}\d", text)
        emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)

        # Enhanced name heuristics: look for "my name is <Name>" or capitalized words near greeting
        name = None
        
        # More comprehensive name extraction patterns
        name_patterns = [
            r"my name is\s+([A-Za-z][A-Za-z\s\-']{1,49})",  # "my name is X"
            r"i['’]m\s+([A-Za-z][A-Za-z\s\-']{1,49})",      # "i'm X"
            r"i am\s+([A-Za-z][A-Za-z\s\-']{1,49})",        # "i am X"
            r"this is\s+([A-Za-z][A-Za-z\s\-']{1,49})",     # "this is X"
            r"call me\s+([A-Za-z][A-Za-z\s\-']{1,49})",     # "call me X"
        ]
        
        for pattern in name_patterns:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                extracted_name = m.group(1).strip()
                # Limit the extracted name to just the first part (before common conjunctions or punctuation)
                # Split on common words that would indicate the end of a name
                end_words = ['and', 'or', 'but', 'so', 'then', 'because', 'if', 'when', 'where', 
                           'while', 'after', 'before', 'since', 'though', 'although', 'unless', 
                           'until', 'whereas', 'wherever', 'for', 'as', 'to', 'from', 'with',
                           'without', 'within', 'during', 'through', 'across', 'among', 'between']
                
                # Split on these words and take the first part
                parts = re.split(r'\s+(?:' + '|'.join(end_words) + r')\s+', extracted_name, flags=re.IGNORECASE)
                extracted_name = parts[0].strip()
                
                # Also split on punctuation
                punctuation_split = re.split(r'[,.!?;:]', extracted_name)
                extracted_name = punctuation_split[0].strip()
                
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
                    'trouble', 'difficulty', 'challenge', 'obstacle', 'barrier', 'hurdle', 'impediment'
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
                contains_skip_word = any(skip_word in extracted_name.lower() for skip_word in skip_words)
                
                # Check if name matches non-name patterns
                matches_non_name_pattern = any(re.search(pattern, extracted_name.lower()) for pattern in non_name_patterns)
                
                # Additional validation to ensure name makes sense
                if (len(extracted_name) >= 2 and len(extracted_name) <= 50 and 
                    extracted_name.replace(' ', '').replace('-', '').replace("'", '').isalpha() and 
                    not contains_skip_word and not matches_non_name_pattern):
                    # Split into tokens and validate
                    tokens = extracted_name.split()
                    if tokens and all(len(token) >= 1 and len(token) <= 20 and token.replace('-', '').replace("'", '').isalpha() for token in tokens):
                        # Capitalize each token but preserve apostrophes and hyphens
                        capitalized_tokens = []
                        for token in tokens:
                            # Capitalize but preserve special characters
                            if "'" in token or "-" in token:
                                # Handle names like O'Connor or Mary-Jane
                                parts = re.split(r"(['\-])", token)
                                capitalized_parts = [part.capitalize() if part.isalpha() else part for part in parts]
                                capitalized_tokens.append("".join(capitalized_parts))
                            else:
                                capitalized_tokens.append(token.capitalize())
                        name = ' '.join(capitalized_tokens)
                        break

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
        
        return contact_info, contact_key
        
    def _extract_role(self, text: str) -> Optional[str]:
        """Extract role/occupation from text"""
        lower = text.lower()
        role = None
        for marker in ["my role is", "i am a", "i work as"]:
            if marker in lower:
                role = lower.split(marker)[-1].strip().split(".\n")[0][:50].title()
                break
        return role
        
    def _clean_response_text(self, response_text: str) -> str:
        """Clean provider tags and special characters from response text"""
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
            
        return response_text
        
    async def _handle_generation_error(self, sender: str, text: str) -> Dict[str, Any]:
        """Handle AI generation errors with fallback strategies"""
        # ENHANCED FALLBACK: Try to provide context-aware response using memory
        try:
            # Check if this is a context-related question
            lower_text = text.lower()
            context_keywords = ["earlier", "before", "previous", "talked about", "discussed", "mentioned", "said"]
            
            if any(keyword in lower_text for keyword in context_keywords):
                # Get recent conversation context from memory
                recent_context = await self.processor_memory_manager.get_recent_context(sender)
                
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
                        "analysis": {"error": "ai_generation_failed"},
                        "generated": {"text": context_summary, "source": "memory_fallback"},
                        "send": {"status": "success", "source": "memory_context"},
                        "context_stored": False
                    }
        
        except Exception as memory_error:
            logger.error("Memory fallback also failed: %s", memory_error)
        
        # AUTO-REPLY: Only when AI generation fails and memory fallback fails
        error_msg = ERROR_MESSAGES["technical"]
        await self.whatsapp.send_message(sender, error_msg)
        return {"error": "ai_generation_failed"}

    # Keep existing methods for compatibility
    async def _enhance_prompt(self, sender: str, text: str) -> str:
        """Enhance user prompt with persona, context, profile information, and advanced memory system"""
        # Get current persona information
        current_persona = self.persona_manager.get_current_persona()
        
        # Detect user emotion and context for dynamic persona adaptation
        user_emotion = self.persona_manager.detect_user_emotion(text)
        context_type = self.persona_manager.detect_context_type(text)
        
        # Get dynamic persona prompt that adapts to user emotion and context
        base_persona_prompt = self.persona_manager.get_dynamic_persona_prompt(user_emotion, context_type)
        
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
            # Use advanced memory system to build comprehensive context with multi-query retrieval
            enhanced_prompt = await self.advanced_memory.build_context_prompt(
                user_id=sender,
                current_message=text,
                base_prompt=base_persona_prompt,
                max_context_length=3000,
                ai_handler=self.ai  # Pass AI handler for multi-query generation
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
            # Apply dynamic persona adaptation to fallback prompt
            dynamic_prompt = self.persona_manager.get_dynamic_persona_prompt(user_emotion, context_type)
            if dynamic_prompt:
                base_prompt = dynamic_prompt
            
            return await self.advanced_memory.build_context_prompt(
                user_id=sender,
                current_message=text,
                base_prompt=base_prompt,
                ai_handler=self.ai  # Pass AI handler for multi-query generation
            )
    
    def get_persona_info(self) -> Dict[str, Any]:
        """Get current persona information"""
        return {
            "current_persona": self.persona_manager.get_current_persona(),
            "available_personas": self.persona_manager.list_personas(),
            "system_prompt": self.persona_manager.get_system_prompt(),
            "stats": self.persona_manager.get_persona_stats()
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
    
    def _requires_reasoning(self, text: str) -> bool:
        """Determine if a message requires reasoning-based processing"""
        # Keywords that suggest complex reasoning is needed
        reasoning_keywords = [
            "decide", "decision", "choose", "select", "determine", "resolve", 
            "plan", "schedule", "organize", "analyze", "evaluate", "compare",
            "solve", "fix", "troubleshoot", "recommend", "suggest", "advise",
            "calculate", "compute", "figure out", "work out", "sort out"
        ]
        
        # Check if any reasoning keywords are in the text
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in reasoning_keywords)
    
    def _calculate_importance_score(self, recent_context: List[Dict[str, Any]], 
                                  response_text: str, contact_info: Dict[str, Any]) -> float:
        """
        Calculate importance score based on conversation content and context.
        Returns a score between 0.0 and 1.0
        """
        import re
        
        importance_score = 0.0
        
        # Base score from conversation length
        if recent_context:
            conv_length_score = min(len(recent_context) / 10.0, 1.0) * 0.3
            importance_score += conv_length_score
        
        # Score from response content - more comprehensive keyword detection
        if response_text:
            response_lower = response_text.lower()
            
            # Action/Task related keywords (higher weight)
            action_keywords = [
                "schedule", "meeting", "appointment", "call", "discuss", "review",
                "approve", "confirm", "verify", "follow up", "follow-up", "urgent",
                "asap", "important", "critical", "priority", "deadline", "sign",
                "payment", "price", "cost", "budget", "quote", "invoice", "bill",
                "charge", "fee", "amount", "deposit", "balance", "account",
                "order", "purchase", "buy", "sell", "contract", "agreement",
                "deliver", "ship", "send", "receive", "pick up", "drop off"
            ]
            action_score = sum(0.05 for keyword in action_keywords if keyword in response_lower)
            importance_score += min(action_score, 0.3)
            
            # Question indicators (questions might need follow-up)
            question_score = min(response_text.count('?') * 0.1, 0.2)
            importance_score += question_score
            
            # Decision-making keywords
            decision_keywords = [
                "decide", "decision", "choose", "select", "pick", "determine",
                "resolve", "settle", "agree", "disagree", "accept", "reject"
            ]
            decision_score = sum(0.03 for keyword in decision_keywords if keyword in response_lower)
            importance_score += min(decision_score, 0.15)
            
            # Time-sensitive keywords
            time_sensitive_keywords = [
                "today", "tomorrow", "tonight", "now", "immediately", "soon",
                "next week", "next month", "by", "before", "after", "deadline"
            ]
            time_score = sum(0.02 for keyword in time_sensitive_keywords if keyword in response_lower)
            importance_score += min(time_score, 0.1)
            
            # Financial keywords
            financial_keywords = [
                "money", "cash", "payment", "fee", "cost", "price", "budget",
                "expense", "income", "revenue", "profit", "loss", "salary",
                "refund", "deposit", "withdraw", "transfer", "balance"
            ]
            financial_score = sum(0.02 for keyword in financial_keywords if keyword in response_lower)
            importance_score += min(financial_score, 0.1)
            
            # Length score (longer, more detailed responses are more important)
            length_score = min(len(response_text) / 500.0, 0.2)
            importance_score += length_score
        
        # Score from contact info (VIP contacts)
        if contact_info and isinstance(contact_info, dict):
            # Check for VIP tags in contact info
            tags = contact_info.get("tags", [])
            if any(tag.lower() in ["vip", "important", "priority", "executive", "manager", "client", "customer"] for tag in tags):
                importance_score += 0.2
            
            # Check for business-related content in contact name/role
            contact_text = f"{contact_info.get('name', '')} {contact_info.get('role', '')}".lower()
            business_keywords = ["manager", "director", "executive", "boss", "client", "customer", "boss", "ceo", "cto", "cfo"]
            business_score = sum(0.1 for keyword in business_keywords if keyword in contact_text)
            importance_score += min(business_score, 0.2)
        
        # Ensure minimum score for basic conversations
        if importance_score == 0.0 and recent_context:
            importance_score = 0.1  # Minimum score for any conversation
            
        # Normalize to 0-1 range
        return min(importance_score, 1.0)
