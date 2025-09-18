import logging
from typing import Dict, Any, List, Tuple, Optional
import re
from datetime import datetime
import asyncio
import os

logger = logging.getLogger(__name__)

# Import EOC detector, classifier if available
try:
    from server.services.eoc_detector import detect_eoc_with_embedding, embed_text
    from server.services.eoc_classifier import classify_eoc
    EOC_DETECTOR_AVAILABLE = True
except ImportError:
    EOC_DETECTOR_AVAILABLE = False
    logger.warning("EOC detector not available")

# Import monitoring if available
try:
    from server.services.monitoring import record_eoc_detection, record_eoc_confirmation
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    logger.warning("Monitoring not available for EOC manager")

class EOCManager:
    def __init__(self, db_manager, config=None):
        self.db_manager = db_manager
        self.config = config
        self.eoc_detector_available = EOC_DETECTOR_AVAILABLE
        self.monitoring_available = MONITORING_AVAILABLE
        
    def detect_eoc(self, text: str, recent_context: List[Dict[str, Any]]) -> Tuple[bool, float, str, Optional[str]]:
        """
        Detect if a conversation has ended based on various methods.
        
        Returns:
            Tuple of (is_eoc, confidence, detection_method, example_id)
        """
        lower = text.lower()
        
        # Get configurable thresholds
        min_conversation_length = 2
        if self.config:
            min_conversation_length = getattr(self.config, "MIN_CONVERSATION_LENGTH", 2)
            
        if len(recent_context) < min_conversation_length:
            return False, 0.0, "conversation_too_short", None
        
        # Traditional keyword-based detection
        eoc_keywords = [
            "thank you", "thanks", "thankyou", "bye", "goodbye", "see you", 
            "regards", "that's all", "that's it", "done", "all good", "all set",
            "i'm done", "we're done", "thats all", "thats it", "that is all", 
            "that's everything", "nothing else", "no more questions", "that's fine",
            "sounds good", "perfect", "great", "awesome", "excellent"
        ]
        is_eoc_keyword = any(k in lower for k in eoc_keywords)
        
        # Calculate keyword-based confidence
        eoc_matches = [k for k in eoc_keywords if k in lower]
        eoc_confidence_keyword = len(eoc_matches) / len(eoc_keywords) if eoc_keywords else 0.0
        
        # Enhanced embedding-based detection (if available)
        is_eoc_embedding = False
        eoc_confidence_embedding = 0.0
        eoc_example_id_embedding = None
        
        if self.eoc_detector_available and EOC_DETECTOR_AVAILABLE:
            try:
                from server.services.eoc_detector import detect_eoc_with_embedding
                is_eoc_embedding, eoc_confidence_embedding, eoc_example_id_embedding = detect_eoc_with_embedding(text)
                logger.debug(f"Embedding-based EOC detection: is_eoc={is_eoc_embedding}, confidence={eoc_confidence_embedding:.3f}")
            except Exception as e:
                logger.warning(f"Embedding-based EOC detection failed: {e}")
        
        # Combined classifier-based detection (if available)
        is_eoc_classified = False
        eoc_probability = 0.0
        
        if self.eoc_detector_available and EOC_DETECTOR_AVAILABLE:
            try:
                from server.services.eoc_classifier import classify_eoc
                is_eoc_classified, eoc_probability = classify_eoc(text, eoc_confidence_keyword, eoc_confidence_embedding)
                logger.debug(f"Classifier-based EOC detection: is_eoc={is_eoc_classified}, probability={eoc_probability:.3f}")
            except Exception as e:
                logger.warning(f"Classifier-based EOC detection failed: {e}")
        
        # Determine final EOC decision using classifier if available, otherwise fallback
        # Get configurable threshold
        min_confidence = 0.60
        if self.config:
            min_confidence = getattr(self.config, "EOC_CONFIDENCE_THRESHOLD", 0.60)
        
        # Additional check for trivial conversations
        total_message_length = sum(
            len(str(ctx.get('message', ''))) + len(str(ctx.get('response', ''))) 
            for ctx in recent_context
        )
        avg_message_length = total_message_length / len(recent_context) if recent_context else 0
        
        # If average message length is very short, it's likely a trivial conversation
        if avg_message_length < 10:
            min_confidence = max(min_confidence, 0.75)
        elif avg_message_length < 20:
            min_confidence = max(min_confidence, 0.65)
        
        # Check for common trivial patterns that shouldn't trigger EOC
        trivial_patterns = [
            "what's the time", "whats the time", "current time", "time please", 
            "what time", "the time", "hi", "hello", "hey", "who are you", "who am i"
        ]
        is_trivial = any(pattern in lower for pattern in trivial_patterns)
        
        # If it's a trivial message, require higher confidence
        if is_trivial:
            min_confidence = max(min_confidence, 0.80)
        
        # Additional check for greeting/identification questions
        if len(recent_context) < 3:
            greeting_patterns = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
            identity_patterns = ["who are you", "who am i", "what's your name", "what is your name"]
            
            is_greeting_conversation = any(
                any(greeting in ctx.get('message', '').lower() for greeting in greeting_patterns) or
                any(identity in ctx.get('message', '').lower() for identity in identity_patterns)
                for ctx in recent_context
            )
            
            if is_greeting_conversation:
                min_confidence = 0.80
        
        # More human-like: recognize that positive affirmations can indicate completion
        positive_affirmations = ["sounds good", "perfect", "great", "awesome", "excellent", "yes", "okay", "alright", "got it", "understood"]
        has_positive_affirmation = any(affirmation in lower for affirmation in positive_affirmations)
        
        # More human-like: recognize that context matters more than specific keywords
        conversation_flow_complete = self._assess_conversation_flow(recent_context)
        user_style_indicates_completion = self._assess_user_completion_style(recent_context)
        
        # NEW: Check for repetitive patterns that indicate stuck conversation
        is_repetitive_pattern = self._detect_repetitive_patterns(recent_context)
        
        # More human-like: Consider the natural rhythm of conversation
        # Look for patterns that indicate the conversation has naturally wound down
        conversation_naturally_winding_down = self._assess_natural_conclusion(recent_context)
        
        if self.eoc_detector_available and EOC_DETECTOR_AVAILABLE and is_eoc_classified and eoc_probability >= min_confidence:
            is_eoc = is_eoc_classified
            eoc_confidence = eoc_probability
            eoc_detected_by = "classifier"
            eoc_example_id = eoc_example_id_embedding
        elif is_eoc_embedding and eoc_confidence_embedding >= min_confidence:
            is_eoc = is_eoc_embedding
            eoc_confidence = eoc_confidence_embedding
            eoc_detected_by = "embedding_similarity"
            eoc_example_id = eoc_example_id_embedding
        elif is_eoc_keyword and eoc_confidence_keyword >= 0.15:
            # For keyword-based detection, be more flexible with keyword matches
            keyword_min_confidence = 0.3 if len(recent_context) >= 3 else 0.4
            # If there's a positive affirmation, be more lenient
            if has_positive_affirmation:
                keyword_min_confidence *= 0.8
            # If conversation flow indicates completion, be more lenient
            if conversation_flow_complete:
                keyword_min_confidence *= 0.9
            # If user style indicates completion, be more lenient
            if user_style_indicates_completion:
                keyword_min_confidence *= 0.95
            # If repetitive pattern detected, increase confidence
            if is_repetitive_pattern:
                keyword_min_confidence *= 0.7  # Lower threshold for repetitive patterns
            # If conversation is naturally winding down, be more lenient
            if conversation_naturally_winding_down:
                keyword_min_confidence *= 0.85
            if eoc_confidence_keyword >= keyword_min_confidence:
                is_eoc = is_eoc_keyword
                eoc_confidence = eoc_confidence_keyword
                eoc_detected_by = "keyword_matching"
                eoc_example_id = None
            else:
                is_eoc = False
                eoc_confidence = 0.0
                eoc_detected_by = "no_eoc_detected"
                eoc_example_id = None
        else:
            # More human-like: even without clear keywords, consider other factors
            if conversation_flow_complete and user_style_indicates_completion and len(recent_context) >= 2:
                # Use a more nuanced approach based on multiple factors
                contextual_confidence = 0.5
                if has_positive_affirmation:
                    contextual_confidence += 0.2
                if len(recent_context) >= 4:
                    contextual_confidence += 0.1
                if avg_message_length > 15:
                    contextual_confidence += 0.1
                # If repetitive pattern detected, increase confidence
                if is_repetitive_pattern:
                    contextual_confidence += 0.3
                # If conversation is naturally winding down, increase confidence
                if conversation_naturally_winding_down:
                    contextual_confidence += 0.2
                    
                if contextual_confidence >= min_confidence:
                    is_eoc = True
                    eoc_confidence = contextual_confidence
                    eoc_detected_by = "contextual_analysis"
                    eoc_example_id = None
                else:
                    is_eoc = False
                    eoc_confidence = 0.0
                    eoc_detected_by = "no_eoc_detected"
                    eoc_example_id = None
            # NEW: Special handling for repetitive patterns
            elif is_repetitive_pattern and len(recent_context) >= 3:
                # If we detect a repetitive pattern, consider ending the conversation
                is_eoc = True
                eoc_confidence = 0.7
                eoc_detected_by = "repetitive_pattern"
                eoc_example_id = None
            # NEW: Special handling for natural conclusion
            elif conversation_naturally_winding_down and len(recent_context) >= 3:
                # If conversation is naturally winding down, consider ending it
                is_eoc = True
                eoc_confidence = 0.65
                eoc_detected_by = "natural_conclusion"
                eoc_example_id = None
            else:
                is_eoc = False
                eoc_confidence = 0.0
                eoc_detected_by = "no_eoc_detected"
                eoc_example_id = None
        
        # Record EOC detection metrics
        if self.monitoring_available and MONITORING_AVAILABLE:
            try:
                from server.services.monitoring import record_eoc_detection, record_eoc_confirmation
                record_eoc_detection(eoc_confidence, eoc_detected_by)
                if is_eoc:
                    record_eoc_confirmation()
            except Exception as e:
                logger.warning(f"Failed to record EOC detection metrics: {e}")
        
        logger.info(f"EOC detection result: is_eoc={is_eoc}, confidence={eoc_confidence:.3f}, method={eoc_detected_by}")
        
        return is_eoc, eoc_confidence, eoc_detected_by, eoc_example_id
        
    def _detect_repetitive_patterns(self, recent_context: List[Dict[str, Any]]) -> bool:
        """
        Detect repetitive patterns in conversation that indicate stuck or looping conversation.
        """
        if len(recent_context) < 3:
            return False
            
        # Check for repeated greetings or similar short messages
        messages = [ctx.get('message', '').lower().strip() for ctx in recent_context[-3:]]
        responses = [ctx.get('response', '').lower().strip() for ctx in recent_context[-3:]]
        
        # Check if last 3 messages are all greetings
        greeting_patterns = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        greeting_count = sum(1 for msg in messages if any(greeting in msg for greeting in greeting_patterns))
        
        if greeting_count >= 3:
            return True
            
        # Check for repeated "whenever you are ready" or similar phrases in responses
        repetitive_response_patterns = [
            "whenever you are ready", "whenever you're ready", "whenever you ready",
            "how can i help you", "how can i help", "how can i assist",
            "is there anything else", "anything else i can help",
            "let me know if you need", "feel free to ask", "happy to assist",
            "please let me know", "let me know if there's"
        ]
        
        repetitive_response_count = 0
        for response in responses:
            for pattern in repetitive_response_patterns:
                if pattern in response:
                    repetitive_response_count += 1
                    break
        
        if repetitive_response_count >= 2:
            return True
            
        # Check for repeated short messages with similar content
        if len(messages) >= 3:
            # Check if messages are very short and similar
            short_message_count = sum(1 for msg in messages if len(msg) < 10)
            if short_message_count >= 3:
                # Check if they contain similar keywords
                words_in_messages = [set(msg.split()) for msg in messages]
                if len(words_in_messages) >= 3:
                    # Check overlap between message word sets
                    overlap_1_2 = len(words_in_messages[0] & words_in_messages[1]) if words_in_messages[0] and words_in_messages[1] else 0
                    overlap_2_3 = len(words_in_messages[1] & words_in_messages[2]) if words_in_messages[1] and words_in_messages[2] else 0
                    if overlap_1_2 > 0 and overlap_2_3 > 0:
                        return True
        
        return False
        
    def _assess_conversation_flow(self, recent_context: List[Dict[str, Any]]) -> bool:
        """
        Assess if the conversation flow indicates completion.
        More human-like: look at the natural flow of conversation.
        """
        if len(recent_context) < 2:
            return False
            
        # Look for patterns that indicate natural conversation completion
        last_message = recent_context[-1].get('message', '').lower()
        second_last_message = recent_context[-2].get('response', '').lower() if len(recent_context) > 1 else ''
        
        # Check if the last message is a response to a question or request
        question_indicators = ['?', 'can you', 'could you', 'would you', 'please']
        request_indicators = ['need', 'want', 'require', 'help me']
        
        is_question_response = any(indicator in second_last_message for indicator in question_indicators)
        is_request_response = any(indicator in second_last_message for indicator in request_indicators)
        
        # Check if the response is conclusive
        conclusive_responses = ['yes', 'no', 'sure', 'okay', 'alright', 'got it', 'understood', 'thanks', 'thank you']
        is_conclusive = any(response in last_message for response in conclusive_responses)
        
        # More human-like: recognize that context matters
        return (is_question_response or is_request_response) and is_conclusive
        
    def _assess_user_completion_style(self, recent_context: List[Dict[str, Any]]) -> bool:
        """
        Assess the user's communication style for completion indicators.
        More human-like: adapt to different user styles.
        """
        if len(recent_context) < 2:
            return False
            
        # Look at message length patterns
        message_lengths = [len(str(ctx.get('message', ''))) for ctx in recent_context]
        response_lengths = [len(str(ctx.get('response', ''))) for ctx in recent_context]
        
        # If messages are getting shorter, it might indicate completion
        if len(message_lengths) >= 3:
            recent_shortening = message_lengths[-1] < message_lengths[-2] < message_lengths[-3]
        else:
            recent_shortening = False
            
        # If responses are getting shorter, it might indicate completion
        if len(response_lengths) >= 3:
            response_shortening = response_lengths[-1] < response_lengths[-2] < response_lengths[-3]
        else:
            response_shortening = False
            
        # More human-like: look for natural communication patterns
        return recent_shortening or response_shortening
        
    def _assess_natural_conclusion(self, recent_context: List[Dict[str, Any]]) -> bool:
        """
        Assess if the conversation is naturally winding down to a conclusion.
        More human-like: recognize natural conversation rhythms.
        """
        if len(recent_context) < 3:
            return False
            
        # Look for patterns that indicate natural conclusion
        last_messages = recent_context[-3:]
        
        # Check if the conversation is becoming more conclusive
        conclusive_indicators = [
            "sounds good", "perfect", "great", "awesome", "excellent", 
            "yes", "okay", "alright", "got it", "understood",
            "thanks", "thank you", "appreciate it", "that helps",
            "that's all", "that's it", "nothing else", "no more questions"
        ]
        
        conclusive_count = 0
        for ctx in last_messages:
            message = ctx.get('message', '').lower()
            response = ctx.get('response', '').lower()
            
            # Check if either message or response contains conclusive indicators
            if any(indicator in message for indicator in conclusive_indicators) or \
               any(indicator in response for indicator in conclusive_indicators):
                conclusive_count += 1
        
        # If at least 2 of the last 3 exchanges show conclusive patterns
        return conclusive_count >= 2
        
    async def handle_state_transition(self, sender: str, is_eoc: bool, eoc_confidence: float, 
                                eoc_detected_by: str, eoc_example_id: Optional[str], 
                                contact_key: str, recent_ctx: List[Dict[str, Any]], 
                                response_text: str) -> str:
        """
        Handle conversation state transitions based on EOC detection.
        
        Returns:
            The new conversation state
        """
        # Check if conversation is already in a specific state
        conversation_state = await self._get_conversation_state(contact_key)
        
        # Handle state transitions
        if conversation_state in ["ended", "reopened"]:
            # If conversation was ended and user sends a new message, reopen it
            if not is_eoc:  # Only reopen for non-EOC messages
                await self._update_conversation_state(
                    contact_key, "ended", "reopened", 
                    {"last_activity": datetime.utcnow().isoformat()},
                    {"message_count": 1, "turns_count": 1}
                )
                logger.info(f"Reopened conversation for {contact_key}")
                
                # Record conversation reopened metric
                if self.monitoring_available and MONITORING_AVAILABLE:
                    try:
                        from server.services.monitoring import record_conversation_reopened
                        record_conversation_reopened()
                    except (ImportError, Exception):
                        pass
                        
                return "reopened"
                
        elif is_eoc and conversation_state == "active":
            # Only move to ended state if confidence is high enough
            # Get configurable threshold
            min_transition_confidence = 0.70  # Reduced from 0.75 to 0.70
            if self.config:
                min_transition_confidence = getattr(self.config, "EOC_TRANSITION_THRESHOLD", 0.70)
                
            if eoc_confidence >= min_transition_confidence:
                # Move from active to pending_end state
                await self._update_conversation_state(
                    contact_key, "active", "pending_end",
                    {
                        "eoc_detected_time": datetime.utcnow().isoformat(),
                        "eoc_confidence": eoc_confidence,
                        "eoc_detected_by": eoc_detected_by,
                        "eoc_example_id": eoc_example_id
                    }
                )
                logger.info(f"Detected EOC for {contact_key}, moved to pending_end state with confidence {eoc_confidence}")
                
                # Apply a timeout-based transition from pending_end to ended
                # In a production system, this would be handled by a background task
                # For now, we'll simulate it with a simple delay
                await asyncio.sleep(0.1)  # Simulate processing time
                
                # Move to ended state immediately for simplicity
                await self._update_conversation_state(
                    contact_key, "pending_end", "ended",
                    {
                        "end_time": datetime.utcnow().isoformat(),
                        "eoc_confidence": eoc_confidence,
                        "eoc_detected_by": eoc_detected_by,
                        "eoc_example_id": eoc_example_id
                    }
                )
                logger.info(f"Moved conversation for {contact_key} to ended state")
                
                # Record conversation ended metric
                if self.monitoring_available and MONITORING_AVAILABLE:
                    try:
                        from server.services.monitoring import record_conversation_ended
                        record_conversation_ended()
                    except (ImportError, Exception):
                        pass
                        
                return "ended"
            else:
                logger.info(f"EOC detected for {contact_key} but confidence too low ({eoc_confidence:.2f}), keeping conversation active")
                return "active"
            
        return conversation_state
        
    async def _get_conversation_state(self, contact_key: str) -> str:
        """Get the current conversation state from database"""
        try:
            get_col = getattr(self.db_manager, 'get_collection', None)
            if callable(get_col):
                convs = get_col('conversations')
                if convs is not None:
                    def _find_conv():
                        find_one_method = getattr(convs, 'find_one', None)
                        if callable(find_one_method):
                            return find_one_method({"phone_number": contact_key}, sort=[("last_activity", -1)])
                        return None
                    
                    recent_conv = await asyncio.to_thread(_find_conv)
                    if recent_conv and isinstance(recent_conv, dict) and 'state' in recent_conv:
                        return recent_conv['state']
        except Exception:
            pass
            
        return "active"  # Default state
        
    async def _update_conversation_state(self, contact_key: str, current_state: str, 
                                   new_state: str, set_fields: Dict[str, Any], 
                                   inc_fields: Optional[Dict[str, int]] = None):
        """Update conversation state in database"""
        try:
            get_col = getattr(self.db_manager, 'get_collection', None)
            convs = None
            if callable(get_col):
                convs = get_col('conversations')
            if convs is not None:
                upd = getattr(convs, 'update_one', None)
                if callable(upd):
                    update_doc = {"$set": {"state": new_state, **set_fields}}
                    if inc_fields:
                        update_doc["$inc"] = inc_fields
                        
                    await asyncio.to_thread(
                        upd,
                        {"phone_number": contact_key, "state": current_state},
                        update_doc,
                        upsert=False,
                    )
        except Exception as e:
            logger.debug("Failed to update conversation state: %s", e)