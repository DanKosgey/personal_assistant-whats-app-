import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from server.models.conversation import ConversationState, IntentType, IntentClassification, ConversationContext
from server.database import db_manager

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation state, intent classification, and context awareness"""
    
    # Use a class-level cache that persists across instances
    _conversation_cache = {}
    
    def __init__(self):
        # Use the class-level cache instead of instance-level
        self.conversation_cache = self._conversation_cache
        # Enhanced intent keywords with more comprehensive patterns
        self.intent_keywords = {
            IntentType.GREETING: {
                "keywords": ["hi", "hello", "hey", "greetings"],
                "patterns": [r"^\s*(hi|hello|hey)\s*[.!]?$", r"^\s*(good\s+(morning|afternoon|evening))\s*[.!]?$"]
            },
            IntentType.QUESTION: {
                "keywords": ["what", "how", "when", "where", "why", "which", "can you", "could you", "do you know"],
                "patterns": [r"\b(what|how|when|where|why|which)\b.*\?", r"\b(can you|could you)\b"]
            },
            IntentType.REQUEST: {
                "keywords": ["please", "can you", "would you", "i need", "i want", "help me", "assist me"],
                "patterns": [r"\b(please|can you|would you)\b", r"\b(i need|i want)\b"]
            },
            IntentType.SCHEDULING: {
                "keywords": ["schedule", "meeting", "appointment", "calendar", "book", "reschedule", "cancel", "plan"],
                "patterns": [r"\b(schedule|meeting|appointment|book)\b", r"\b(cancel|reschedule)\s+(meeting|appointment)\b"]
            },
            IntentType.INFORMATION: {
                "keywords": ["tell me", "info", "information", "details", "about", "explain", "describe"],
                "patterns": [r"\b(tell me|info|information|details)\b", r"\b(about|explain|describe)\b.*\b(me|it)\b"]
            },
            IntentType.ACTION: {
                "keywords": ["do this", "task", "plan", "steps", "todo", "action", "execute", "run"],
                "patterns": [r"\b(do this|task|plan|steps|todo|action)\b", r"\b(execute|run)\b"]
            },
            IntentType.ESCALATION: {
                "keywords": ["urgent", "important", "manager", "owner", "escalate", "asap", "emergency", "critical"],
                "patterns": [r"\b(urgent|asap|emergency|critical)\b", r"\b(escalate to|manager|owner)\b"]
            },
            IntentType.TIME_QUERY: {
                "keywords": ["time", "what time", "current time", "clock", "now"],
                "patterns": [r"\b(what time|current time|time is it)\b", r"\b(now|clock)\b"]
            },
            IntentType.IDENTITY: {
                "keywords": ["my name", "who am i", "identity", "profile", "who are you"],
                "patterns": [r"\b(my name|who am i|identity|profile)\b", r"\b(who are you)\b.*\b(me|I)\b"]
            },
            IntentType.CLOSING: {
                "keywords": ["bye", "goodbye", "see you", "thanks", "thank you", "that's all", "later", "ok bye"],
                "patterns": [r"^\s*(bye|goodbye|see you|thanks|later|ok bye)\s*[.!]?$", r"\b(that's all|that is all)\b"]
            }
        }
        
        # Additional contextual keywords for better classification
        self.contextual_modifiers = {
            "urgency_indicators": ["urgent", "asap", "emergency", "critical", "immediately", "now"],
            "politeness_indicators": ["please", "thank you", "thanks", "kindly", "appreciate"],
            "question_indicators": ["?", "can you", "could you", "would you mind", "is it possible"]
        }
        
    async def get_conversation_context(self, user_id: str, message: str) -> ConversationContext:
        """Get or create conversation context for a user"""
        # Try to load from cache first
        if user_id in self.conversation_cache:
            context = self.conversation_cache[user_id]
            # Update context with new message
            context.last_interaction = datetime.now()
            context.interaction_count += 1
            context.recent_messages.append({"message": message, "timestamp": datetime.now()})
            # Keep only last 10 messages
            if len(context.recent_messages) > 10:
                context.recent_messages = context.recent_messages[-10:]
        else:
            # Try to load from database
            context = await self._load_context_from_db(user_id)
            if context:
                # Update existing context with new message
                context.last_interaction = datetime.now()
                context.interaction_count += 1
                context.recent_messages.append({"message": message, "timestamp": datetime.now()})
                # Keep only last 10 messages
                if len(context.recent_messages) > 10:
                    context.recent_messages = context.recent_messages[-10:]
            else:
                # Create new context
                context = ConversationContext(
                    user_id=user_id,
                    current_state=ConversationState.GREETING,
                    intent=IntentType.UNKNOWN,
                    intent_confidence=0.0,
                    last_interaction=datetime.now(),
                    interaction_count=1,
                    engagement_score=0.0,
                    session_start=datetime.now(),
                    recent_messages=[{"message": message, "timestamp": datetime.now()}],
                    user_preferences={}
                )
            
            # Store in cache
            self.conversation_cache[user_id] = context
            
        # Update conversation state based on message and history
        context.current_state = self._determine_conversation_state(context, message)
        
        # Classify intent
        intent_classification = self.classify_intent(message)
        context.intent = intent_classification.intent
        context.intent_confidence = intent_classification.confidence
        
        # Update engagement score
        context.engagement_score = self._calculate_engagement_score(context)
        
        return context
        
    def _determine_conversation_state(self, context: ConversationContext, message: str) -> ConversationState:
        """Determine the current conversation state based on context and message"""
        # If this is the first interaction, it's a greeting
        if context.interaction_count == 1:
            return ConversationState.GREETING
            
        # Check if user is saying goodbye
        lower_message = message.lower()
        if any(closing_word in lower_message for closing_word in self.intent_keywords[IntentType.CLOSING]["keywords"]):
            return ConversationState.CLOSING
            
        # Check for repetitive simple greetings which might indicate a quick conversation
        if context.recent_messages and len(context.recent_messages) > 2:
            recent_messages = context.recent_messages[-3:]  # Last 3 messages
            simple_greetings = [msg for msg in recent_messages 
                              if isinstance(msg, dict) and 'message' in msg 
                              and msg['message'].lower().strip() in ['hi', 'hello', 'hey']]
            if len(simple_greetings) >= 2:
                # If we have multiple simple greetings in a row, keep conversation minimal
                return ConversationState.ACTIVE
            
        # If we've had several interactions and the last few were task-focused, stay task-focused
        if context.interaction_count > 3 and context.current_state == ConversationState.TASK_FOCUSED:
            # Check if this message is still task-related
            intent = self.classify_intent(message)
            if intent.intent in [IntentType.REQUEST, IntentType.ACTION, IntentType.SCHEDULING]:
                return ConversationState.TASK_FOCUSED
                
        # Classify current message intent
        intent = self.classify_intent(message)
        
        # Transition logic - prevent repetitive greetings
        if intent.intent == IntentType.GREETING:
            # Check if we've recently sent a greeting (within last 5 messages)
            recent_greetings = 0
            if context.recent_messages:
                for msg in context.recent_messages[-5:]:  # Check last 5 messages
                    if isinstance(msg, dict) and 'response' in msg:
                        response = msg['response'].lower()
                        if any(greeting in response for greeting in ['good morning', 'good afternoon', 'good evening', 'hello', 'hi']):
                            recent_greetings += 1
            
            # If we've had 2 or more recent greetings, don't go to greeting state
            if recent_greetings >= 2:
                return ConversationState.ACTIVE
            elif context.current_state != ConversationState.GREETING:
                # Don't go back to greeting unless it's been a while
                if datetime.now() - context.last_interaction > timedelta(minutes=30):
                    return ConversationState.GREETING
                else:
                    # For simple greetings in ongoing conversation, use ACTIVE state
                    return ConversationState.ACTIVE
            else:
                # Default to current state if already in greeting
                return context.current_state
        elif intent.intent in [IntentType.REQUEST, IntentType.ACTION, IntentType.SCHEDULING]:
            return ConversationState.TASK_FOCUSED
        elif intent.intent == IntentType.QUESTION:
            return ConversationState.ACTIVE
        elif intent.intent == IntentType.CLOSING:
            return ConversationState.CLOSING
        else:
            return context.current_state
            
    def classify_intent(self, message: str) -> IntentClassification:
        """Classify user intent with weighted scoring and pattern matching"""
        lower_message = message.lower()
        scores = {}
        matched_keywords = {}
        matched_patterns = {}
        reasoning_details = []
        
        # Score each intent type
        for intent_type, intent_data in self.intent_keywords.items():
            keyword_score = 0
            pattern_score = 0
            matched_kw = []
            matched_pat = []
            
            # Check for exact keyword matches with weighting
            for keyword in intent_data["keywords"]:
                if keyword in lower_message:
                    # Weight longer keywords higher
                    weight = min(len(keyword.split()), 3)  # Cap at 3
                    keyword_score += weight
                    matched_kw.append(keyword)
            
            # Check for pattern matches
            for pattern in intent_data["patterns"]:
                import re
                if re.search(pattern, lower_message, re.IGNORECASE):
                    pattern_score += 2  # Patterns are worth more
                    matched_pat.append(pattern)
            
            # Calculate total score
            total_score = keyword_score + pattern_score
            
            scores[intent_type] = total_score
            matched_keywords[intent_type] = matched_kw
            matched_patterns[intent_type] = matched_pat
            
            # Add reasoning details
            if total_score > 0:
                reasoning_details.append(f"{intent_type.value}: {keyword_score} keyword pts, {pattern_score} pattern pts")
        
        # Apply contextual modifiers
        modifier_effects = []
        for modifier_type, modifiers in self.contextual_modifiers.items():
            modifier_count = sum(1 for modifier in modifiers if modifier in lower_message)
            if modifier_count > 0:
                modifier_effects.append(f"{modifier_type}: {modifier_count} found")
        
        # Find the intent with the highest score
        if scores:
            best_intent = max(scores.items(), key=lambda x: x[1])
            # Normalize confidence to 0-1 range with a reasonable maximum
            max_possible_score = 20  # Adjust based on expected max score
            confidence = min(best_intent[1] / max_possible_score, 1.0)
            
            reasoning = f"Scored {best_intent[1]} points for {best_intent[0].value} intent"
            if reasoning_details:
                reasoning += f" ({', '.join(reasoning_details[:3])})"
            if modifier_effects:
                reasoning += f" | Modifiers: {', '.join(modifier_effects)}"
            
            return IntentClassification(
                intent=best_intent[0],
                confidence=confidence,
                keywords=matched_keywords.get(best_intent[0], []),
                patterns_matched=matched_patterns.get(best_intent[0], []),
                reasoning=reasoning
            )
        else:
            return IntentClassification(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                keywords=[],
                patterns_matched=[],
                reasoning="No matching keywords or patterns found"
            )
            
    def _calculate_engagement_score(self, context: ConversationContext) -> float:
        """Calculate user engagement score based on interaction patterns"""
        # Base score on interaction count and recency
        base_score = min(context.interaction_count / 10.0, 1.0)
        
        # Boost for recent interactions
        time_since_last = datetime.now() - context.last_interaction
        recency_boost = max(0, 1 - (time_since_last.total_seconds() / 3600))  # Decrease by hour
        
        # Boost for varied intents (shows engagement)
        if context.recent_messages:
            unique_intents = len(set(msg.get('intent') for msg in context.recent_messages if isinstance(msg, dict) and 'intent' in msg))
            intent_variety_boost = min(unique_intents / 5.0, 1.0)
        else:
            intent_variety_boost = 0
        
        # Boost for longer messages (indicates more thoughtful engagement)
        if context.recent_messages:
            avg_message_length = sum(len(str(msg.get('message', ''))) for msg in context.recent_messages) / len(context.recent_messages)
            length_boost = min(avg_message_length / 100.0, 1.0)  # Normalize by 100 chars
        else:
            length_boost = 0
            
        # Penalty for repetitive messages (indicates disengagement)
        repetitive_penalty = 0
        if context.recent_messages and len(context.recent_messages) > 3:
            # Check for simple repetitive messages like "Hi"
            simple_greetings = [msg for msg in context.recent_messages[-5:] 
                              if isinstance(msg, dict) and 'message' in msg 
                              and msg['message'].lower().strip() in ['hi', 'hello', 'hey']]
            if len(simple_greetings) >= 3:
                repetitive_penalty = 0.3  # Reduce engagement score for repetitive simple messages
        
        # Combine scores with weights
        total_score = (base_score * 0.3) + (recency_boost * 0.2) + (intent_variety_boost * 0.25) + (length_boost * 0.25) - repetitive_penalty
        return max(0, min(total_score, 1.0))  # Ensure score is between 0 and 1
        
    def get_conversation_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get detailed analytics for a conversation"""
        if user_id not in self.conversation_cache:
            return {}
            
        context = self.conversation_cache[user_id]
        
        # Calculate session duration
        session_duration = datetime.now() - context.session_start
        
        # Calculate message frequency
        message_frequency = context.interaction_count / max(session_duration.total_seconds() / 60, 1)  # Messages per minute
        
        # Calculate intent distribution
        intent_distribution = {}
        for msg in context.recent_messages:
            if isinstance(msg, dict) and 'intent' in msg:
                intent = msg['intent']
                intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
        
        # Calculate response time patterns
        response_times = []
        for i in range(1, len(context.recent_messages)):
            if isinstance(context.recent_messages[i], dict) and isinstance(context.recent_messages[i-1], dict):
                current_time = context.recent_messages[i].get('timestamp')
                previous_time = context.recent_messages[i-1].get('timestamp')
                if current_time and previous_time:
                    response_time = (current_time - previous_time).total_seconds()
                    response_times.append(response_time)
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "session_duration": session_duration.total_seconds(),
            "interaction_count": context.interaction_count,
            "engagement_score": context.engagement_score,
            "message_frequency": message_frequency,
            "intent_distribution": intent_distribution,
            "avg_response_time": avg_response_time,
            "conversation_state": context.current_state.value,
            "intent_confidence": context.intent_confidence
        }
        
    def get_user_engagement_level(self, user_id: str) -> str:
        """Get user engagement level as a descriptive string"""
        if user_id not in self.conversation_cache:
            return "Unknown"
            
        context = self.conversation_cache[user_id]
        score = context.engagement_score
        
        if score >= 0.8:
            return "Highly Engaged"
        elif score >= 0.5:
            return "Moderately Engaged"
        elif score >= 0.2:
            return "Lightly Engaged"
        else:
            return "Low Engagement"
            
    def get_conversation_maturity(self, user_id: str) -> str:
        """Get conversation maturity level as a descriptive string"""
        if user_id not in self.conversation_cache:
            return "Unknown"
            
        context = self.conversation_cache[user_id]
        interaction_count = context.interaction_count
        
        if interaction_count >= 10:
            return "Mature (10+ exchanges)"
        elif interaction_count >= 5:
            return "Developing (5-9 exchanges)"
        elif interaction_count >= 2:
            return "Establishing (2-4 exchanges)"
        else:
            return "Initial (0-1 exchanges)"
        
    def cleanup_old_conversations(self):
        """Remove old conversations from cache and database to prevent memory leaks"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        expired_users = [
            user_id for user_id, context in self.conversation_cache.items()
            if context.last_interaction < cutoff_time
        ]
        for user_id in expired_users:
            del self.conversation_cache[user_id]
        logger.info(f"Cleaned up {len(expired_users)} expired conversations from cache")
        
        # Also clean up old conversations from database
        try:
            if db_manager and db_manager.db is not None:
                conversations = db_manager.get_collection('conversations')
                if conversations is not None:
                    cutoff_iso = cutoff_time.isoformat()
                    result = conversations.delete_many({"last_interaction": {"$lt": cutoff_iso}})
                    logger.info(f"Cleaned up {result.deleted_count} expired conversations from database")
        except Exception as e:
            logger.warning(f"Failed to clean up old conversations from database: {e}")
        
    def update_response_in_context(self, user_id: str, response_text: str):
        """Update the context with the AI's response to track conversation flow"""
        if user_id in self.conversation_cache:
            context = self.conversation_cache[user_id]
            if context.recent_messages:
                # Update the last message with the AI's response
                last_message = context.recent_messages[-1]
                if isinstance(last_message, dict) and 'response' not in last_message:
                    last_message['response'] = response_text
                    
            # Save context to database
            self._save_context_to_db(user_id, context)
    
    async def _load_context_from_db(self, user_id: str) -> Optional[ConversationContext]:
        """Load conversation context from database"""
        try:
            if db_manager and db_manager.db is not None:
                conversations = db_manager.get_collection('conversations')
                if conversations is not None:
                    # Find the most recent conversation for this user
                    if hasattr(db_manager, '_using_motor') and db_manager._using_motor:
                        # Motor (async)
                        recent_conv = await conversations.find_one(
                            {"user_id": user_id}, 
                            sort=[("last_interaction", -1)]
                        )
                    else:
                        # PyMongo (sync) - wrap in asyncio.to_thread
                        recent_conv = await asyncio.to_thread(
                            conversations.find_one,
                            {"user_id": user_id}, 
                            sort=[("last_interaction", -1)]
                        )
                    
                    if recent_conv:
                        # Check if conversation is too old (more than 24 hours)
                        last_interaction = recent_conv.get('last_interaction')
                        if last_interaction:
                            last_interaction_dt = datetime.fromisoformat(last_interaction)
                            if datetime.now() - last_interaction_dt < timedelta(hours=24):
                                # Create ConversationContext from stored data
                                return ConversationContext(
                                    user_id=recent_conv['user_id'],
                                    current_state=ConversationState(recent_conv.get('current_state', 'GREETING')),
                                    intent=IntentType(recent_conv.get('intent', 'UNKNOWN')),
                                    intent_confidence=recent_conv.get('intent_confidence', 0.0),
                                    last_interaction=datetime.fromisoformat(recent_conv['last_interaction']),
                                    interaction_count=recent_conv.get('interaction_count', 1),
                                    engagement_score=recent_conv.get('engagement_score', 0.0),
                                    session_start=datetime.fromisoformat(recent_conv.get('session_start', datetime.now().isoformat())),
                                    recent_messages=recent_conv.get('recent_messages', []),
                                    user_preferences=recent_conv.get('user_preferences', {})
                                )
        except Exception as e:
            logger.warning(f"Failed to load conversation context from database for {user_id}: {e}")
        
        return None
    
    def _save_context_to_db(self, user_id: str, context: ConversationContext):
        """Save conversation context to database"""
        try:
            if db_manager and db_manager.db is not None:
                conversations = db_manager.get_collection('conversations')
                if conversations is not None:
                    # Prepare data for storage
                    context_data = {
                        "user_id": context.user_id,
                        "current_state": context.current_state.value,
                        "intent": context.intent.value,
                        "intent_confidence": context.intent_confidence,
                        "last_interaction": context.last_interaction.isoformat(),
                        "interaction_count": context.interaction_count,
                        "engagement_score": context.engagement_score,
                        "session_start": context.session_start.isoformat(),
                        "recent_messages": context.recent_messages,
                        "user_preferences": context.user_preferences,
                        "updated_at": datetime.now().isoformat()
                    }
                    
                    # Update or insert conversation record
                    conversations.update_one(
                        {"user_id": user_id},
                        {"$set": context_data},
                        upsert=True
                    )
                else:
                    logger.warning(f"Failed to get conversations collection for user {user_id}")
            else:
                logger.warning(f"Database manager not available for user {user_id}")
        except Exception as e:
            logger.warning(f"Failed to save conversation context to database for {user_id}: {e}")
