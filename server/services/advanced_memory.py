"""
Advanced Memory and Context System for WhatsApp AI Agent
Implements the comprehensive memory architecture with:
- Short-term conversational context
- Long-term persistent memory with vector embeddings
- Cross-session context awareness
- User profile and preference management
- Conversation summarization
"""

import json
import hashlib
import logging
import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import uuid
import os

# Import Google Generative AI for embeddings
try:
    import google.generativeai as genai
    from google.generativeai.types import content_types as glm_content_types
    from google.ai import generativelanguage as glm
    GOOGLE_GENERATIVE_AI_AVAILABLE = True
except ImportError:
    genai = None
    glm_content_types = None
    glm = None
    GOOGLE_GENERATIVE_AI_AVAILABLE = False

# Import prompt functions for summary generation
from ..prompts import build_summary_prompt, build_detailed_summary_prompt

# Import profile service for database integration
from ..services.profile_service import ProfileService

logger = logging.getLogger(__name__)

class MemorySystemError(Exception):
    """Base exception for memory system errors"""
    pass

class MemoryStorageError(MemorySystemError):
    """Exception for memory storage operations"""
    pass

class MemorySearchError(MemorySystemError):
    """Exception for memory search operations"""
    pass

class MemoryValidationError(MemorySystemError):
    """Exception for memory validation errors"""
    pass

class MemoryType(Enum):
    """Types of memories stored in the system"""
    PROFILE = "profile"                    # User profile information
    PREFERENCE = "preference"              # User preferences and settings
    CONVERSATION_SUMMARY = "conversation_summary"  # Summarized conversations
    ENTITY = "entity"                      # Extracted entities (contacts, etc.)
    FACT = "fact"                         # Important facts about the user
    INTERACTION = "interaction"            # Significant interactions
    CONTEXT = "context"                   # Cross-session context

@dataclass
class MemoryEntry:
    """Individual memory entry with metadata"""
    id: str
    user_id: str
    memory_type: MemoryType
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    importance: float  # 0.0 - 1.0 relevance score
    source_ref: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "source_ref": self.source_ref
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            memory_type=MemoryType(data["memory_type"]),
            content=data["content"],
            metadata=data["metadata"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            importance=data["importance"],
            source_ref=data.get("source_ref")
        )

class AdvancedMemoryManager:
    def __init__(self, google_api_key: Optional[str] = None):
        self.context_ttl = timedelta(minutes=int(os.getenv("CONTEXT_TTL_MINUTES", "60")))
        self.max_short_term_messages = int(os.getenv("MAX_SHORT_TERM_MESSAGES", "20"))
        self.chunk_size = int(os.getenv("CONVERSATION_CHUNK_SIZE", "1000"))
        self.embed_save_threshold = int(os.getenv("EMBED_SAVE_THRESHOLD", "30"))  # Default threshold for saving embeddings
        self.enable_google_embeddings = os.getenv("ENABLE_ADVANCED_MEMORY_EMBEDDINGS", "false").lower() in ("1", "true", "yes")
        self.google_api_key = google_api_key or os.getenv("GEMINI_API_KEY")
        
        # In-memory caches
        self.short_term_memory: Dict[str, List[Dict[str, Any]]] = {}
        self.long_term_memory: Dict[str, List[MemoryEntry]] = {}
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Vector embeddings cache (would connect to actual vector DB in production)
        self.embeddings_cache: Dict[str, List[float]] = {}
        
        # Profile service for database integration
        self.profile_service = ProfileService()
        
        # Initialize Google Generative AI if available and enabled
        if self.enable_google_embeddings and self.google_api_key and GOOGLE_GENERATIVE_AI_AVAILABLE:
            try:
                if genai is not None:
                    genai.configure(api_key=self.google_api_key)
                logger.info("Google Generative AI initialized for Advanced Memory")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Generative AI: {e}")
                self.enable_google_embeddings = False
        elif self.enable_google_embeddings:
            logger.warning("Google Generative AI not available but advanced memory embeddings are enabled")
            self.enable_google_embeddings = False
        
        logger.info(f"Advanced Memory Manager initialized (Google embeddings: {'enabled' if self.enable_google_embeddings else 'disabled'})")
    
    async def get_or_create_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get or create user profile with default structure, syncing with database when available"""
        try:
            # Validate user_id
            if not user_id:
                raise MemoryValidationError("User ID cannot be empty or None")
            if not isinstance(user_id, str):
                raise MemoryValidationError("User ID must be a string")
            if len(user_id) > 50:
                raise MemoryValidationError(f"User ID too long: {len(user_id)} characters (max 50)")
            if not user_id.strip():
                raise MemoryValidationError("User ID cannot be whitespace only")
            
            # Sanitize user_id
            user_id = user_id.strip()
            
            # Check if we have a cached profile
            if user_id not in self.user_profiles:
                # Try to get profile from database
                db_profile = None
                try:
                    db_profile = await self.profile_service.get_or_create_profile(user_id, auto_create=False)
                except Exception as e:
                    logger.debug(f"Failed to get profile from database for {user_id}: {e}")
                
                # Create profile with database information if available
                if db_profile:
                    self.user_profiles[user_id] = {
                        "user_id": user_id,
                        "name": getattr(db_profile, 'name', None),
                        "language": getattr(db_profile, 'language', 'English'),
                        "timezone": getattr(db_profile, 'timezone', 'UTC'),
                        "preferences": {},
                        "contact_info": {
                            "phone": getattr(db_profile, 'phone', user_id),
                            "persona": getattr(db_profile, 'persona', None),
                            "tags": getattr(db_profile, 'tags', [])
                        },
                        "created_at": getattr(db_profile, 'created_at', datetime.now().astimezone().isoformat()),
                        "last_interaction": None,
                        "interaction_count": 0,
                        "conversation_style": "professional",
                        "topics_of_interest": [],
                        "context_summary": ""
                    }
                    logger.debug(f"Created user profile for {user_id} from database")
                else:
                    # Create default profile
                    self.user_profiles[user_id] = {
                        "user_id": user_id,
                        "name": None,
                        "language": "English",
                        "timezone": "UTC",
                        "preferences": {},
                        "contact_info": {},
                        "created_at": datetime.now().astimezone().isoformat(),
                        "last_interaction": None,
                        "interaction_count": 0,
                        "conversation_style": "professional",
                        "topics_of_interest": [],
                        "context_summary": ""
                    }
                    logger.debug(f"Created new user profile for {user_id}")
            
            return self.user_profiles[user_id]
            
        except MemoryValidationError:
            raise
        except Exception as e:
            logger.error(f"Error creating user profile for {user_id}: {e}")
            raise MemoryStorageError(f"Failed to create user profile: {e}")
    
    async def add_short_term_memory(self, user_id: str, message: str, response: str, 
                                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add to short-term conversation memory"""
        try:
            # Validate inputs
            if not user_id:
                raise MemoryValidationError("User ID cannot be empty or None")
            if not isinstance(user_id, str):
                raise MemoryValidationError("User ID must be a string")
            if len(user_id) > 50:
                raise MemoryValidationError("User ID too long (max 50 characters)")
            if not user_id.strip():
                raise MemoryValidationError("User ID cannot be whitespace only")
            
            if not message:
                raise MemoryValidationError("Message cannot be empty or None")
            if not isinstance(message, str):
                raise MemoryValidationError("Message must be a string")
            if not message.strip():
                raise MemoryValidationError("Message cannot be whitespace only")
            
            if not response:
                raise MemoryValidationError("Response cannot be empty or None")
            if not isinstance(response, str):
                raise MemoryValidationError("Response must be a string")
            if not response.strip():
                raise MemoryValidationError("Response cannot be whitespace only")
            
            # Sanitize inputs
            user_id = user_id.strip()
            message = message.strip()[:2000]  # Limit message length
            response = response.strip()[:2000]  # Limit response length
            
            if user_id not in self.short_term_memory:
                self.short_term_memory[user_id] = []
            
            entry = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "message": message,
                "response": response,
                "metadata": metadata or {}
            }
            
            self.short_term_memory[user_id].append(entry)
            
            # Keep only the most recent messages
            if len(self.short_term_memory[user_id]) > self.max_short_term_messages:
                # Archive old messages before removing
                await self._archive_old_messages(user_id, self.short_term_memory[user_id][:-self.max_short_term_messages])
                self.short_term_memory[user_id] = self.short_term_memory[user_id][-self.max_short_term_messages:]
            
            # Update user profile
            profile = await self.get_or_create_user_profile(user_id)
            profile["last_interaction"] = datetime.now().astimezone().isoformat()
            profile["interaction_count"] += 1
            
            logger.debug(f"Added short-term memory for user {user_id}: {len(self.short_term_memory[user_id])} messages")
            
        except MemoryValidationError:
            raise
        except Exception as e:
            logger.error(f"Error adding short-term memory for {user_id}: {e}")
            raise MemoryStorageError(f"Failed to add short-term memory: {e}")
    
    async def add_long_term_memory(self, user_id: str, memory_type: MemoryType, 
                                 content: str, importance: float = 0.5,
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add to long-term persistent memory"""
        memory_id = str(uuid.uuid4())
        
        memory_entry = MemoryEntry(
            id=memory_id,
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
            timestamp=datetime.now().astimezone(),
            importance=importance
        )
        
        if user_id not in self.long_term_memory:
            self.long_term_memory[user_id] = []
        
        self.long_term_memory[user_id].append(memory_entry)
        
        # Generate embedding for vector search (placeholder - would use actual embedding service)
        await self._generate_embedding(memory_id, content)
        
        logger.info(f"Added long-term memory ({memory_type.value}) for user {user_id}: {content[:50]}...")
        return memory_id
    
    async def update_memory_score(self, user_id: str, memory_id: str, access_count: int = 1) -> None:
        """Update memory score based on access frequency, recency, and base importance"""
        try:
            if user_id in self.long_term_memory:
                for memory in self.long_term_memory[user_id]:
                    if memory.id == memory_id:
                        # Boost score based on access count (frequency factor)
                        frequency_boost = min(0.1 * access_count, 0.3)  # Cap frequency boost at 0.3
                        
                        # Calculate recency factor (0.0 to 1.0, newer memories get higher scores)
                        current_time = datetime.now().astimezone()
                        age_hours = (current_time - memory.timestamp).total_seconds() / 3600
                        # Recency score: 1.0 for very recent, 0.1 for very old (240 hours = 10 days)
                        recency_factor = max(0.1, 1.0 - (age_hours / 240.0))
                        
                        # Base importance factor (from memory type and content)
                        base_importance = memory.importance
                        
                        # Calculate composite score with weighted factors
                        # Recency: 40%, Frequency: 30%, Base Importance: 30%
                        composite_score = (
                            recency_factor * 0.4 +
                            min(frequency_boost, 1.0) * 0.3 +
                            base_importance * 0.3
                        )
                        
                        # Update the memory's importance score
                        memory.importance = min(composite_score, 1.0)
                        
                        # Update access metadata in memory's metadata field
                        if 'access_count' not in memory.metadata:
                            memory.metadata['access_count'] = 0
                        memory.metadata['access_count'] += access_count
                        
                        # Update timestamp to boost recency score for next calculation
                        memory.timestamp = current_time
                        break
        except Exception as e:
            logger.warning(f"Error updating memory score for {user_id}: {e}")
    
    async def decay_memory_scores(self, user_id: str) -> None:
        """Decay memory scores periodically based on age, access frequency, and memory type"""
        try:
            if user_id in self.long_term_memory:
                current_time = datetime.now().astimezone()
                retained_memories = []
                
                for memory in self.long_term_memory[user_id]:
                    # Calculate age in hours for more granular decay
                    age_hours = (current_time - memory.timestamp).total_seconds() / 3600
                    
                    # Base decay factor based on age
                    if age_hours > 720:  # Older than 30 days
                        base_decay = 0.85
                    elif age_hours > 168:  # Older than 7 days
                        base_decay = 0.95
                    elif age_hours > 24:  # Older than 1 day
                        base_decay = 0.98
                    else:  # Recent memories (less than 1 day)
                        base_decay = 0.995
                    
                    # Adjust decay based on access frequency
                    access_count = memory.metadata.get('access_count', 0)
                    if access_count > 10:  # Frequently accessed
                        frequency_factor = 1.05  # Slightly boost importance
                    elif access_count > 5:  # Moderately accessed
                        frequency_factor = 1.02
                    elif access_count > 1:  # Accessed a few times
                        frequency_factor = 0.99
                    else:  # Rarely accessed
                        frequency_factor = 0.95
                    
                    # Adjust decay based on memory type
                    type_factor = 1.0
                    if memory.memory_type == MemoryType.PROFILE:
                        type_factor = 1.1  # Boost profile information
                    elif memory.memory_type == MemoryType.PREFERENCE:
                        type_factor = 1.05  # Boost preferences
                    elif memory.memory_type == MemoryType.CONVERSATION_SUMMARY:
                        type_factor = 1.02  # Boost conversation summaries
                    elif memory.memory_type == MemoryType.ENTITY:
                        type_factor = 1.03  # Boost entities
                    
                    # Apply composite decay
                    composite_decay = base_decay * frequency_factor * type_factor
                    memory.importance *= composite_decay
                    
                    # Keep memories with importance > 0.05, archive others
                    # Lower threshold to retain more valuable memories
                    if memory.importance > 0.05:
                        retained_memories.append(memory)
                    else:
                        logger.debug(f"Decayed memory {memory.id} below threshold, removing (importance: {memory.importance:.3f})")
                
                self.long_term_memory[user_id] = retained_memories
                logger.debug(f"Decayed memory scores for user {user_id}: {len(retained_memories)} memories retained")
        except Exception as e:
            logger.warning(f"Error decaying memory scores for {user_id}: {e}")
    
    async def get_relevant_memories(self, user_id: str, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Get relevant memories with scoring and filtering"""
        try:
            # First decay scores to keep them fresh
            await self.decay_memory_scores(user_id)
            
            # Search memories
            memories = await self.search_memories_multi_query(user_id, query, limit=limit*2)  # Get more to filter
            
            # Filter out low-importance memories
            filtered_memories = [m for m in memories if m.importance > 0.2]
            
            # Sort by importance and return top results
            filtered_memories.sort(key=lambda x: x.importance, reverse=True)
            
            # Update scores for accessed memories
            for memory in filtered_memories[:limit]:
                await self.update_memory_score(user_id, memory.id)
            
            return filtered_memories[:limit]
        except Exception as e:
            logger.error(f"Error getting relevant memories for {user_id}: {e}")
            return []
    
    async def get_recent_context(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation context"""
        if user_id not in self.short_term_memory:
            return []
        
        messages = self.short_term_memory[user_id]
        return messages[-limit:] if messages else []
    
    async def search_memories(self, user_id: str, query: str, memory_types: Optional[List[MemoryType]] = None,
                            limit: int = 5) -> List[MemoryEntry]:
        """Search long-term memories using semantic similarity"""
        try:
            # Validate inputs
            if not user_id:
                raise MemoryValidationError("User ID cannot be empty or None")
            if not isinstance(user_id, str):
                raise MemoryValidationError("User ID must be a string")
            if len(user_id) > 50:
                raise MemoryValidationError("User ID too long (max 50 characters)")
            if not user_id.strip():
                raise MemoryValidationError("User ID cannot be whitespace only")
            
            # Handle None or empty query gracefully
            if not query:
                logger.debug("Search query is None or empty, returning empty results")
                return []
            if not isinstance(query, str):
                logger.debug("Search query is not a string, returning empty results")
                return []
            if not query.strip():
                logger.debug("Search query is whitespace only, returning empty results")
                return []
            
            if not isinstance(limit, int):
                raise MemoryValidationError("Limit must be an integer")
            if limit <= 0:
                raise MemoryValidationError("Limit must be positive")
            if limit > 100:
                raise MemoryValidationError("Limit too high (max 100)")
            
            # Sanitize inputs
            user_id = user_id.strip()
            query = query.strip()
            
            if user_id not in self.long_term_memory:
                logger.debug(f"No long-term memories found for user {user_id}")
                return []
            
            memories = self.long_term_memory[user_id]
            
            # Filter by memory types if specified
            if memory_types:
                try:
                    memories = [m for m in memories if m.memory_type in memory_types]
                except Exception as e:
                    logger.warning(f"Error filtering memories by type: {e}")
            
            # Simple keyword-based search (would use vector similarity in production)
            query_lower = query.lower()
            scored_memories = []
            
            for memory in memories:
                try:
                    # Handle None content gracefully
                    content = memory.content if memory.content is not None else ""
                    # Ensure content is a string
                    if not isinstance(content, str):
                        content = str(content)
                    content_lower = content.lower()
                    # Simple scoring based on keyword matches and importance
                    keyword_score = sum(1 for word in query_lower.split() if word in content_lower)
                    total_score = (keyword_score * 0.7) + (memory.importance * 0.3)
                    
                    if total_score > 0:
                        scored_memories.append((total_score, memory))
                except Exception as e:
                    logger.warning(f"Error scoring memory {memory.id}: {e}")
                    continue
            
            # Sort by relevance and return top results
            scored_memories.sort(key=lambda x: x[0], reverse=True)
            result = [memory for _, memory in scored_memories[:limit]]
            
            logger.debug(f"Memory search for '{query}' returned {len(result)} results")
            return result
            
        except MemoryValidationError:
            raise
        except Exception as e:
            logger.error(f"Error searching memories for {user_id}: {e}")
            raise MemorySearchError(f"Failed to search memories: {e}")
    
    async def get_user_summary(self, user_id: str) -> str:
        """Generate a comprehensive user summary for AI context"""
        profile = await self.get_or_create_user_profile(user_id)
        recent_context = await self.get_recent_context(user_id, limit=5)
        
        # Build summary sections
        summary_parts = []
        
        # Profile information
        if profile.get("name"):
            summary_parts.append(f"User: {profile['name']}")
        
        # Preferences and settings
        if profile.get("preferences"):
            prefs = ", ".join([f"{k}: {v}" for k, v in profile["preferences"].items() if v])
            if prefs:
                summary_parts.append(f"Preferences: {prefs}")
        
        # Language and communication style
        if profile.get("language", "English") != "English":
            summary_parts.append(f"Language: {profile['language']}")
        if profile.get("conversation_style"):
            summary_parts.append(f"Style: {profile['conversation_style']}")
        
        # Recent context summary
        if recent_context:
            last_topics = []
            for ctx in recent_context[-3:]:
                message = ctx.get("message", "")
                if len(message) > 0:
                    # Extract key topics (simple approach)
                    words = message.lower().split()
                    important_words = [w for w in words if len(w) > 4 and w.isalpha()]
                    last_topics.extend(important_words[:2])
            
            if last_topics:
                unique_topics = list(dict.fromkeys(last_topics))  # Remove duplicates while preserving order
                summary_parts.append(f"Recent topics: {', '.join(unique_topics[:5])}")
        
        # Interaction history
        interaction_count = profile.get("interaction_count", 0)
        if interaction_count > 0:
            summary_parts.append(f"Interactions: {interaction_count}")
        
        return " | ".join(summary_parts) if summary_parts else "New user"
    
    def _identify_repetitive_patterns(self, recent_context: List[Dict[str, Any]]) -> List[str]:
        """Identify repetitive patterns in recent conversation context"""
        if not recent_context:
            return []
        
        repetitive_patterns = []
        
        # Check for common repetitive phrases
        common_repetitive_phrases = [
            "whenever you are ready",
            "how can i help you",
            "is there anything else",
            "let me know if you need",
            "feel free to ask",
            "happy to assist",
            "good morning",
            "good afternoon",
            "good evening",
            "hope you're doing well",
            "hope this message finds you well",
            "have a great day",
            "looking forward to hearing from you",
            "don't hesitate to ask",
            "here to help",
            "at your service",
            "productive day",
            "efficient conversation"
        ]
        
        # Collect all responses
        responses = [ctx.get("response", "") for ctx in recent_context if ctx.get("response")]
        
        # Check for repeated phrases in responses
        for phrase in common_repetitive_phrases:
            count = sum(1 for response in responses if phrase.lower() in response.lower())
            if count > 1:  # If phrase appears more than once
                repetitive_patterns.append(phrase)
        
        # Check for exact repeated responses
        response_counts = {}
        for response in responses:
            # Normalize response for comparison
            normalized = response.lower().strip()
            response_counts[normalized] = response_counts.get(normalized, 0) + 1
        
        for response, count in response_counts.items():
            if count > 1 and len(response) > 10:  # If response appears more than once and is long enough
                repetitive_patterns.append(response[:50] + ("..." if len(response) > 50 else ""))
        
        # Check for repetitive greeting patterns
        greeting_patterns = ["good morning", "good afternoon", "good evening", "hello", "hi"]
        greeting_count = sum(1 for response in responses if any(greeting in response.lower() for greeting in greeting_patterns))
        if greeting_count > 2:  # If more than 2 greetings in recent context
            repetitive_patterns.append("excessive greetings")
        
        return repetitive_patterns[:5]  # Return top 5 repetitive patterns
    
    async def build_context_prompt(self, user_id: str, current_message: str,
                                 base_prompt: str, max_context_length: int = 2000, ai_handler=None) -> str:
        """Build enhanced prompt with user context and relevant memories using multi-query retrieval"""
        
        # Get user summary
        user_summary = await self.get_user_summary(user_id)
        
        # Use multi-query retrieval to find relevant memories
        relevant_memories = await self.search_memories_multi_query(user_id, current_message, ai_handler, limit=5)
        
        # Get recent conversation context
        recent_context = await self.get_recent_context(user_id, limit=5)  # Reduced from 10
        
        # Identify repetitive patterns in recent context
        repetitive_patterns = self._identify_repetitive_patterns(recent_context)
        
        # Build the enhanced prompt
        prompt_parts = [base_prompt]
        
        # Add user context
        if user_summary and user_summary != "New user":
            prompt_parts.append(f"\nUSER CONTEXT: {user_summary}")
        
        # Add relevant memories
        if relevant_memories:
            memory_text = "\nRELEVANT MEMORIES:"
            for memory in relevant_memories:
                memory_text += f"\n- {memory.content} ({memory.memory_type.value})"
            prompt_parts.append(memory_text)
        
        # Add recent conversation with emphasis on flow
        if recent_context:
            conversation_text = "\nRECENT CONVERSATION:"
            # Show last 2 exchanges to provide better context (reduced from 3)
            for i, ctx in enumerate(recent_context[-2:]):
                message = ctx.get("message", "")
                response = ctx.get("response", "")
                if message and response:
                    # Add indicators for conversation flow
                    flow_indicator = ""
                    if i == len(recent_context[-2:]) - 1:  # Last exchange
                        flow_indicator = " [Most Recent]"
                    conversation_text += f"\nUser: {message[:50]}..."  # Reduced length
                    conversation_text += f"\nAssistant: {response[:50]}...{flow_indicator}"  # Reduced length
            prompt_parts.append(conversation_text)
        
        # Add repetitive pattern warnings to avoid them
        if repetitive_patterns:
            pattern_text = "\nAVOID REPETITIVE PATTERNS:"
            for pattern in repetitive_patterns:
                pattern_text += f"\n- Avoid repeating: '{pattern}'"
            prompt_parts.append(pattern_text)
        
        # Current message
        prompt_parts.append(f"\nCURRENT MESSAGE: {current_message}")
        
        # Enhanced instructions for natural conversation flow - make it more concise
        prompt_parts.append("\nIMPORTANT INSTRUCTIONS:")
        prompt_parts.append("\n1. KEEP RESPONSES VERY CONCISE - AVOID VERBOSE OR LENGTHY RESPONSES")
        prompt_parts.append("\n2. For simple greetings like 'Hi', respond with a BRIEF acknowledgment (1-2 words max)")
        prompt_parts.append("\n3. NEVER provide conversation summaries unless explicitly requested")
        prompt_parts.append("\n4. DON'T explain what you're doing or why - just respond directly")
        prompt_parts.append("\n5. AVOID meta-commentary about the conversation or your processes")
        prompt_parts.append("\n6. NEVER repeat the same phrases, greetings, or sign-offs")
        prompt_parts.append("\n7. Keep responses to 1-2 sentences maximum for most interactions")
        
        # Combine and check length
        full_prompt = "".join(prompt_parts)
        
        # Truncate if too long (keep most important parts)
        if len(full_prompt) > max_context_length:
            # Keep base prompt, user summary, and current message
            essential_parts = [
                base_prompt,
                f"\nUSER CONTEXT: {user_summary}" if user_summary != "New user" else "",
                f"\nCURRENT MESSAGE: {current_message}",
                "\nRespond concisely and directly. Avoid summaries and verbose explanations."
            ]
            full_prompt = "".join([p for p in essential_parts if p])
        
        return full_prompt
    
    async def generate_multi_queries(self, current_message: str, ai_handler=None) -> List[str]:
        """Generate multiple queries with enhanced hypothetical question generation to improve memory retrieval"""
        queries = [current_message]  # Always include the original query
        
        # If we have an AI handler, use it to generate additional queries with better prompting
        if ai_handler is not None:
            try:
                # Enhanced prompt for generating more diverse and contextually relevant queries
                multi_query_prompt = f"""
You are an expert at reformulating questions to maximize information retrieval from a memory system.

Original question: "{current_message}"

Generate 4-6 alternative ways to ask the same question that would help retrieve relevant memories.

Consider these reformulation strategies:
1. Different phrasings (How vs What vs Why vs When)
2. Context-specific variations (past tense, future tense, conditional)
3. Broader and narrower scope questions
4. Related concept questions
5. Action-oriented questions
6. Information-seeking vs decision-making questions

Examples for "What did we decide last time?":
- "Previous meeting conclusion"
- "Last discussion outcome"
- "Agreed upon actions from our prior conversation"
- "What was the final decision on this matter?"
- "How did we resolve this previously?"
- "What actions were assigned in our last discussion?"

Return only the alternative queries, one per line, without any other text.
                """
                
                ai_response = await ai_handler.generate(multi_query_prompt)
                if ai_response and ai_response.get("text"):
                    # Split the response into lines and add non-empty lines as queries
                    additional_queries = [line.strip() for line in ai_response["text"].split('\n') if line.strip()]
                    queries.extend(additional_queries[:6])  # Limit to 6 additional queries
            except Exception as e:
                logger.warning(f"Failed to generate enhanced multi-queries: {e}")
        
        # Add more sophisticated standard query variations based on question type
        question_type = self._classify_question_type(current_message)
        standard_variations = self._generate_standard_variations(current_message, question_type)
        queries.extend(standard_variations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in queries:
            if query.lower() not in seen:
                seen.add(query.lower())
                unique_queries.append(query)
        
        return unique_queries[:10]  # Limit to 10 total queries
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question to generate appropriate variations"""
        question_lower = question.lower().strip()
        
        # Decision-based questions
        if any(word in question_lower for word in ['decide', 'decision', 'agree', 'conclude', 'conclusion', 'resolve']):
            return 'decision'
        
        # Information-seeking questions
        elif any(word in question_lower for word in ['what', 'how', 'when', 'where', 'who', 'which', 'tell me', 'explain']):
            return 'information'
        
        # Action-based questions
        elif any(word in question_lower for word in ['do', 'should', 'can', 'could', 'will', 'would', 'need', 'must']):
            return 'action'
        
        # Temporal questions
        elif any(word in question_lower for word in ['last', 'previous', 'before', 'earlier', 'recent', 'past', 'next', 'future']):
            return 'temporal'
        
        # Default
        return 'general'
    
    def _generate_standard_variations(self, current_message: str, question_type: str) -> List[str]:
        """Generate standard query variations based on question type"""
        variations = []
        
        # Base variations for all question types
        base_variations = [
            f"related to {current_message}",
            f"information about {current_message}",
            f"details on {current_message}",
            f"context for {current_message}"
        ]
        variations.extend(base_variations)
        
        # Type-specific variations
        if question_type == 'decision':
            variations.extend([
                f"decision regarding {current_message}",
                f"conclusion about {current_message}",
                f"agreed points on {current_message}",
                f"resolution of {current_message}"
            ])
        elif question_type == 'information':
            variations.extend([
                f"facts about {current_message}",
                f"key details of {current_message}",
                f"important points about {current_message}",
                f"relevant information on {current_message}"
            ])
        elif question_type == 'action':
            variations.extend([
                f"actions for {current_message}",
                f"steps to handle {current_message}",
                f"tasks related to {current_message}",
                f"procedures for {current_message}"
            ])
        elif question_type == 'temporal':
            variations.extend([
                f"previous {current_message}",
                f"historical {current_message}",
                f"earlier discussion of {current_message}",
                f"past events related to {current_message}"
            ])
        else:  # general
            variations.extend([
                f"overview of {current_message}",
                f"summary for {current_message}",
                f"background on {current_message}",
                f"discussion about {current_message}"
            ])
        
        return variations
    
    async def search_memories_multi_query(self, user_id: str, query: str, ai_handler=None, 
                                        limit: int = 5) -> List[MemoryEntry]:
        """Search memories using multiple query phrasings to improve recall"""
        try:
            # Generate multiple queries
            queries = await self.generate_multi_queries(query, ai_handler)
            
            # Search with each query and collect results
            all_results = []
            for q in queries:
                try:
                    results = await self.search_memories(user_id, q, limit=limit)
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(f"Error searching with query '{q}': {e}")
                    continue
            
            # Remove duplicates based on memory ID and sort by importance
            unique_results = {}
            for result in all_results:
                if result.id not in unique_results:
                    unique_results[result.id] = result
                else:
                    # Keep the one with higher importance
                    if result.importance > unique_results[result.id].importance:
                        unique_results[result.id] = result
            
            # Sort by importance and return top results
            sorted_results = sorted(unique_results.values(), key=lambda x: x.importance, reverse=True)
            return sorted_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in multi-query search for {user_id}: {e}")
            # Fall back to single query search
            return await self.search_memories(user_id, query, limit=limit)
    
    async def update_user_preference(self, user_id: str, key: str, value: Any) -> None:
        """Update user preference and add to long-term memory"""
        profile = await self.get_or_create_user_profile(user_id)
        profile["preferences"][key] = value
        
        # Also store as long-term memory
        await self.add_long_term_memory(
            user_id=user_id,
            memory_type=MemoryType.PREFERENCE,
            content=f"User preference: {key} = {value}",
            importance=0.8,
            metadata={"preference_key": key, "preference_value": value}
        )
        
        logger.info(f"Updated preference for {user_id}: {key} = {value}")
    
    async def extract_and_store_entities(self, user_id: str, text: str) -> List[str]:
        """Extract entities from text and store them"""
        try:
            # Validate inputs
            if not user_id:
                raise MemoryValidationError("User ID cannot be empty or None")
            if not isinstance(user_id, str):
                raise MemoryValidationError("User ID must be a string")
            if len(user_id) > 50:
                raise MemoryValidationError("User ID too long (max 50 characters)")
            if not user_id.strip():
                raise MemoryValidationError("User ID cannot be whitespace only")
            
            # Handle None or invalid text gracefully
            if text is None:
                logger.debug(f"Text is None for user {user_id}, returning empty entities")
                return []
            if not isinstance(text, str):
                logger.warning(f"Text is not a string for user {user_id}, converting to string")
                text = str(text)
            
            # Sanitize inputs
            user_id = user_id.strip()
            
            # Handle empty or very long text gracefully
            if not text or len(text.strip()) == 0:
                logger.debug(f"Empty text for user {user_id}, returning empty entities")
                return []
            
            # Limit text length to prevent excessive processing
            if len(text) > 5000:
                logger.warning(f"Text too long for user {user_id} ({len(text)} chars), truncating to 5000")
                text = text[:5000]
            
            import re
            entities = []
            
            try:
                # Extract phone numbers
                phones = re.findall(r"\+?\d[\d\s\-()]{6,}\d", text)
                for phone in phones:
                    try:
                        clean_phone = re.sub(r"[^0-9+]", "", phone)
                        if len(clean_phone) >= 7:  # Minimum valid phone length
                            entities.append(f"phone:{clean_phone}")
                            await self.add_long_term_memory(
                                user_id=user_id,
                                memory_type=MemoryType.ENTITY,
                                content=f"Phone number: {clean_phone}",
                                importance=0.9
                            )
                    except Exception as e:
                        logger.warning(f"Error processing phone {phone}: {e}")
                        continue
            except Exception as e:
                logger.warning(f"Error extracting phone numbers: {e}")
            
            try:
                # Extract email addresses
                emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
                for email in emails:
                    try:
                        if len(email) <= 100:  # Reasonable email length limit
                            entities.append(f"email:{email}")
                            await self.add_long_term_memory(
                                user_id=user_id,
                                memory_type=MemoryType.ENTITY,
                                content=f"Email address: {email}",
                                importance=0.9
                            )
                    except Exception as e:
                        logger.warning(f"Error processing email {email}: {e}")
                        continue
            except Exception as e:
                logger.warning(f"Error extracting email addresses: {e}")
            
            try:
                # Extract names
                name_patterns = [
                    r"my name is\s+([A-Za-z ,.'-]{2,50})",
                    r"i am\s+([A-Za-z ,.'-]{2,50})",
                    r"call me\s+([A-Za-z ,.'-]{2,30})",
                    r"i'm\s+([A-Za-z ,.'-]{2,50})",
                    r"name:\s*([A-Za-z ,.'-]{2,50})",
                    r"([A-Za-z ,.'-]{2,50})\s*,\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"  # Name before email
                ]
                
                for pattern in name_patterns:
                    try:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        for match in matches:
                            try:
                                name = match.strip().title()
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
                                contains_skip_word = any(skip_word in name.lower() for skip_word in skip_words)
                                
                                # Check if name matches non-name patterns
                                matches_non_name_pattern = any(re.search(pattern, name.lower()) for pattern in non_name_patterns)
                                
                                # Additional validation to ensure name makes sense
                                if (len(name) > 1 and len(name) <= 50 and name.replace(' ', '').isalpha() and 
                                    not contains_skip_word and not matches_non_name_pattern):
                                    entities.append(f"name:{name}")
                                    # Update profile only if it doesn't already have a name
                                    profile = await self.get_or_create_user_profile(user_id)
                                    if profile.get("name") is None and self._is_valid_name(name):
                                        profile["name"] = name
                                    
                                    await self.add_long_term_memory(
                                        user_id=user_id,
                                        memory_type=MemoryType.PROFILE,
                                        content=f"User's name: {name}",
                                        importance=1.0
                                    )
                                    break  # Only take the first valid name
                            except Exception as e:
                                logger.warning(f"Error processing name {match}: {e}")
                                continue
                    except Exception as e:
                        logger.warning(f"Error with name pattern {pattern}: {e}")
                        continue
            except Exception as e:
                logger.warning(f"Error extracting names: {e}")
            
            logger.debug(f"Extracted {len(entities)} entities from text")
            return entities
            
        except MemoryValidationError:
            raise
        except Exception as e:
            logger.error(f"Error in entity extraction for {user_id}: {e}")
            # Return empty list instead of crashing
            return []
    
    def _is_valid_name(self, name: str) -> bool:
        """Validate if a string is a reasonable name"""
        if not name or not isinstance(name, str):
            return False
            
        # Trim whitespace and punctuation
        name = name.strip().strip('.,;:!?')
        
        # Check basic requirements
        if len(name) < 2 or len(name) > 50:
            return False
            
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
        
        # If name contains skip words or matches non-name patterns, reject it
        if contains_skip_word or matches_non_name_pattern:
            return False
            
        # Split into tokens
        tokens = name.split()
        
        if len(tokens) == 0:
            return False
        elif len(tokens) == 1:
            # Single token, check if it's reasonable
            # Allow apostrophes and hyphens in names
            cleaned_token = tokens[0].replace("'", "").replace("-", "")
            # Check if token is reasonable
            if len(cleaned_token) >= 2 and cleaned_token.isalpha():
                # Additional check for single letter names (unlikely to be real names)
                if len(cleaned_token) < 2:
                    return False
                # Additional check for common greeting words
                if name.lower() in ['hi', 'hey', 'hello']:
                    return False
                return True
            else:
                return False
        else:
            # Multiple tokens, check if all tokens are reasonable
            name_tokens = tokens[:3]  # Allow up to 3 name parts
            
            # Additional validation: check if all tokens are reasonable name parts
            reasonable_name_parts = [
                'mr', 'mrs', 'ms', 'dr', 'jr', 'sr', 'ii', 'iii', 'iv', 'v',
                'von', 'van', 'de', 'di', 'le', 'la', 'du', 'des', 'del', 'della'
            ]
            
            # Check if tokens are reasonable
            valid_tokens = True
            for token in name_tokens:
                cleaned_token = token.replace("'", "").replace("-", "").lower()
                # Token should be alphabetic and not a common non-name word
                if not cleaned_token.isalpha() or (len(cleaned_token) < 2 and cleaned_token not in reasonable_name_parts):
                    valid_tokens = False
                    break
            
            if not valid_tokens:
                return False
                
            # Check each part for validity
            valid_parts = True
            for part in name_tokens:
                cleaned_part = part.replace("'", "").replace("-", "")
                if not (len(part) >= 1 and cleaned_part.isalpha()):
                    valid_parts = False
                    break
            
            if not valid_parts:
                return False
                
            # Check if any part contains skip words
            if any(skip_word in part.lower() for part in name_tokens for skip_word in skip_words):
                return False
                
            return True

    async def _extract_entities_with_ai(self, text: str, ai_handler=None) -> List[Dict[str, Any]]:
        """Extract entities from text using AI without storing them.
        
        Args:
            text: The text to extract entities from
            ai_handler: The AI handler to use for extraction
            
        Returns:
            List of extracted entities with their details
        """
        entities = []
        
        # Use AI for entity extraction if available
        if ai_handler is not None:
            try:
                # Create a prompt for entity extraction
                entity_extraction_prompt = f"""
Extract and identify the following entities from the text below:

Text: "{text}"

Please identify and categorize:
1. People (names, roles, relationships)
2. Organizations (companies, institutions)
3. Dates and times
4. Projects or initiatives
5. Preferences or important facts

For each entity, provide:
- Entity name
- Category
- Context (how it relates to the user)
- Confidence score (0.0-1.0)

Format your response as JSON:
{{
    "entities": [
        {{
            "name": "entity_name",
            "category": "person|organization|date|project|preference|other",
            "context": "contextual information",
            "confidence": 0.9
        }}
    ]
}}
                """
                
                # Get AI response
                ai_response = await ai_handler.generate(entity_extraction_prompt)
                
                if ai_response and ai_response.get("text"):
                    import json
                    try:
                        # Parse the JSON response
                        response_text = ai_response["text"]
                        # Extract JSON from the response if it's wrapped in other text
                        json_start = response_text.find("{")
                        json_end = response_text.rfind("}") + 1
                        if json_start != -1 and json_end > json_start:
                            json_text = response_text[json_start:json_end]
                            parsed_response = json.loads(json_text)
                            
                            # Process extracted entities
                            entities = parsed_response.get("entities", [])
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse AI entity extraction response: {e}")
            except Exception as e:
                logger.warning(f"AI entity extraction failed: {e}")
        
        return entities

    async def extract_and_store_entities_with_ai(self, user_id: str, text: str, ai_handler=None) -> List[str]:
        """Extract entities from text using AI and store them with improved linking"""
        try:
            # Validate inputs
            if not user_id:
                raise MemoryValidationError("User ID cannot be empty or None")
            if not isinstance(user_id, str):
                raise MemoryValidationError("User ID must be a string")
            if len(user_id) > 50:
                raise MemoryValidationError("User ID too long (max 50 characters)")
            if not user_id.strip():
                raise MemoryValidationError("User ID cannot be whitespace only")
            
            # Handle None or invalid text gracefully
            if text is None:
                logger.debug(f"Text is None for user {user_id}, returning empty entities")
                return []
            if not isinstance(text, str):
                logger.warning(f"Text is not a string for user {user_id}, converting to string")
                text = str(text)
            
            # Sanitize inputs
            user_id = user_id.strip()
            
            # Handle empty or very long text gracefully
            if not text or len(text.strip()) == 0:
                logger.debug(f"Empty text for user {user_id}, returning empty entities")
                return []
            
            # Limit text length to prevent excessive processing
            if len(text) > 5000:
                logger.warning(f"Text too long for user {user_id} ({len(text)} chars), truncating to 5000")
                text = text[:5000]
            
            entities = []
            
            # Use AI for entity extraction
            extracted_entities = await self._extract_entities_with_ai(text, ai_handler)
            
            # Process extracted entities
            for entity in extracted_entities:
                entity_name = entity.get("name")
                category = entity.get("category")
                context = entity.get("context")
                confidence = entity.get("confidence", 0.0)
                
                if entity_name and category:
                    entities.append(f"{category}:{entity_name}")
                    
                    # Store in long-term memory with appropriate importance
                    importance = min(confidence, 1.0)
                    await self.add_long_term_memory(
                        user_id=user_id,
                        memory_type=MemoryType.ENTITY,
                        content=f"{category.title()}: {entity_name} - {context}",
                        importance=importance,
                        metadata={
                            "entity_name": entity_name,
                            "entity_category": category,
                            "context": context,
                            "confidence": confidence
                        }
                    )
                    
                    # Update profile for person entities with improved linking
                    if category == "person" and confidence > 0.7:
                        profile = await self.get_or_create_user_profile(user_id)
                        # Create or update person entity linking
                        if "contacts" not in profile:
                            profile["contacts"] = {}
                        
                        # Normalize entity name for consistent linking
                        normalized_name = self._normalize_entity_name(entity_name)
                        
                        # Check if we already have this entity with a different name
                        existing_entity = self._find_existing_entity(profile["contacts"], entity_name)
                        if existing_entity:
                            # Update existing entity with new information
                            profile["contacts"][existing_entity]["mentions"] = profile["contacts"][existing_entity].get("mentions", 0) + 1
                            profile["contacts"][existing_entity]["last_mentioned"] = datetime.now().astimezone().isoformat()
                            # Update confidence if higher
                            if confidence > profile["contacts"][existing_entity].get("confidence", 0.0):
                                profile["contacts"][existing_entity]["confidence"] = confidence
                                profile["contacts"][existing_entity]["context"] = context
                        else:
                            # Create new entity
                            profile["contacts"][normalized_name] = {
                                "name": entity_name,
                                "context": context,
                                "confidence": confidence,
                                "mentions": 1,
                                "last_mentioned": datetime.now().astimezone().isoformat()
                            }
                        
                        # If this is likely the user's name, update profile
                        if "my name" in text.lower() or "i am" in text.lower():
                            profile["name"] = entity_name

            # Fall back to regex-based extraction if no entities were extracted by AI
            if not entities:
                logger.debug(f"No entities extracted by AI for user {user_id}, falling back to regex-based extraction")
                return await self.extract_and_store_entities(user_id, text)
            
            logger.debug(f"Extracted {len(entities)} entities from text using AI")
            return entities
        except MemoryValidationError:
            raise
        except Exception as e:
            logger.error(f"Error in AI entity extraction for {user_id}: {e}")
            # Fall back to regex-based extraction
            return await self.extract_and_store_entities(user_id, text)

    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for consistent linking"""
        # Remove titles and normalize
        titles = ["mr", "mrs", "ms", "dr", "prof", "sir", "madam"]
        normalized = name.lower().strip()
        for title in titles:
            if normalized.startswith(title + " "):
                normalized = normalized[len(title) + 1:]
                break
        return normalized.replace(" ", "_").replace(".", "").replace(",", "")
    
    def _find_existing_entity(self, contacts: Dict[str, Any], entity_name: str) -> Optional[str]:
        """Find existing entity with similar name for linking"""
        normalized_name = self._normalize_entity_name(entity_name)
        
        # Check for exact match
        if normalized_name in contacts:
            return normalized_name
        
        # Check for partial matches
        for contact_key, contact_info in contacts.items():
            contact_name = contact_info.get("name", "").lower()
            # Check if names are similar (using simple string matching)
            if normalized_name in contact_name or contact_name in normalized_name:
                return contact_key
            
            # Check for common variations
            name_parts = normalized_name.split("_")
            contact_parts = contact_name.split(" ")
            
            # If we have at least one matching part
            if any(part in contact_parts for part in name_parts if len(part) > 2):
                return contact_key
        
        return None
    
    async def summarize_conversation(self, user_id: str, conversation_metadata: Optional[Dict[str, Any]] = None) -> Any:
        """
        Generate a summary of the recent conversation context with metadata.
        
        Args:
            user_id: The user ID to summarize conversation for
            conversation_metadata: Additional metadata about the conversation
            
        Returns:
            Summary of the conversation
        """
        try:
            # Get recent context
            recent_context = await self.get_recent_context(user_id)
            if not recent_context:
                return "No recent conversation context available."
            
            # Build conversation transcript
            transcript_lines = []
            for ctx in recent_context:
                message = ctx.get('message', '') or ''
                response = ctx.get('response', '') or ''
                transcript_lines.append(f"User: {message}")
                transcript_lines.append(f"Assistant: {response}")
            
            transcript = "\n".join(transcript_lines)
            
            # Get user profile and preferences
            user_profile = await self.get_or_create_user_profile(user_id)
            user_preferences = user_profile.get("preferences", {})
            
            # Build contact info
            contact_info = {
                "name": user_preferences.get("name", "User"),
                "phone": user_id,
                "role": user_preferences.get("role", "Unknown"),
                "timezone": user_preferences.get("timezone", "UTC")
            }
            
            # Create summary prompt with metadata
            if conversation_metadata:
                metadata_str = "\nConversation Metadata:\n"
                for key, value in conversation_metadata.items():
                    metadata_str += f"- {key}: {value}\n"
                
                enhanced_transcript = f"{transcript}\n\n{metadata_str}"
            else:
                enhanced_transcript = transcript
            
            # Generate summary using prompts
            summary_prompt = build_detailed_summary_prompt(
                contact_display=contact_info["name"],
                contact_phone=contact_info["phone"],
                transcript=enhanced_transcript
            )
            
            current_summary_prompt = summary_prompt if summary_prompt else build_summary_prompt(
                contact_display=contact_info["name"],
                contact_phone=contact_info["phone"],
                transcript=transcript
            )
            
            # Extract key elements from the conversation
            key_topics = []
            action_items = []
            urgency_indicators = 0
            
            # Enhanced action item patterns
            action_patterns = {
                "Schedule meeting": [r"schedule.*meeting", r"set.*meeting", r"book.*meeting"],
                "Process payment": [r"payment", r"pay", r"invoice", r"bill"],
                "Provide information": [r"information", r"details", r"info", r"question", r"clarify"],
                "Fix issue": [r"fix", r"resolve", r"problem", r"issue", r"broken", r"error"],
                "Sign document": [r"sign", r"contract", r"agreement", r"document"],
                "Call back": [r"call", r"phone", r"contact.*phone"],
                "Review feedback": [r"feedback", r"review", r"evaluate"],
                "Process order": [r"order", r"purchase", r"buy", r"product"],
                "Resolve complaint": [r"complaint", r"dissatisfied", r"problem"],
                "Follow up": [r"follow.*up", r"opportunity", r"business"],
                "Review document": [r"document", r"file", r"report"],
                "Send email": [r"email", r"send", r"forward"],
                "Confirm details": [r"confirm", r"verify", r"check"],
                "Prepare proposal": [r"proposal", r"quote", r"estimate"]
            }
            
            # Analyze conversation for key elements
            for ctx in recent_context:
                message = (ctx.get('message', '') or '').lower()
                response = (ctx.get('response', '') or '').lower()
                
                # Extract key topics
                words = message.split()
                important_words = [w for w in words if len(w) > 4 and w.isalpha()]
                key_topics.extend(important_words)
                
                # Count urgency indicators
                urgency_keywords = ['urgent', 'asap', 'emergency', 'immediately', 'now', 'critical', 'important']
                urgency_indicators += sum(1 for word in urgency_keywords if word in message or word in response)
                
                # Enhanced action item extraction
                for action, patterns in action_patterns.items():
                    if any(re.search(pattern, message) for pattern in patterns):
                        # Extract a short snippet of the action context
                        context_snippet = message[:50] + ('...' if len(message) > 50 else '')
                        action_items.append(f"{action}: {context_snippet}")
            
            # Deduplicate topics while preserving order
            key_topics = list(dict.fromkeys(key_topics))
            
            # Build summary
            if key_topics:
                summary_text = f"Discussed {', '.join(key_topics[:3])} with {contact_info['name']}."
            else:
                summary_text = f"Conversation with {contact_info['name']} about {transcript[:100]}..."
            
            # Add urgency info
            if urgency_indicators > 0:
                summary_text += " [Urgent]"
            
            # Add action items
            if action_items:
                # Deduplicate action items
                unique_action_items = list(dict.fromkeys(action_items))
                summary_text += f"\nActions: {', '.join(unique_action_items[:3])}"
            
            # Retrieve relevant memories using RAG
            relevant_memories = []
            try:
                # Use the conversation summary as query for memory search
                # Ensure summary_text is not None before searching
                if summary_text is not None:
                    # Ensure summary_text is a string
                    if not isinstance(summary_text, str):
                        summary_text = str(summary_text)
                    if summary_text.strip():
                        relevant_memories = await self.search_memories(user_id, summary_text, limit=2)
                    else:
                        logger.debug("Skipping memory search - summary_text is empty")
                else:
                    logger.debug("Skipping memory search - summary_text is None")
            except Exception as e:
                logger.warning(f"Failed to retrieve relevant memories: {e}")
            
            # Create memory-enriched summary
            if relevant_memories and summary_text:
                memory_context = "\nRelevant Past Context:\n"
                for i, memory in enumerate(relevant_memories, 1):
                    content = memory.content if memory.content is not None else ""
                    # Ensure content is a string
                    if not isinstance(content, str):
                        content = str(content)
                    memory_context += f"{i}. {content[:100]}... ({memory.memory_type.value})\n"
            
            # Create memory entry for this summary
            # Ensure summary_text is not None
            safe_summary_text = summary_text if summary_text is not None else "Conversation summary could not be generated."
            # Ensure safe_summary_text is a string
            if not isinstance(safe_summary_text, str):
                safe_summary_text = str(safe_summary_text)
            
            summary_entry = MemoryEntry(
                id=f"summary_{user_id}_{datetime.now().astimezone().isoformat()}",
                user_id=user_id,
                content=safe_summary_text,
                memory_type=MemoryType.CONVERSATION_SUMMARY,
                importance=0.8,
                timestamp=datetime.now().astimezone(),
                metadata={
                    "summary_type": "conversation",
                    "source": "advanced_memory",
                    "context_length": len(recent_context)
                }
            )
            
            # Store summary in long-term memory
            # Ensure summary_text is not None
            safe_summary_text = summary_text if summary_text is not None else "Conversation summary could not be generated."
            # Ensure safe_summary_text is a string
            if not isinstance(safe_summary_text, str):
                safe_summary_text = str(safe_summary_text)
            
            await self.add_long_term_memory(
                user_id=user_id,
                memory_type=MemoryType.CONVERSATION_SUMMARY,
                content=safe_summary_text,
                importance=0.8,
                metadata={
                    "summary_type": "conversation",
                    "source": "advanced_memory",
                    "context_length": len(recent_context),
                    "contact": contact_info
                }
            )
            
            return summary_entry
            
        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            # Return a simple fallback summary
            return MemoryEntry(
                id=f"summary_{user_id}_{datetime.now().astimezone().isoformat()}",
                user_id=user_id,
                content="Conversation summary could not be generated due to an error.",
                memory_type=MemoryType.CONVERSATION_SUMMARY,
                importance=0.5,
                timestamp=datetime.now().astimezone(),
                metadata={"error": str(e)}
            )

    async def summarize_conversation_structured(self, user_id: str, summary_type: str = "short") -> Optional[str]:
        """
        Summarize recent conversation with structured templates
        
        Args:
            user_id: User ID
            summary_type: "short", "detailed", or "bullet"
            
        Returns:
            Structured summary string
        """
        recent_context = await self.get_recent_context(user_id, limit=10)
        
        if len(recent_context) < 3:
            return None
        
        # Create context text
        context_text = "\n".join([
            f"User: {ctx.get('message', '')}\nAssistant: {ctx.get('response', '')}" 
            for ctx in recent_context
        ])
        
        # For now, return a structured summary based on the simple approach
        # In production, this would use AI with structured prompts
        topics = []
        user_messages = []
        
        for ctx in recent_context:
            message = ctx.get("message", "")
            if message:
                user_messages.append(message)
                words = [w.lower() for w in message.split() if len(w) > 4 and w.isalpha()]
                topics.extend(words)
        
        topic_counts = {}
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        topic_list = [topic for topic, _ in top_topics]
        
        if summary_type == "short":
            summary = f"TL;DR: Discussed {', '.join(topic_list[:2])} in {len(recent_context)} exchanges"
        elif summary_type == "detailed":
            summary = (
                f"TL;DR: Conversation about {', '.join(topic_list)}\n"
                f"Highlights:\n"
                f"- {len(recent_context)} message exchanges\n"
                f"- Key topics: {', '.join(topic_list)}\n"
                f"- Recent discussion focus\n"
                f"Next Actions:\n"
                f"- Follow up on discussed topics\n"
                f"- Continue conversation as needed\n"
                f"- Review key points"
            )
        else:  # bullet
            summary = (
                f"• Topics: {', '.join(topic_list)}\n"
                f"• Messages: {len(recent_context)} exchanges\n"
                f"• Status: Completed discussion"
            )
        
        return summary
    
    async def chunk_conversation(self, conversation_text: str) -> List[str]:
        """
        Split a long conversation into chunks for hierarchical summarization.
        
        Args:
            conversation_text: The full conversation text to chunk
            
        Returns:
            List of conversation chunks
        """
        if len(conversation_text) <= self.chunk_size:
            return [conversation_text]
        
        chunks = []
        words = conversation_text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > self.chunk_size and current_chunk:
                # Create chunk and start new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    async def create_chunk_summaries(self, chunks: List[str], contact_info: Dict[str, Any]) -> List[str]:
        """
        Create summaries for each conversation chunk.
        
        Args:
            chunks: List of conversation chunks
            contact_info: Contact information for context
            
        Returns:
            List of chunk summaries
        """
        chunk_summaries = []
        
        for i, chunk in enumerate(chunks):
            try:
                # In a real implementation, this would use AI to summarize each chunk
                # For now, we'll create a simple placeholder summary
                summary = f"Chunk {i+1} summary: {chunk[:100]}..."
                chunk_summaries.append(summary)
            except Exception as e:
                logger.warning(f"Error creating summary for chunk {i}: {e}")
                chunk_summaries.append(f"Chunk {i+1} summary: [Error generating summary]")
        
        return chunk_summaries
    
    async def _archive_old_messages(self, user_id: str, old_messages: List[Dict[str, Any]]) -> None:
        """Archive old messages to long-term memory"""
        try:
            for message in old_messages:
                # Convert message to long-term memory entry
                await self.add_long_term_memory(
                    user_id=user_id,
                    memory_type=MemoryType.CONVERSATION_SUMMARY,
                    content=f"User: {message.get('message', '')}\nAssistant: {message.get('response', '')}",
                    importance=0.3,
                    metadata={
                        "archived": True,
                        "timestamp": message.get("timestamp", datetime.now().isoformat())
                    }
                )
            logger.debug(f"Archived {len(old_messages)} old messages for user {user_id}")
        except Exception as e:
            logger.warning(f"Error archiving old messages for user {user_id}: {e}")
    
    async def _update_rolling_summary(self, user_id: str, new_messages: List[Dict[str, Any]], ai_handler=None) -> None:
        """Create rolling summaries of conversation chunks with enhanced AI integration and topic detection"""
        try:
            # Check if we have enough messages to create a summary
            if len(new_messages) < 3:  # Need at least 3 messages for a meaningful summary
                return
            
            # Initialize sentiment variable
            dominant_sentiment = 'neutral'
            
            # Get user profile for context
            profile = await self.get_or_create_user_profile(user_id)
            
            # Create a conversation transcript from the new messages
            transcript = "\n".join([
                f"User: {msg.get('message', '')}\nAssistant: {msg.get('response', '')}"
                for msg in new_messages
            ])
            
            # Use AI to generate a more sophisticated summary with enhanced topic detection
            summary_content = None
            if ai_handler is not None:
                try:
                    # Enhanced prompt with better structure for topic detection
                    summary_prompt = f"""
Analyze the following conversation segment and create a comprehensive summary that captures:
1. Main topics discussed (identify key themes and subjects)
2. Key decisions made (explicit agreements or conclusions)
3. Action items identified (tasks, follow-ups, commitments)
4. User's intent or needs (what they're trying to accomplish)
5. Sentiment and tone (user's emotional state)
6. Context shifts (changes in subject or focus)

Conversation with {profile.get('name', 'user')}:
{transcript}

Provide a structured summary in this format:
TOPICS: [comma-separated main topics with brief descriptions]
DECISIONS: [key decisions made with brief explanations]
ACTIONS: [action items with responsible parties if mentioned]
INTENT: [user's primary goals or needs]
SENTIMENT: [user's emotional state during conversation]
CONTEXT: [notable context shifts or transitions]

Summary:
                    """.strip()
                    
                    ai_response = await ai_handler.generate(summary_prompt)
                    if ai_response and ai_response.get("text"):
                        summary_content = ai_response["text"]
                except Exception as e:
                    logger.warning(f"AI summary generation failed: {e}")
            
            # Fallback to enhanced simple summary if AI fails
            if not summary_content:
                # Enhanced topic extraction with better filtering
                all_text = " ".join([f"{msg.get('message', '')} {msg.get('response', '')}" for msg in new_messages])
                words = [w.lower() for w in all_text.split() if len(w) > 4 and w.isalpha() and w not in [
                    'that', 'this', 'with', 'have', 'from', 'what', 'when', 'where', 'which', 'who', 'whom', 'whose',
                    'why', 'how', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                    'between', 'among', 'within', 'without', 'toward', 'across', 'along', 'around', 'behind',
                    'beside', 'beyond', 'inside', 'outside', 'under', 'until', 'since', 'while', 'though',
                    'although', 'because', 'therefore', 'consequently', 'furthermore', 'moreover', 'nevertheless',
                    'however', 'otherwise', 'likewise', 'similarly', 'finally', 'eventually', 'eventually',
                    'previously', 'recently', 'currently', 'immediately', 'eventually', 'frequently', 'occasionally',
                    'sometimes', 'always', 'never', 'often', 'rarely', 'seldom', 'usually', 'generally',
                    'specifically', 'particularly', 'especially', 'primarily', 'mainly', 'largely', 'mostly',
                    'completely', 'entirely', 'fully', 'partially', 'slightly', 'somewhat', 'considerably',
                    'significantly', 'substantially', 'remarkably', 'notably', 'particularly', 'especially'
                ]]
                
                # Count word frequencies
                topic_counts = {}
                for word in words:
                    topic_counts[word] = topic_counts.get(word, 0) + 1
                
                # Sort by frequency and get top topics
                top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                topic_list = [topic for topic, count in top_topics if count > 1]  # Only include topics mentioned more than once
                
                # Enhanced summary with sentiment detection
                sentiment_indicators = {
                    'positive': ['good', 'great', 'excellent', 'wonderful', 'fantastic', 'amazing', 'awesome', 'brilliant'],
                    'negative': ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'frustrating', 'annoying', 'upset'],
                    'urgent': ['urgent', 'asap', 'immediately', 'now', 'quick', 'hurry', 'rush', 'emergency']
                }
                
                sentiment_scores = {'positive': 0, 'negative': 0, 'urgent': 0}
                for indicator_type, indicators in sentiment_indicators.items():
                    for indicator in indicators:
                        sentiment_scores[indicator_type] += all_text.lower().count(indicator)
                
                # Determine dominant sentiment
                dominant_sentiment = 'neutral'
                if max(sentiment_scores.values()) > 0:
                    dominant_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])[0]
                
                summary_content = f"Conversation segment with {profile.get('name', 'user')} covering topics: {', '.join(topic_list[:3])} (sentiment: {dominant_sentiment})"
            
            # Store as a memory node with higher importance for rolling summaries
            await self.add_long_term_memory(
                user_id=user_id,
                memory_type=MemoryType.CONVERSATION_SUMMARY,
                content=summary_content,
                importance=0.8,  # High importance for conversation segments
                metadata={
                    "summary_type": "rolling",
                    "message_count": len(new_messages),
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "chunk_id": str(uuid.uuid4()),  # Add unique identifier for tracking
                    "dominant_sentiment": dominant_sentiment if 'dominant_sentiment' in locals() else 'neutral'
                }
            )
            
            logger.debug(f"Created enhanced rolling summary for user {user_id} with {len(new_messages)} messages")
        except Exception as e:
            logger.warning(f"Error creating enhanced rolling summary for user {user_id}: {e}")

    async def process_conversation_chunk(self, user_id: str, messages: List[Dict[str, Any]], ai_handler=None) -> None:
        """Process a chunk of conversation messages and create summaries when needed"""
        try:
            # Add messages to short-term memory
            for message in messages:
                if user_id not in self.short_term_memory:
                    self.short_term_memory[user_id] = []
                
                self.short_term_memory[user_id].append(message)
                
                # Keep only the most recent messages
                if len(self.short_term_memory[user_id]) > self.max_short_term_messages:
                    # Archive old messages before removing
                    old_messages = self.short_term_memory[user_id][:-self.max_short_term_messages]
                    await self._archive_old_messages(user_id, old_messages)
                    self.short_term_memory[user_id] = self.short_term_memory[user_id][-self.max_short_term_messages:]
            
            # Check if we need to create a rolling summary
            if len(self.short_term_memory.get(user_id, [])) >= 10:  # Every 10 messages
                # Take the last 10 messages for summarization
                recent_messages = self.short_term_memory[user_id][-10:]
                await self._update_rolling_summary(user_id, recent_messages, ai_handler)
                
                # Clear some older messages to prevent memory bloat
                if len(self.short_term_memory[user_id]) > 15:
                    # Keep the last 5 messages, archive the rest
                    messages_to_archive = self.short_term_memory[user_id][:-5]
                    await self._archive_old_messages(user_id, messages_to_archive)
                    self.short_term_memory[user_id] = self.short_term_memory[user_id][-5:]
            
            # Update user profile
            profile = await self.get_or_create_user_profile(user_id)
            profile["last_interaction"] = datetime.now().astimezone().isoformat()
            profile["interaction_count"] += len(messages)
            
        except Exception as e:
            logger.error(f"Error processing conversation chunk for {user_id}: {e}")

    async def get_memory_stats(self, user_id: str):
        try:
            short_term_count = len(self.short_term_memory.get(user_id, []))
            long_term_count = len(self.long_term_memory.get(user_id, []))
            embeddings_count = len([k for k in self.embeddings_cache.keys() if k.startswith(user_id)]) if self.embeddings_cache else 0
            
            return {
                "short_term_messages": short_term_count,
                "long_term_memories": long_term_count,
                "cached_embeddings": embeddings_count,
                "memory_system": "advanced"
            }
        except Exception as e:
            logger.error(f"Error getting memory stats for {user_id}: {e}")
            return {
                "short_term_messages": 0,
                "long_term_memories": 0,
                "cached_embeddings": 0,
                "memory_system": "advanced",
                "error": str(e)
            }
    
    async def _generate_embedding(self, memory_id: str, content: str) -> None:
        """Generate embedding for memory content (placeholder implementation)"""
        try:
            # This is a placeholder implementation - in a real system, this would use
            # an embedding model to generate vector representations of the content
            # For now, we'll just store a mock embedding
            if self.enable_google_embeddings and self.google_api_key and GOOGLE_GENERATIVE_AI_AVAILABLE and genai is not None:
                try:
                    # Generate embedding using Google Generative AI
                    embedding = genai.embed_content(model='models/embedding-001', content=content)
                    if embedding and hasattr(embedding, 'embedding'):
                        self.embeddings_cache[memory_id] = embedding.embedding
                        logger.debug(f"Generated embedding for memory {memory_id}")
                    else:
                        # Fallback to mock embedding
                        self.embeddings_cache[memory_id] = [0.1] * 768  # Mock embedding
                except Exception as e:
                    logger.warning(f"Failed to generate Google embedding: {e}")
                    # Fallback to mock embedding
                    self.embeddings_cache[memory_id] = [0.1] * 768  # Mock embedding
            else:
                # Fallback to mock embedding
                self.embeddings_cache[memory_id] = [0.1] * 768  # Mock embedding
        except Exception as e:
            logger.warning(f"Error generating embedding for memory {memory_id}: {e}")
            # Ensure we always have a fallback
            self.embeddings_cache[memory_id] = [0.1] * 768  # Mock embedding
