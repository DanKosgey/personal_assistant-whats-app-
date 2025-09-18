import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, memory, advanced_memory):
        self.memory = memory
        self.advanced_memory = advanced_memory
        
    async def store_context(self, sender: str, text: str, response_text: str, contact_info: Dict[str, Any]):
        """Store message and response in both memory systems"""
        try:
            # Store message and response in both memory systems
            await self.memory.add_to_context(sender, text, response_text)  # Existing system
            await self.advanced_memory.add_short_term_memory(sender, text, response_text)  # New system
            
            # Extract and store entities using advanced memory
            entities = await self.advanced_memory.extract_and_store_entities(sender, text)
            if entities:
                logger.info(f"Extracted entities for {sender}: {entities}")
                
        except Exception as e:
            logger.error(f"Error storing context: {e}")
            
    async def update_preferences(self, sender: str, text: str, name: Optional[str], role: Optional[str]):
        """Update user preferences in both memory systems"""
        lower = text.lower()
        
        # Update name preference
        if "my name is" in lower and name:
            await self.memory.set_preference(sender, "name", name)  # Legacy
            await self.advanced_memory.update_user_preference(sender, "name", name)  # Advanced

        # Update role preference
        if role:
            await self.memory.set_preference(sender, "role", role)
            await self.advanced_memory.update_user_preference(sender, "role", role)

        # Update timezone preference
        if "timezone" in lower:
            if "utc" in text.upper() or "gmt" in text.upper():
                await self.memory.set_preference(sender, "timezone", text)
                await self.advanced_memory.update_user_preference(sender, "timezone", text)

        # Update work hours preference
        if "work hours" in lower or "working hours" in lower:
            if ":" in text and "-" in text:
                await self.memory.set_preference(sender, "work_hours", text)
                await self.advanced_memory.update_user_preference(sender, "work_hours", text)
                
    async def get_recent_context(self, sender: str) -> List[Dict[str, Any]]:
        """Get recent conversation context with intelligent fallbacks"""
        try:
            # Try advanced memory first
            context = await self.advanced_memory.get_recent_context(sender)
            if context:
                return context
        except Exception as e:
            logger.warning(f"Failed to get context from advanced memory: {e}")
        
        # Fallback to legacy memory
        try:
            return await self.memory.get_recent_context(sender)
        except Exception as e:
            logger.warning(f"Failed to get context from legacy memory: {e}")
            return []
        
    async def get_user_memory(self, sender: str):
        """Get user memory with intelligent fallbacks"""
        try:
            # Try advanced memory first
            profile = await self.advanced_memory.get_or_create_user_profile(sender)
            if profile:
                return profile
        except Exception as e:
            logger.warning(f"Failed to get user memory from advanced memory: {e}")
        
        # Fallback to legacy memory
        try:
            return await self.memory.get_user_memory(sender)
        except Exception as e:
            logger.warning(f"Failed to get user memory from legacy memory: {e}")
            return None
            
    async def get_preference(self, sender: str, preference_name: str, default=None):
        """Get a user preference with intelligent fallbacks"""
        try:
            # Try advanced memory first
            profile = await self.advanced_memory.get_or_create_user_profile(sender)
            if profile and "preferences" in profile:
                preference = profile["preferences"].get(preference_name)
                if preference is not None:
                    return preference
        except Exception as e:
            logger.warning(f"Failed to get preference from advanced memory: {e}")
        
        # Fallback to legacy memory
        try:
            return await self.memory.get_preference(sender, preference_name, default)
        except Exception as e:
            logger.warning(f"Failed to get preference from legacy memory: {e}")
            return default
        
    async def add_contact(self, sender: str, contact_info: Dict[str, Any]):
        """Add contact to memory with intelligent fallbacks"""
        try:
            # Try advanced memory first
            profile = await self.advanced_memory.get_or_create_user_profile(sender)
            if profile and contact_info:
                # Update profile with contact info
                for key, value in contact_info.items():
                    if key in ["name", "phone", "timezone", "language"]:
                        profile[key] = value
                    else:
                        profile["contact_info"][key] = value
        except Exception as e:
            logger.warning(f"Failed to add contact to advanced memory: {e}")
        
        # Also add to legacy memory
        try:
            await self.memory.add_contact(sender, contact_info)
        except Exception as e:
            logger.warning(f"Failed to add contact to legacy memory: {e}")
            
    async def build_context_prompt(self, sender: str, current_message: str, base_prompt: str) -> str:
        """Build context prompt with intelligent fallbacks"""
        try:
            # Try advanced memory first
            return await self.advanced_memory.build_context_prompt(sender, current_message, base_prompt)
        except Exception as e:
            logger.warning(f"Failed to build context prompt with advanced memory: {e}")
        
        # Fallback to legacy approach
        try:
            recent_context = await self.get_recent_context(sender)
            if recent_context:
                context_text = "\n".join([
                    f"User: {ctx.get('message', '')}\nAssistant: {ctx.get('response', '')}" 
                    for ctx in recent_context[-5:]  # Last 5 exchanges
                ])
                return f"{base_prompt}\n\nRecent conversation:\n{context_text}\n\nCurrent message: {current_message}"
            else:
                return f"{base_prompt}\n\nCurrent message: {current_message}"
        except Exception as e:
            logger.warning(f"Failed to build context prompt with legacy approach: {e}")
            return f"{base_prompt}\n\nCurrent message: {current_message}"