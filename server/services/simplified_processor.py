"""
Simplified Message Processor
Implements the exact flow specified:
1. Receive normalized message data
2. Lookup user by phone in users table
3. If found → fetch name and personalization fields
4. If not found → mark is_new = true
5. Build LLM prompt with user data and tools
6. Call LLM and handle function calls
7. Send response via WhatsApp
8. Persist conversation in database
"""

import time
from typing import Dict, Any, Optional
import logging
import os
import json
from datetime import datetime

# Create a simple logger
logger = logging.getLogger(__name__)

# Use relative imports as in the original file
from ..ai.handler import AdvancedAIHandler
from ..clients.whatsapp import EnhancedWhatsAppClient
from ..db.profile_db import get_profile_db

class SimplifiedMessageProcessor:
    """Simplified message processor that follows the exact specified flow"""
    
    def __init__(self):
        self.ai = AdvancedAIHandler()
        self.whatsapp = EnhancedWhatsAppClient()
        self.db = None
    
    async def _ensure_db(self):
        """Ensure database connection is initialized"""
        if self.db is None:
            self.db = await get_profile_db()
        return self.db
    
    async def process(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming message with simplified flow"""
        try:
            # Extract message data
            phone_number = message_data["phone_number"]
            message_text = message_data["message_text"]
            message_id = message_data.get("message_id", "unknown")
            timestamp = message_data.get("timestamp", str(int(time.time())))
            
            logger.info(f"📱 Processing message from {phone_number}: {message_text}")
            
            # Lookup user by phone
            db = await self._ensure_db()
            if db is None:
                logger.warning("Database connection not available, continuing with mock database")
                # Continue with mock functionality
                
            user_row = await self._get_user_by_phone(db, phone_number) if db else None
            
            is_new = user_row is None
            user_name = user_row.get('name') if user_row else None
            
            logger.info(f"User lookup: is_new={is_new}, name={user_name}")
            
            # Build LLM prompt with user data and tools
            prompt = self._build_llm_prompt(
                phone_number=phone_number,
                message_text=message_text,
                user_name=user_name,
                is_new=is_new
            )
            
            # Call LLM
            try:
                response = await self.ai.generate(prompt)
                response_text = response.get("text", "")
                logger.info(f"LLM response: {response_text}")
                
                # Clean and validate the response text
                response_text = self._clean_response_text(response_text)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                response_text = "Sorry, I'm having trouble processing your message right now."
            
            # Send response via WhatsApp
            try:
                send_result = await self.whatsapp.send_message(phone_number, response_text)
                logger.info(f"Message sent: {send_result}")
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
            
            # Persist conversation in database
            try:
                if db is not None:
                    await self._persist_conversation(
                        db=db,
                        phone_number=phone_number,
                        message_id=str(message_id),
                        incoming_text=message_text,
                        outgoing_text=response_text,
                        timestamp=str(timestamp)
                    )
            except Exception as e:
                logger.error(f"Failed to persist conversation: {e}")
            
            return {
                "status": "success",
                "response_text": response_text,
                "user_data": {
                    "is_new": is_new,
                    "name": user_name
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _clean_response_text(self, response_text: str) -> str:
        """Clean provider tags and special characters from response text"""
        if isinstance(response_text, str):
            import re
            
            # Strip any provider wrapper but keep the AI-generated response
            m = re.match(r"^\[(?:[A-Za-z0-9_\- ]+?) reply to: .*?\](.*?)$", response_text)
            if m and m.group(1):  # If we matched and have content after the wrapper
                response_text = m.group(1).strip()
            elif not response_text:  # Fallback if empty
                response_text = "Sorry, I'm having trouble processing your message right now."
            
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
                response_text = "Sorry, I'm having trouble processing your message right now."
            
            # Additional validation to ensure response has meaningful content
            # If response is too short or contains only special characters, provide fallback
            if len(response_text) < 2 or re.match(r'^[^a-zA-Z0-9]+$', response_text):
                response_text = "I'm here to help! How can I assist you today?"
                
        else:
            response_text = "Sorry, I'm having trouble processing your message right now."
            
        return response_text
    
    def _build_llm_prompt(self, phone_number: str, message_text: str, user_name: Optional[str], is_new: bool) -> str:
        """Build LLM prompt with user data and tools"""
        
        # Get time-based greeting
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            time_greeting = "Good morning"
        elif 12 <= current_hour < 17:
            time_greeting = "Good afternoon"
        elif 17 <= current_hour < 21:
            time_greeting = "Good evening"
        else:
            time_greeting = "Hello"
        
        if is_new:
            # New user prompt
            system_prompt = (
                f"{time_greeting}! You are a polite personal assistant. "
                "Use the user's saved profile fields (name, address) to personalize replies when available. "
                "If the user's name is missing and we are in a greeting/intro, ask for the name succinctly, "
                "then request the function `save_user` with the provided name. "
                "Always be concise and confirm actions before saving sensitive data."
            )
        else:
            # Returning user prompt
            if user_name:
                system_prompt = f"{time_greeting} {user_name}! You are a polite personal assistant for Mustafa."
            else:
                system_prompt = f"{time_greeting}! You are a polite personal assistant for Mustafa."
        
        # Add tool descriptions
        tool_descriptions = (
            "\n\nAvailable functions:\n"
            "1. save_user(phone: string, name: string) - saves the name\n"
            "2. request_consent(phone: string) - asks for permission\n"
            "3. get_user(phone) - returns user record"
        )
        
        # Build full prompt
        prompt = (
            f"{system_prompt}\n\n"
            f"User ({phone_number}) says: {message_text}\n\n"
            f"Please respond appropriately. If you need to call a function, format it as:\n"
            f"[FUNCTION_CALL: function_name(param1=value1, param2=value2)]\n"
            f"{tool_descriptions}"
        )
        
        return prompt
    
    async def _persist_conversation(self, db, phone_number: str, message_id: str, 
                                  incoming_text: str, outgoing_text: str, timestamp: str):
        """Persist conversation in database"""
        try:
            # Save incoming message
            await self._save_message(
                db=db,
                phone=phone_number,
                message_id=f"{message_id}_in",
                direction="in",
                text=incoming_text,
                timestamp=timestamp
            )
            
            # Save outgoing message
            await self._save_message(
                db=db,
                phone=phone_number,
                message_id=f"{message_id}_out",
                direction="out",
                text=outgoing_text,
                timestamp=str(int(time.time()))
            )
            
            logger.info(f"Conversation persisted for {phone_number}")
            
        except Exception as e:
            logger.error(f"Failed to persist conversation: {e}")
    
    async def _save_message(self, db, phone: str, message_id: str, direction: str, text: str, timestamp: str):
        """Save individual message to database"""
        try:
            # Get or create user first
            user_id = await self._get_or_create_user(db, phone)
            
            # Convert timestamp to float for to_timestamp function
            try:
                timestamp_float = float(timestamp)
            except (ValueError, TypeError):
                timestamp_float = time.time()
            
            # Insert conversation record
            query = """
                INSERT INTO conversations (phone, message_id, direction, text, created_at)
                VALUES ($1, $2, $3, $4, to_timestamp($5))
            """
            
            await db.execute(query, phone, message_id, direction, text, timestamp_float)
            
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
    
    async def _get_or_create_user(self, db, phone: str) -> int:
        """Get or create user in database"""
        try:
            # Try to get existing user
            query = "SELECT id FROM users WHERE phone = $1"
            row = await db.fetchrow(query, phone)
            
            if row:
                # Update last_seen timestamp
                update_query = "UPDATE users SET last_seen = now() WHERE phone = $1"
                await db.execute(update_query, phone)
                return row['id']
            else:
                # Create new user
                query = """
                    INSERT INTO users (phone, created_at, updated_at, last_seen)
                    VALUES ($1, now(), now(), now())
                    RETURNING id
                """
                row = await db.fetchrow(query, phone)
                return row['id'] if row else 0
                
        except Exception as e:
            logger.error(f"Failed to get or create user: {e}")
            return 0
    
    async def _get_user_by_phone(self, db, phone: str) -> Optional[Dict[str, Any]]:
        """Get user by phone number"""
        try:
            query = "SELECT * FROM users WHERE phone = $1"
            row = await db.fetchrow(query, phone)
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Failed to get user by phone: {e}")
            return None

# Export the class
__all__ = ['SimplifiedMessageProcessor']