import logging
import re
import asyncio
import os
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta

# Fix the import path for ProfileService - use absolute import
from server.services.profile_service import ProfileService
from server.models.profiles import UpsertProfileRequest, ConsentType, ConsentMethod
from server.tools.time_tools import LLMTimeTools
from server.services.attachment_handler import AttachmentHandler

logger = logging.getLogger(__name__)

class ContactAssistant:
    def __init__(self, profile_service: ProfileService, whatsapp_client):
        self.profile_service = profile_service
        self.whatsapp_client = whatsapp_client
        self.time_tools = LLMTimeTools()
        self.attachment_handler = AttachmentHandler(whatsapp_client)
        
        # Get owner number from environment
        self.owner_number = os.getenv('PA_OWNER_NUMBER')
        
        # Greeting templates for initial contact with time-based greetings (shorter for WhatsApp)
        self.initial_greetings = [
            "Good {time_greeting}! I believe we haven't met before. Can you share your name? I'm Boniface's assistant.",
            "Hi there! I don't think we've met. What's your name?",
            "Hello! I don't know your name yet. Could you tell me what I should call you?"
        ]
        
        # Confirmation templates for when name is provided (shorter for WhatsApp)
        self.confirmation_messages = [
            "Thanks {name}! Nice to meet you. How can I help you today?",
            "Great to meet you {name}! What can I do for you?",
            "Thanks {name}! How can I assist you?"
        ]
        
        # Simple response templates for known contacts with simple greetings
        self.simple_responses = [
            "Hello {name}!",
            "Hi {name}!",
            "Hey {name}!",
            "Hello!",
            "Hi!",
            "Hey!"
        ]
        
    def _get_time_greeting(self) -> str:
        """Get appropriate time-based greeting"""
        time_info = self.time_tools.get_current_time()
        hour = time_info['hour']
        
        # Determine appropriate time greeting with more nuanced timing
        if 5 <= hour < 12:
            if 5 <= hour < 9:
                return "early morning"
            elif 9 <= hour < 12:
                return "morning"
        elif 12 <= hour < 17:
            if 12 <= hour < 14:
                return "afternoon"
            elif 14 <= hour < 17:
                return "afternoon"
        elif 17 <= hour < 21:
            if 17 <= hour < 19:
                return "evening"
            elif 19 <= hour < 21:
                return "evening"
        else:
            if 21 <= hour <= 23 or 0 <= hour < 5:
                return "night"
        
        # Fallback
        return "day"
    
    def _get_personalized_greeting(self, contact_info: Dict[str, Any]) -> str:
        """Generate a personalized greeting based on time, relationship, and VIP status"""
        time_greeting = self._get_time_greeting()
        name = contact_info.get("name")
        relationship = contact_info.get("relationship", "unknown")
        is_vip = contact_info.get("vip", False)
        
        # For simple responses, avoid time-based greetings
        if name:
            # Use simple greeting without time component to avoid repetition
            return f"Hi {name}!"
        else:
            return "Hello!"
        
    async def process_contact(self, sender: str, message_text: str, message_id: str, attachments: Optional[List[Dict[str, Any]]] = None) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Process contact information for an incoming message.
        
        Returns:
            Tuple of (should_continue_processing, contact_info, response_text)
        """
        try:
            # Normalize phone number
            logger.debug(f"Contact assistant processing sender: {sender}")
            normalized_phone = self._normalize_phone(sender)
            logger.debug(f"Contact assistant normalized phone: {sender} -> {normalized_phone}")
            
            # Check if sender is the owner
            is_owner = self._is_owner(normalized_phone)
            
            # Get or create profile with better error handling
            profile = await self.profile_service.get_or_create_profile(normalized_phone)
            
            # Handle case when profile database is not available
            if profile is None:
                logger.warning(f"Profile service returned None for {normalized_phone} - database may be temporarily unavailable")
                # Try one more time with a slight delay to handle race conditions
                logger.debug(f"Retrying profile fetch for {normalized_phone} after 100ms delay")
                await asyncio.sleep(0.1)  # 100ms delay
                profile = await self.profile_service.get_or_create_profile(normalized_phone)
                
                if profile is None:
                    # Try one final time with a longer delay
                    logger.debug(f"Final retry profile fetch for {normalized_phone} after 500ms delay")
                    await asyncio.sleep(0.5)  # 500ms delay
                    profile = await self.profile_service.get_or_create_profile(normalized_phone)
                    
                    if profile is None:
                        # As a last resort, create a minimal temporary profile for this session only
                        logger.error(f"Profile service still returned None for {normalized_phone} after all retries")
                        logger.warning(f"Creating temporary profile for {normalized_phone} to avoid service disruption")
                        
                        # Create a minimal contact info object to continue processing
                        contact_info = {
                            "name": None,
                            "phone": normalized_phone,
                            "description": None,
                            "tags": [],
                            "is_owner": is_owner,
                            "relationship": "unknown",
                            "vip": False,
                            "attachments": []
                        }
                        
                        # For owner, we still want to process but with a warning
                        if is_owner:
                            # Return empty string for owner to continue processing without notification
                            return True, contact_info, ""
                        
                        # For regular users, continue processing without notifying about technical difficulties
                        return True, contact_info, ""
                    else:
                        logger.info(f"Successfully fetched profile for {normalized_phone} on final retry")
            
            # Special handling for owner
            if is_owner:
                return await self._process_owner_contact(profile, normalized_phone)
            
            # Process any attachments
            attachment_summaries = []
            if attachments:
                for attachment in attachments:
                    processed_attachment = await self.attachment_handler.process_attachment(attachment, normalized_phone)
                    attachment_summary = self.attachment_handler.get_attachment_summary(processed_attachment)
                    attachment_summaries.append(attachment_summary)
            
            # Check if this is a name response to a previous greeting
            if getattr(profile, 'attributes', {}).get('pending_identity', False) and not getattr(profile, 'name', None):
                # This is likely a response to our name request
                parsed_name = self._parse_name(message_text)
                if parsed_name:
                    # Update profile with the provided name
                    await self._update_contact_name(profile, parsed_name)
                    confirmation = self.confirmation_messages[0].format(name=parsed_name)
                    contact_info = {
                        "name": parsed_name,
                        "phone": normalized_phone,
                        "description": getattr(profile, 'description', None),
                        "tags": getattr(profile, 'tags', []),
                        "is_owner": False,
                        "relationship": getattr(profile, 'attributes', {}).get('relationship', 'unknown'),
                        "vip": 'vip' in getattr(profile, 'tags', []),
                        "attachments": attachment_summaries
                    }
                    return True, contact_info, confirmation
                else:
                    # Could not parse name, ask for clarification
                    return True, None, "Sorry, I didn't catch your name. How should I address you?"
            
            # Check if we have the user's name - if not, this is a new contact
            if not getattr(profile, 'name', None):
                # Contact info for new users
                contact_info = {
                    "name": None,
                    "phone": normalized_phone,
                    "description": getattr(profile, 'description', None),
                    "tags": getattr(profile, 'tags', []),
                    "is_owner": False,
                    "relationship": getattr(profile, 'attributes', {}).get('relationship', 'unknown'),
                    "vip": 'vip' in getattr(profile, 'tags', []),
                    "attachments": attachment_summaries,
                    "is_new_contact": True  # Flag to indicate this is a new contact
                }
                
                # Check if we've already sent a greeting to this contact recently
                last_greeting_time = getattr(profile, 'attributes', {}).get('last_greeting_time')
                should_send_greeting = True
                
                if last_greeting_time:
                    try:
                        # Parse the last greeting time
                        last_greeting_dt = datetime.fromisoformat(last_greeting_time)
                        # If less than 5 minutes have passed, don't send another greeting
                        if datetime.now() - last_greeting_dt < timedelta(minutes=5):
                            should_send_greeting = False
                    except Exception:
                        # If parsing fails, proceed with sending greeting
                        pass
                
                if should_send_greeting:
                    # Grant consent for updating attributes
                    await self._grant_consent_for_attributes(profile)
                    
                    # Update last greeting time in profile
                    try:
                        update_request = UpsertProfileRequest(
                            phone=profile.phone,
                            fields={
                                'attributes': {**getattr(profile, 'attributes', {}), 'last_greeting_time': datetime.now().isoformat()}
                            },
                            reason="Updated last greeting time",
                            expected_version=None,
                            session_id=None
                        )
                        await self.profile_service.upsert_profile(update_request)
                    except Exception as e:
                        logger.error(f"Failed to update last greeting time for {profile.phone}: {e}")
                    
                    # Mark profile as pending identity collection
                    await self._mark_pending_identity(profile)
                    
                    # Instead of sending a fixed greeting, let the AI handle this naturally
                    # Return contact info with no name and continue processing
                    return True, contact_info, ""
                else:
                    # Don't send a greeting, just continue with normal processing
                    return True, None, ""
            else:
                # Contact is complete, continue with normal processing for existing users
                contact_info = {
                    "name": getattr(profile, 'name', None),
                    "phone": normalized_phone,
                    "description": getattr(profile, 'description', None),
                    "tags": getattr(profile, 'tags', []),
                    "is_owner": is_owner,
                    "relationship": getattr(profile, 'attributes', {}).get('relationship', 'unknown'),
                    "vip": getattr(profile, 'tags', []) and 'vip' in getattr(profile, 'tags', []) or False,
                    "attachments": attachment_summaries  # Include attachment summaries
                }
            
            # Extract relationship and VIP status from message
            relationship, is_vip = await self._extract_relationship_and_vip_status(profile, message_text)
            
            # Update profile with relationship and VIP status if they were extracted
            profile_needs_update = False
            update_fields = {}
            
            # Update relationship if extracted and not already set
            current_relationship = getattr(profile, 'attributes', {}).get('relationship')
            if relationship and not current_relationship:
                update_fields['attributes'] = {**getattr(profile, 'attributes', {}), 'relationship': relationship}
                profile_needs_update = True
            
            # Update VIP status if detected and not already set
            current_tags = getattr(profile, 'tags', [])
            if is_vip and 'vip' not in current_tags:
                update_fields['tags'] = current_tags + ['vip']
                profile_needs_update = True
            
            # Apply profile updates if needed
            if profile_needs_update:
                try:
                    # Grant consent for updating attributes if needed
                    if 'attributes' in update_fields:
                        await self._grant_consent_for_attributes(profile)
                    
                    update_request = UpsertProfileRequest(
                        phone=profile.phone,
                        fields=update_fields,
                        reason="Auto-updated from conversation analysis",
                        expected_version=None,
                        session_id=None
                    )
                    await self.profile_service.upsert_profile(update_request)
                    logger.info(f"Auto-updated profile for {profile.phone} with relationship={relationship}, VIP={is_vip}")
                except Exception as e:
                    logger.error(f"Failed to auto-update profile for {profile.phone}: {e}")
            
            # If we have the user's name, add time greeting for personalized responses
            if contact_info["name"]:
                contact_info["time_greeting"] = self._get_time_greeting()
            
            # Special handling for simple greetings to ensure personalization
            # Only treat actual greetings as greetings, not questions or other messages
            lower_text = message_text.lower().strip()
            is_simple_greeting = lower_text in ["hi", "hello", "hey"] or \
                                (len(lower_text.split()) == 1 and lower_text in ["hi", "hello", "hey"]) or \
                                (len(lower_text.split()) == 2 and lower_text in ["hi there", "hello there"])
            
            if contact_info["name"] and is_simple_greeting:
                # Check if we've recently sent a greeting to avoid repetition
                last_greeting_time = getattr(profile, 'attributes', {}).get('last_greeting_time')
                should_send_greeting = True
                
                if last_greeting_time:
                    try:
                        # Parse the last greeting time
                        last_greeting_dt = datetime.fromisoformat(last_greeting_time)
                        # If less than 5 minutes have passed, don't send another greeting
                        if datetime.now() - last_greeting_dt < timedelta(minutes=5):
                            should_send_greeting = False
                    except Exception:
                        # If parsing fails, proceed with sending greeting
                        pass
                
                if should_send_greeting:
                    # Grant consent for updating attributes
                    await self._grant_consent_for_attributes(profile)
                    
                    # Update last greeting time in profile BEFORE sending the greeting
                    try:
                        update_request = UpsertProfileRequest(
                            phone=profile.phone,
                            fields={
                                'attributes': {**getattr(profile, 'attributes', {}), 'last_greeting_time': datetime.now().isoformat()}
                            },
                            reason="Updated last greeting time",
                            expected_version=None,
                            session_id=None
                        )
                        await self.profile_service.upsert_profile(update_request)
                    except Exception as e:
                        logger.error(f"Failed to update last greeting time for {profile.phone}: {e}")
                    
                    # Generate personalized greeting for simple greetings
                    import random
                    personalized_greeting = random.choice(self.simple_responses).format(name=contact_info["name"])
                    return True, contact_info, personalized_greeting
                else:
                    # Don't send a greeting, just continue with normal processing
                    # Use minimal response for simple greetings
                    return True, contact_info, ""
            
            # Generate personalized greeting for returning contacts with sufficient profile
            if contact_info["name"] and len(getattr(profile, 'tags', [])) > 0:
                # For returning contacts, we can use the personalized greeting in the response
                # This would be used by the main processor to enhance responses
                pass
            
            return True, contact_info, ""
            
        except Exception as e:
            logger.error(f"Error processing contact for {sender}: {e}")
            logger.exception(e)  # Log full traceback
            # Even in case of error, we should try to continue processing with minimal contact info
            try:
                # Simple normalization without using class methods to avoid issues
                if sender and sender.startswith('+'):
                    normalized_phone = sender
                elif sender and sender.isdigit():
                    normalized_phone = f"+{sender}"
                else:
                    normalized_phone = sender
                        
                contact_info = {
                    "name": None,
                    "phone": normalized_phone,
                    "description": None,
                    "tags": [],
                    "is_owner": False,
                    "relationship": "unknown",
                    "vip": False,
                    "attachments": []
                }
                # Continue processing without notifying the user about technical issues
                return True, contact_info, ""
            except Exception as fallback_e:
                logger.error(f"Fallback error processing contact for {sender}: {fallback_e}")
                # Continue processing without notifying the user about technical issues
                return False, None, ""

    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number to E.164 format"""
        if not phone:
            return phone
            
        # Remove all non-digit characters except +
        cleaned = re.sub(r'[^\d+]', '', phone)
    
        # Add + if missing and starts with country code
        if not cleaned.startswith('+') and len(cleaned) > 10:
            cleaned = '+' + cleaned
    
        # Validate format (similar to profile service)
        if not re.match(r'^\+?[1-9]\d{6,14}$', cleaned):
            logger.warning(f"Invalid phone number format after normalization: {phone} -> {cleaned}")
            # Still return the cleaned version to avoid breaking the flow
            return cleaned
        
        logger.debug(f"Contact assistant normalized phone: {phone} -> {cleaned}")
        return cleaned
    
    def _parse_name(self, text: str) -> Optional[str]:
        """Extract a reasonable display name from a free-text reply with improved extraction and validation"""
        if not text:
            return None
            
        # Strip salutations and courtesy phrases
        text = re.sub(r'^(this is|my name is|i am|i\'m)\s+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(mr\.?|mrs\.?|ms\.?|dr\.?)\s+', '', text, flags=re.IGNORECASE)
        
        # Trim whitespace and punctuation
        text = text.strip().strip('.,;:!?')
        
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
        contains_skip_word = any(skip_word in text.lower() for skip_word in skip_words)
        
        # Check if name matches non-name patterns
        matches_non_name_pattern = any(re.search(pattern, text.lower()) for pattern in non_name_patterns)
        
        # If name contains skip words or matches non-name patterns, reject it
        if contains_skip_word or matches_non_name_pattern:
            logger.debug(f"Rejected name '{text}' as it contains skip words or matches non-name patterns")
            return None
        
        # Split into tokens
        tokens = text.split()
        
        if len(tokens) == 0:
            return None
        elif len(tokens) == 1:
            # Single token, use as name
            name = tokens[0].capitalize()
            # Additional validation for single token names
            # Allow apostrophes and hyphens in names
            cleaned_token = tokens[0].replace("'", "").replace("-", "")
            # Check if token is reasonable
            if len(name) >= 2 and len(name) <= 50 and cleaned_token.isalpha():
                # Additional check for single letter names (unlikely to be real names)
                if len(cleaned_token) < 2:
                    return None
                # Additional check for common greeting words
                if name.lower() in ['hi', 'hey', 'hello']:
                    return None
                return name
            else:
                return None
        else:
            # Multiple tokens, use first two as First Last (but allow up to 3 for longer names)
            name_tokens = tokens[:3]  # Allow up to 3 name parts to prevent truncation
            
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
                return None
            
            # Capitalize each token but preserve apostrophes and hyphens
            capitalized_tokens = []
            for token in name_tokens:
                # Capitalize but preserve special characters
                if "'" in token or "-" in token:
                    # Handle names like O'Connor or Mary-Jane
                    parts = re.split(r"(['\-])", token)
                    capitalized_parts = [part.capitalize() if part.isalpha() else part for part in parts]
                    capitalized_tokens.append("".join(capitalized_parts))
                else:
                    capitalized_tokens.append(token.capitalize())
        
            name = ' '.join(capitalized_tokens)
            # Additional validation for multi-token names
            if len(name) >= 2 and len(name) <= 50:
                # Check each part for validity
                valid_parts = True
                for part in name_tokens:
                    cleaned_part = part.replace("'", "").replace("-", "")
                    if not (len(part) >= 1 and cleaned_part.isalpha()):
                        valid_parts = False
                        break
                
                if valid_parts:
                    # Check if any part contains skip words
                    if not any(skip_word in part.lower() for part in name_tokens for skip_word in skip_words):
                        return name
        return None
    
    async def _extract_relationship_and_vip_status(self, profile, message_text: str) -> Tuple[Optional[str], bool]:
        """Extract relationship and VIP status from message text and existing profile"""
        # Start with existing relationship if available
        relationship = getattr(profile, 'attributes', {}).get('relationship', None)
        vip = False
        
        # Check for VIP tags in existing profile
        if getattr(profile, 'tags', []):
            vip = 'vip' in getattr(profile, 'tags', [])
        
        # Extract relationship from message if not already set
        if not relationship:
            lower_text = message_text.lower()
            relationship_keywords = {
                'client': ['client', 'customer', 'customer service', 'service request', 'order', 'purchase'],
                'friend': ['friend', 'buddy', 'pal', 'hang out', 'catch up', 'personal'],
                'family': ['family', 'brother', 'sister', 'mom', 'dad', 'parent', 'uncle', 'aunt', 'cousin'],
                'colleague': ['colleague', 'coworker', 'work with', 'work together', 'team', 'department', 'office'],
                'business': ['business', 'partner', 'vendor', 'supplier', 'contract', 'deal', 'proposal'],
                'lawyer': ['lawyer', 'attorney', 'legal', 'court', 'case', 'lawsuit'],
                'doctor': ['doctor', 'physician', 'medical', 'appointment', 'prescription', 'health'],
                'assistant': ['assistant', 'help', 'support', 'question', 'inquiry']
            }
            
            # Look for relationship indicators in the message
            for rel, keywords in relationship_keywords.items():
                if any(keyword in lower_text for keyword in keywords):
                    relationship = rel
                    break
        
        # Check for VIP indicators with more sophisticated patterns
        if not vip:
            lower_text = message_text.lower()
            # VIP indicators with weights
            vip_indicators = {
                'urgent': 3,
                'asap': 3,
                'emergency': 3,
                'immediately': 2,
                'important': 2,
                'priority': 2,
                'vip': 3,
                'executive': 2,
                'manager': 2,
                'ceo': 3,
                'director': 2,
                'critical': 2,
                'time sensitive': 2,
                'deadline': 2
            }
            
            # Calculate VIP score
            vip_score = 0
            for indicator, weight in vip_indicators.items():
                if indicator in lower_text:
                    vip_score += weight
            
            # Set VIP flag if score exceeds threshold
            vip = vip_score >= 3
        
        return relationship, vip
    
    async def _update_contact_name(self, profile, name: str) -> None:
        """Update contact with provided name and clear pending identity flag"""
        try:
            # Extract relationship and VIP status from message
            # This would need to be called with the message text
            # For now, we'll just update the name
            
            update_request = UpsertProfileRequest(
                phone=profile.phone,
                fields={
                    'name': name,
                    'attributes': {**getattr(profile, 'attributes', {}), 'pending_identity': False}
                },
                reason="Name provided by user",
                expected_version=None,
                session_id=None
            )
            
            await self.profile_service.upsert_profile(update_request)
            logger.info(f"Updated contact name for {profile.phone} to {name}")
            
        except Exception as e:
            logger.error(f"Failed to update contact name for {profile.phone}: {e}")
    
    async def _grant_consent_for_attributes(self, profile) -> bool:
        """Grant consent for updating attributes field"""
        try:
            consent_response = await self.profile_service.update_consent(
                phone=profile.phone,
                consent_type=ConsentType.MEMORY_STORAGE,
                granted=True,
                method=ConsentMethod.API_CALL,
                actor="contact_assistant"
            )
            
            if not consent_response.success:
                logger.warning(f"Failed to grant consent for {profile.phone}: {consent_response.message}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error granting consent for {profile.phone}: {e}")
            return False
    
    async def _mark_pending_identity(self, profile) -> None:
        """Mark a profile as pending identity collection"""
        try:
            # Grant consent for updating attributes
            await self._grant_consent_for_attributes(profile)
            
            update_request = UpsertProfileRequest(
                phone=profile.phone,
                fields={
                    'attributes': {**getattr(profile, 'attributes', {}), 'pending_identity': True}
                },
                reason="Pending identity collection",
                expected_version=None,
                session_id=None
            )
            
            await self.profile_service.upsert_profile(update_request)
            logger.info(f"Marked profile {profile.phone} as pending identity")
            
        except Exception as e:
            logger.error(f"Failed to mark profile {profile.phone} as pending identity: {e}")
            
    def _is_owner(self, phone: str) -> bool:
        """Check if the phone number belongs to the owner"""
        if not self.owner_number:
            return False
            
        # Normalize both numbers for comparison
        normalized_owner = self._normalize_phone(self.owner_number)
        normalized_phone = self._normalize_phone(phone)
        
        return normalized_owner == normalized_phone
        
    async def _process_owner_contact(self, profile, normalized_phone: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """Process contact information for the owner"""
        contact_info = {
            "name": getattr(profile, 'name', None) or "Boniface",
            "phone": normalized_phone,
            "description": getattr(profile, 'description', None),
            "tags": getattr(profile, 'tags', []),
            "is_owner": True,
            "relationship": "owner",
            "vip": True  # Owner is always VIP
        }
        
        # If owner doesn't have a name set, we can set a default one
        if not contact_info["name"]:
            contact_info["name"] = "Boniface"
            
        # Generate a personalized greeting for the owner
        time_greeting = self._get_time_greeting()
        greeting = f"Good {time_greeting}, Boniface! I'm your WhatsApp assistant. How can I help you today?"
        
        # Return contact info with personalized greeting for the owner
        return True, contact_info, greeting