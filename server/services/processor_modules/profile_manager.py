import logging
from typing import Dict, Any, Optional
import re

logger = logging.getLogger(__name__)

class ProfileManager:
    def __init__(self, profile_service):
        self.profile_service = profile_service
    
    async def update_from_message(self, sender: str, message_text: str, response_text: str) -> None:
        """Extract and update profile information from message with enhanced persona detection"""
        try:
            # Get existing profile
            profile = await self.profile_service.get_or_create_profile(sender, auto_create=False)
            if not profile:
                return
            
            # Extract information from message
            extracted_fields = {}
            lower_message = message_text.lower()
            lower_response = response_text.lower()
            
            # Enhanced name extraction - only if profile doesn't already have a name
            if not getattr(profile, 'name', None):
                name = self._extract_name(message_text)
                if name:
                    extracted_fields['name'] = name
            
            # Enhanced role/occupation extraction
            role = self._extract_role(message_text)
            if role:
                extracted_fields['persona'] = role[:64]  # Limit length
            
            # Enhanced timezone extraction
            timezone = self._extract_timezone(message_text)
            if timezone:
                extracted_fields['timezone'] = timezone
            
            # Enhanced language preference extraction
            language = self._extract_language_preference(message_text)
            if language:
                extracted_fields['language'] = language
            
            # Extract relationship context from conversation
            relationship = self._extract_relationship_context(message_text, response_text)
            if relationship:
                # Update attributes with relationship context
                current_attributes = getattr(profile, 'attributes', {})
                if 'relationship' not in current_attributes or not current_attributes.get('relationship'):
                    extracted_fields['attributes'] = {**current_attributes, 'relationship': relationship}
            
            # Extract interests/topics from conversation
            interests = self._extract_interests(message_text, response_text)
            if interests:
                # Update tags with interests
                current_tags = getattr(profile, 'tags', [])
                new_tags = list(set(current_tags + interests))  # Deduplicate
                if len(new_tags) > len(current_tags):  # Only update if new tags were added
                    extracted_fields['tags'] = new_tags
            
            # Update profile if we extracted any fields
            if extracted_fields:
                from server.models.profiles import UpsertProfileRequest
                upsert_request = UpsertProfileRequest(
                    phone=sender,
                    fields=extracted_fields,
                    reason=f"Extracted from conversation: {message_text[:50]}...",
                    expected_version=None,
                    session_id=None
                )
                
                profile_response = await self.profile_service.upsert_profile(
                    upsert_request, actor="profile_manager"
                )
                
                if profile_response.success:
                    logger.info(f"Updated profile for {sender} with fields: {list(extracted_fields.keys())}")
                else:
                    logger.warning(f"Failed to update profile for {sender}: {profile_response.message}")
                    
        except Exception as e:
            logger.error(f"Error updating profile from message: {e}")
    
    def _extract_name(self, text: str) -> Optional[str]:
        """Extract name with enhanced pattern matching and validation"""
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
                    'broke', 'poor', 'rich', 'busy', 'tired', 'hungry', 'thirsty', 'angry', 
                    'happy', 'sad', 'excited', 'loan', 'debt', 'money', 'cash', 'house', 'home',
                    'authorities', 'involved', 'problem', 'issue', 'help', 'assistance', 
                    'developer', 'engineer', 'manager', 'director', 'doctor', 'lawyer',
                    'teacher', 'student', 'professor', 'nurse', 'accountant', 'consultant',
                    'sales', 'marketing', 'support', 'service', 'customer', 'client',
                    'urgent', 'important', 'asap', 'emergency', 'immediate', 'critical',
                    'request', 'order', 'purchase', 'buy', 'sell', 'payment', 'price',
                    'cost', 'budget', 'quote', 'invoice', 'bill', 'charge', 'fee', 'amount',
                    'deposit', 'balance', 'account', 'transaction', 'transfer', 'send',
                    'receive', 'withdraw', 'deposit', 'credit', 'debit', 'card', 'loan',
                    'mortgage', 'insurance', 'policy', 'claim', 'benefit', 'pension',
                    'retirement', 'investment', 'stock', 'bond', 'fund', 'portfolio',
                    'meeting', 'appointment', 'schedule', 'calendar', 'event', 'party',
                    'celebration', 'birthday', 'anniversary', 'wedding', 'graduation',
                    'conference', 'seminar', 'workshop', 'training', 'course', 'class',
                    'question', 'answer', 'solution', 'problem', 'issue', 'trouble',
                    'difficulty', 'challenge', 'obstacle', 'barrier', 'hurdle', 'impediment',
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
                contains_skip_word = any(skip_word in extracted_name.lower() for skip_word in skip_words)
                
                # Check if name matches non-name patterns
                matches_non_name_pattern = any(re.search(pattern, extracted_name.lower()) for pattern in non_name_patterns)
                
                # If name contains skip words or matches non-name patterns, reject it
                if contains_skip_word or matches_non_name_pattern:
                    logger.debug(f"Rejected name '{extracted_name}' as it contains skip words or matches non-name patterns")
                    return None
                
                # Split into tokens and validate
                tokens = extracted_name.split()
                if tokens and all(len(token) >= 1 and len(token) <= 20 and token.replace('-', '').replace("'", '').isalpha() for token in tokens):
                    # Additional validation: check if all tokens are reasonable name parts
                    reasonable_name_parts = [
                        'mr', 'mrs', 'ms', 'dr', 'jr', 'sr', 'ii', 'iii', 'iv', 'v',
                        'von', 'van', 'de', 'di', 'le', 'la', 'du', 'des', 'del', 'della'
                    ]
                    
                    # Check if tokens are reasonable
                    valid_tokens = True
                    for token in tokens:
                        cleaned_token = token.replace("'", "").replace("-", "").lower()
                        # Token should be alphabetic and not a common non-name word
                        if not cleaned_token.isalpha() or (len(cleaned_token) < 2 and cleaned_token not in reasonable_name_parts):
                            valid_tokens = False
                            break
                
                    if valid_tokens:
                        # Additional check for common greeting words
                        first_token = tokens[0].lower()
                        if first_token in ['hi', 'hey', 'hello']:
                            return None
                        
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
                        return ' '.join(capitalized_tokens)
        return None
    
    def _extract_role(self, text: str) -> Optional[str]:
        """Extract role/occupation with enhanced detection"""
        lower = text.lower()
        role = None
        role_patterns = [
            (r"my role is\s+([A-Za-z0-9\s,.'\-]{2,64})", "role"),
            (r"i work as\s+([A-Za-z0-9\s,.'\-]{2,64})", "work"),
            (r"i am a\s+([A-Za-z0-9\s,.'\-]{2,64})", "role"),
            (r"i['’]m a\s+([A-Za-z0-9\s,.'\-]{2,64})", "role"),
            (r"my job is\s+([A-Za-z0-9\s,.'\-]{2,64})", "job"),
            (r"i work in\s+([A-Za-z0-9\s,.'\-]{2,64})", "work"),
        ]
        
        for pattern, pattern_type in role_patterns:
            match = re.search(pattern, lower)
            if match:
                extracted = match.group(1).strip().title()
                if len(extracted) > 1 and len(extracted) <= 64:
                    # Map common roles to standardized personas
                    role_mapping = {
                        'manager': 'Business Manager',
                        'director': 'Executive Director',
                        'developer': 'Software Developer',
                        'engineer': 'Engineer',
                        'doctor': 'Medical Professional',
                        'lawyer': 'Legal Professional',
                        'teacher': 'Educator',
                        'student': 'Student',
                        'sales': 'Sales Professional',
                        'marketing': 'Marketing Professional',
                    }
                    role = role_mapping.get(extracted.lower(), extracted)
                    break
        return role
    
    def _extract_timezone(self, text: str) -> Optional[str]:
        """Extract timezone information"""
        lower = text.lower()
        if 'timezone' in lower or 'time zone' in lower:
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
                    return match.group(1)
        return None
    
    def _extract_language_preference(self, text: str) -> Optional[str]:
        """Extract language preference"""
        lower = text.lower()
        lang_keywords = {
            'english': 'en', 'french': 'fr', 'spanish': 'es', 'german': 'de',
            'swahili': 'sw', 'arabic': 'ar', 'mandarin': 'zh', 'hindi': 'hi',
            'kiswahili': 'sw', 'kikuyu': 'ki', 'luo': 'luo'
        }
        
        for lang_name, lang_code in lang_keywords.items():
            if lang_name in lower and ('speak' in lower or 'language' in lower or 'prefer' in lower):
                return lang_code
        return None
    
    def _extract_relationship_context(self, message_text: str, response_text: str) -> Optional[str]:
        """Extract relationship context from conversation"""
        combined_text = (message_text + " " + response_text).lower()
        relationship_keywords = {
            'client': ['client', 'customer', 'customer service', 'service request'],
            'friend': ['friend', 'buddy', 'pal', 'hang out', 'catch up'],
            'family': ['family', 'brother', 'sister', 'mom', 'dad', 'parent', 'uncle', 'aunt', 'cousin'],
            'colleague': ['colleague', 'coworker', 'work with', 'work together', 'team', 'department'],
            'business': ['business', 'partner', 'vendor', 'supplier', 'contract', 'deal'],
            'professional': ['professional', 'consultant', 'advisor', 'expert'],
        }
        
        for relationship, keywords in relationship_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return relationship
        return None
    
    def _extract_interests(self, message_text: str, response_text: str) -> list:
        """Extract interests/topics from conversation"""
        combined_text = (message_text + " " + response_text).lower()
        interest_keywords = {
            'technology': ['tech', 'technology', 'software', 'app', 'digital', 'computer', 'ai', 'artificial intelligence'],
            'business': ['business', 'entrepreneur', 'startup', 'company', 'corporate'],
            'health': ['health', 'medical', 'doctor', 'hospital', 'wellness', 'fitness'],
            'education': ['education', 'school', 'university', 'student', 'learn', 'study'],
            'travel': ['travel', 'vacation', 'trip', 'tourism', 'destination'],
            'finance': ['finance', 'money', 'investment', 'bank', 'loan', 'credit'],
            'sports': ['sports', 'football', 'soccer', 'basketball', 'tennis', 'golf'],
            'entertainment': ['movie', 'film', 'music', 'concert', 'show', 'entertainment'],
        }
        
        interests = []
        for interest, keywords in interest_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                interests.append(interest)
        return interests