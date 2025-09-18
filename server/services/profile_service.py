"""
Profile Service - Business logic layer for WhatsApp Profile Management System
Handles upserts, merging, consent management, field validation, and business rules
"""

import logging
from typing import Optional, List, Dict, Any, Tuple, Set
from datetime import datetime, timedelta
import uuid
import asyncio
import re

from ..repositories.profile_repository import ProfileRepository, QueryOptions
from ..models.profiles import (
    Profile, ProfileCreate, ProfileUpdate, ProfileStatus,
    ProfileHistory, ProfileConsent, ConsentType, ConsentMethod,
    UpsertProfileRequest, MergeProfilesRequest, ProfileResponse,
    ProfileListResponse, ProfileStatsResponse, VersionConflictError
)

logger = logging.getLogger(__name__)

class ProfileValidationError(Exception):
    """Exception for profile validation errors"""
    pass

class ProfileMergeError(Exception):
    """Exception for profile merge errors"""
    pass

class ConsentRequiredError(Exception):
    """Exception when consent is required but not provided"""
    pass

class ProfileService:
    """Service layer for profile management with business logic"""
    
    def __init__(self, repository: Optional[ProfileRepository] = None):
        self.repository = repository or ProfileRepository()
        
        # Essential fields that trigger inquiries
        self.essential_fields = ['name', 'consent']
        
        # Optional fields that enhance the profile
        self.optional_fields = ['display_name', 'persona', 'description', 'timezone', 'language']
        
        # Fields that require consent to store
        self.consent_required_fields = ['name', 'description', 'persona', 'attributes']
    
    # ============================================================================
    # CORE PROFILE OPERATIONS
    # ============================================================================
    
    async def get_or_create_profile(self, phone: str, auto_create: bool = True) -> Optional[Profile]:
        """Get existing profile or create new one if it doesn't exist"""
        try:
            # Normalize phone number
            logger.debug(f"get_or_create_profile called with phone: {phone}")
            normalized_phone = self._normalize_phone(phone)
            logger.debug(f"get_or_create_profile normalized: {phone} -> {normalized_phone}")
            
            # Try to get existing profile
            logger.debug(f"Attempting to fetch existing profile for {normalized_phone}")
            profile = await self.repository.get_profile_by_phone(normalized_phone)
            logger.debug(f"Repository returned profile for {normalized_phone}: {profile is not None}")
            
            if profile:
                # Update last_seen
                try:
                    await self.repository.update_profile(
                        profile.id, 
                        {'last_seen': datetime.utcnow()},
                        reason="Profile accessed"
                    )
                except Exception as e:
                    logger.warning(f"Failed to update last_seen for profile {normalized_phone}: {e}")
                return profile
            
            if not auto_create:
                return None
            
            # Create new profile with minimal data
            logger.debug(f"Attempting to create new profile for {normalized_phone}")
            profile_data = ProfileCreate(
                phone=normalized_phone,
                language='en',
                consent=False,  # Default to no consent
                consent_date=None,
                last_seen=datetime.utcnow(),
                name=None,
                display_name=None,
                persona=None,
                description=None,
                timezone=None,
                attributes={},
                tags=[],
                updated_by="auto_create"
            )
            
            try:
                profile = await self.repository.create_profile(profile_data, "auto_create")
                if profile:
                    logger.info(f"Auto-created profile for {normalized_phone}")
                return profile
            except Exception as e:
                # Handle duplicate key error - another process may have created the profile
                error_msg = str(e).lower()
                if "duplicate key" in error_msg or "unique constraint" in error_msg:
                    logger.info(f"Profile for {normalized_phone} already exists, fetching existing profile")
                    logger.debug(f"Duplicate key error details: {e}")
                    # Try to get the existing profile with enhanced retry logic to handle race conditions
                    retry_count = 0
                    max_retries = 10  # Increased retries
                    base_delay = 0.05  # Reduced base delay for more attempts
                    
                    while retry_count < max_retries:
                        try:
                            # Add a small delay before first attempt to allow for transaction commit
                            if retry_count == 0:
                                logger.debug(f"Waiting {base_delay}s before first retry for {normalized_phone}")
                                await asyncio.sleep(base_delay)
                            
                            logger.debug(f"Attempt {retry_count + 1} to fetch profile for {normalized_phone}")
                            profile = await self.repository.get_profile_by_phone(normalized_phone)
                            if profile:
                                logger.debug(f"Successfully fetched profile for {normalized_phone} on attempt {retry_count + 1}")
                                # Update last_seen
                                try:
                                    await self.repository.update_profile(
                                        profile.id, 
                                        {'last_seen': datetime.utcnow()},
                                        reason="Profile accessed after duplicate key error"
                                    )
                                except Exception as update_e:
                                    logger.warning(f"Failed to update last_seen for profile {normalized_phone}: {update_e}")
                                return profile
                            else:
                                logger.debug(f"Profile not found for {normalized_phone} on attempt {retry_count + 1}")
                                # Wait a bit before retrying to allow for transaction commit
                                # Exponential backoff with jitter
                                delay = base_delay * (2 ** retry_count) + (0.005 * retry_count)
                                logger.debug(f"Waiting {delay}s before next retry for {normalized_phone}")
                                await asyncio.sleep(delay)
                                retry_count += 1
                        except Exception as fetch_e:
                            logger.warning(f"Failed to fetch profile for {normalized_phone} on attempt {retry_count + 1}: {fetch_e}")
                            # Exponential backoff with jitter
                            delay = base_delay * (2 ** retry_count) + (0.005 * retry_count)
                            await asyncio.sleep(delay)
                            retry_count += 1
                
                    # If we still couldn't get the profile after retries, try one final attempt with longer delay
                    logger.warning(f"Failed to fetch profile for {normalized_phone} after {max_retries} retries, trying one final attempt")
                    try:
                        # One final longer wait
                        logger.debug(f"Waiting 1.0s before final attempt for {normalized_phone}")
                        await asyncio.sleep(1.0)
                        logger.debug(f"Final attempt to fetch profile for {normalized_phone}")
                        profile = await self.repository.get_profile_by_phone(normalized_phone)
                        if profile:
                            logger.debug(f"Successfully fetched profile for {normalized_phone} on final attempt")
                            # Update last_seen
                            try:
                                await self.repository.update_profile(
                                    profile.id, 
                                    {'last_seen': datetime.utcnow()},
                                    reason="Profile accessed after final retry"
                                )
                            except Exception as update_e:
                                logger.warning(f"Failed to update last_seen for profile {normalized_phone}: {update_e}")
                            return profile
                    except Exception as final_e:
                        logger.error(f"Final attempt to fetch profile for {normalized_phone} failed: {final_e}")
                    
                    logger.error(f"Failed to fetch profile for {normalized_phone} after all retries")
                else:
                    logger.error(f"Failed to create profile for {phone}: {e}")
                # Return None instead of raising exception to allow app to continue
                return None
            
        except Exception as e:
            logger.error(f"Failed to get or create profile for {phone}: {e}")
            logger.exception(e)  # Log full traceback
            # Return None instead of raising exception to allow app to continue
            return None
    
    async def upsert_profile(self, request: UpsertProfileRequest, actor: str = "system") -> ProfileResponse:
        """Upsert profile with business logic validation"""
        profile = None  # Initialize profile variable
        try:
            # Normalize phone
            normalized_phone = self._normalize_phone(request.phone)
            
            # Get or create profile
            profile = await self.get_or_create_profile(normalized_phone)
            
            # Handle case when profile database is not available
            if profile is None:
                return ProfileResponse(
                    success=False,
                    message="Profile service unavailable - database not accessible",
                    profile=None
                )
            
            # Validate fields
            validated_fields = await self._validate_profile_fields(request.fields, profile)
            
            # Check consent requirements
            await self._check_consent_requirements(validated_fields, profile)
            
            # Update profile
            try:
                updated_profile = await self.repository.update_profile(
                    profile.id,
                    validated_fields,
                    actor,
                    request.expected_version,
                    request.reason
                )
                
                return ProfileResponse(
                    success=True,
                    profile=updated_profile,
                    message="Profile updated successfully"
                )
                
            except ValueError as e:
                if "Version conflict" in str(e):
                    # Handle version conflict
                    current_profile = await self.repository.get_profile_by_id(profile.id)
                    return ProfileResponse(
                        success=False,
                        message="Version conflict - profile was modified by another process",
                        profile=current_profile
                    )
                raise
                
        except ConsentRequiredError as e:
            # profile might not be defined if an exception occurred early
            return ProfileResponse(
                success=False,
                message=f"Consent required: {str(e)}",
                profile=profile if 'profile' in locals() else None
            )
        except ProfileValidationError as e:
            # profile might not be defined if an exception occurred early
            return ProfileResponse(
                success=False,
                message=f"Validation error: {str(e)}",
                profile=profile if 'profile' in locals() else None
            )
        except Exception as e:
            logger.error(f"Failed to upsert profile: {e}")
            return ProfileResponse(
                success=False,
                message=f"Internal error: {str(e)}"
            )
    
    async def merge_profiles(self, request: MergeProfilesRequest, actor: str = "system") -> ProfileResponse:
        """Merge duplicate profiles with conflict resolution"""
        try:
            # Get both profiles
            primary_phone = self._normalize_phone(request.primary_phone)
            duplicate_phone = self._normalize_phone(request.duplicate_phone)
            
            primary_profile = await self.repository.get_profile_by_phone(primary_phone)
            duplicate_profile = await self.repository.get_profile_by_phone(duplicate_phone)
            
            # Handle case when profile database is not available
            if primary_profile is None or duplicate_profile is None:
                return ProfileResponse(
                    success=False,
                    message="Profile service unavailable - database not accessible",
                    profile=None
                )
            
            if not primary_profile:
                raise ProfileMergeError(f"Primary profile not found: {primary_phone}")
            if not duplicate_profile:
                raise ProfileMergeError(f"Duplicate profile not found: {duplicate_phone}")
            
            if primary_profile.id == duplicate_profile.id:
                raise ProfileMergeError("Cannot merge profile with itself")
            
            # Merge the profiles
            merged_profile = await self._merge_profile_data(
                primary_profile, duplicate_profile, request.merge_strategy
            )
            
            # Update primary profile
            updated_primary = await self.repository.update_profile(
                primary_profile.id,
                merged_profile,
                actor,
                reason=f"Merged with {duplicate_phone}: {request.reason}"
            )
            
            # Mark duplicate as merged
            await self.repository.update_profile(
                duplicate_profile.id,
                {
                    'status': ProfileStatus.MERGED.value,
                    'merged_into': primary_profile.id
                },
                actor,
                reason=f"Merged into {primary_phone}: {request.reason}"
            )
            
            logger.info(f"Merged profiles: {duplicate_phone} -> {primary_phone}")
            
            return ProfileResponse(
                success=True,
                profile=updated_primary,
                message=f"Successfully merged {duplicate_phone} into {primary_phone}"
            )
            
        except Exception as e:
            logger.error(f"Failed to merge profiles: {e}")
            return ProfileResponse(
                success=False,
                message=f"Merge failed: {str(e)}"
            )
    
    # ============================================================================
    # CONSENT MANAGEMENT
    # ============================================================================
    
    async def update_consent(
        self, 
        phone: str, 
        consent_type: ConsentType, 
        granted: bool,
        method: ConsentMethod = ConsentMethod.API_CALL,
        actor: str = "system"
    ) -> ProfileResponse:
        """Update user consent with proper tracking"""
        try:
            profile = await self.get_or_create_profile(phone)
            
            # Handle case when profile database is not available
            if profile is None:
                return ProfileResponse(
                    success=False,
                    message="Profile service unavailable - database not accessible",
                    profile=None
                )
            
            # Create consent record
            consent_data = {
                'profile_id': profile.id,
                'consent_type': consent_type,
                'granted': granted,
                'consent_method': method
            }
            
            # Update profile consent if it's memory storage consent
            if consent_type == ConsentType.MEMORY_STORAGE:
                await self.repository.update_profile(
                    profile.id,
                    {
                        'consent': granted,
                        'consent_date': datetime.utcnow()
                    },
                    actor,
                    reason=f"Consent {'granted' if granted else 'revoked'} via {method.value}"
                )
            
            updated_profile = await self.repository.get_profile_by_id(profile.id)
            
            return ProfileResponse(
                success=True,
                profile=updated_profile,
                message=f"Consent {'granted' if granted else 'revoked'} successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to update consent for {phone}: {e}")
            return ProfileResponse(
                success=False,
                message=f"Failed to update consent: {str(e)}"
            )
    
    async def check_consent_required(self, profile: Profile, fields: Dict[str, Any]) -> List[str]:
        """Check which fields require consent that hasn't been granted"""
        # Handle case when profile is None (database not available)
        if profile is None:
            return []
        
        if profile.consent:
            return []  # Consent already granted
        
        required_consent_fields = []
        for field_name in fields.keys():
            if field_name in self.consent_required_fields:
                required_consent_fields.append(field_name)
        
        return required_consent_fields
    
    # ============================================================================
    # FIELD INQUIRY SYSTEM
    # ============================================================================
    
    async def get_missing_fields(self, phone: str) -> Dict[str, Any]:
        """Get missing essential and optional fields for a profile"""
        try:
            profile = await self.get_or_create_profile(phone)
            
            # Handle case when profile database is not available
            if profile is None:
                return {'essential': [], 'optional': [], 'total_missing': 0}
            
            missing_essential = []
            missing_optional = []
            
            # Check essential fields
            for field in self.essential_fields:
                if not self._has_field_value(profile, field):
                    missing_essential.append(field)
            
            # Check optional fields (only if consent is granted)
            if profile.consent:
                for field in self.optional_fields:
                    if not self._has_field_value(profile, field):
                        missing_optional.append(field)
            
            return {
                'essential': missing_essential,
                'optional': missing_optional,
                'total_missing': len(missing_essential) + len(missing_optional)
            }
            
        except Exception as e:
            logger.error(f"Failed to get missing fields for {phone}: {e}")
            return {'essential': [], 'optional': [], 'total_missing': 0}
    
    async def should_inquire_field(self, profile: Profile, field_name: str) -> bool:
        """Determine if we should inquire about a missing field"""
        # Handle case when profile is None (database not available)
        if profile is None:
            return False
        
        # Don't inquire if field already has value
        if self._has_field_value(profile, field_name):
            return False
        
        # Don't inquire about consent-requiring fields if no consent
        if field_name in self.consent_required_fields and not profile.consent:
            return False
        
        # Check if we've recently inquired about this field
        # Skip this check if profile.id is None (shouldn't happen but just in case)
        if profile.id is not None:
            recent_inquiries = await self._get_recent_inquiries(profile.id, field_name)
            if recent_inquiries:
                return False
        
        # Check field priority
        if field_name in self.essential_fields:
            return True
        
        if field_name in self.optional_fields and profile.consent:
            return True
        
        return False

    # ============================================================================
    # PROFILE ANALYSIS AND ENHANCEMENT
    # ============================================================================
    
    async def analyze_profile_completeness(self, profile: Profile) -> Dict[str, Any]:
        """Analyze how complete a profile is"""
        # Handle case when profile is None (database not available)
        if profile is None:
            return {'overall_completeness': 0, 'quality_score': 0}
        
        try:
            total_fields = len(self.essential_fields) + len(self.optional_fields)
            filled_fields = 0
            
            # Count filled essential fields
            filled_essential = 0
            for field in self.essential_fields:
                if self._has_field_value(profile, field):
                    filled_essential += 1
                    filled_fields += 1
            
            # Count filled optional fields (only if consent granted)
            filled_optional = 0
            if profile.consent:
                for field in self.optional_fields:
                    if self._has_field_value(profile, field):
                        filled_optional += 1
                        filled_fields += 1
            
            essential_completeness = filled_essential / len(self.essential_fields)
            overall_completeness = filled_fields / total_fields
            
            return {
                'overall_completeness': round(overall_completeness * 100, 1),
                'essential_completeness': round(essential_completeness * 100, 1),
                'filled_essential': filled_essential,
                'total_essential': len(self.essential_fields),
                'filled_optional': filled_optional,
                'total_optional': len(self.optional_fields),
                'has_consent': profile.consent,
                'quality_score': self._calculate_quality_score(profile)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze profile completeness: {e}")
            return {'overall_completeness': 0, 'quality_score': 0}
    
    async def suggest_profile_improvements(self, profile: Profile) -> List[Dict[str, str]]:
        """Suggest improvements for a profile"""
        # Handle case when profile is None (database not available)
        if profile is None:
            return []
        
        suggestions = []
        
        try:
            # Missing essential fields
            for field in self.essential_fields:
                if not self._has_field_value(profile, field):
                    suggestions.append({
                        'type': 'missing_essential',
                        'field': field,
                        'priority': 'high',
                        'suggestion': f"Add {field} to improve profile completeness"
                    })
            
            # Missing optional fields (if consent granted)
            if profile.consent:
                for field in self.optional_fields:
                    if not self._has_field_value(profile, field):
                        suggestions.append({
                            'type': 'missing_optional',
                            'field': field,
                            'priority': 'medium',
                            'suggestion': f"Consider adding {field} for better personalization"
                        })
            
            # Consent suggestions
            if not profile.consent:
                suggestions.append({
                    'type': 'consent',
                    'field': 'consent',
                    'priority': 'high',
                    'suggestion': 'Request consent to enable memory storage and personalization'
                })
            
            # Data quality suggestions
            if profile.description and len(profile.description) < 50:
                suggestions.append({
                    'type': 'quality',
                    'field': 'description',
                    'priority': 'low',
                    'suggestion': 'Description could be more detailed'
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to suggest profile improvements: {e}")
            return []

    # ============================================================================
    # SEARCH AND FILTERING
    # ============================================================================
    
    async def search_profiles(
        self, 
        query: Optional[str] = None, 
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        per_page: int = 50
    ) -> ProfileListResponse:
        """Search profiles with advanced filtering"""
        try:
            options = QueryOptions(
                limit=per_page,
                offset=(page - 1) * per_page,
                include_deleted=filters.get('include_deleted', False) if filters else False
            )
            
            if query:
                # Text search
                profiles = await self.repository.search_profiles(query, options=options)
                total = len(profiles)  # Approximate for search results
            else:
                # List with filters
                profiles, total = await self.repository.list_profiles(options)
            
            # Apply additional filters
            if filters:
                profiles = self._apply_filters(profiles, filters)
            
            return ProfileListResponse(
                success=True,
                profiles=profiles,
                total=total,
                page=page,
                per_page=per_page
            )
            
        except Exception as e:
            logger.error(f"Failed to search profiles: {e}")
            return ProfileListResponse(
                success=False,
                message=f"Search failed: {str(e)}"
            )
    
    async def get_profile_statistics(self) -> ProfileStatsResponse:
        """Get comprehensive profile statistics"""
        try:
            stats = await self.repository.get_profile_stats()
            
            return ProfileStatsResponse(
                total_profiles=stats.get('total_profiles', 0),
                active_profiles=stats.get('active_profiles', 0),
                consented_profiles=stats.get('consented_profiles', 0),
                recent_interactions=stats.get('recent_interactions', 0),
                top_personas=stats.get('top_personas', []),
                top_languages=stats.get('top_languages', [])
            )
            
        except Exception as e:
            logger.error(f"Failed to get profile statistics: {e}")
            return ProfileStatsResponse()
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number to E.164 format"""
        if not phone:
            raise ProfileValidationError("Phone number is required")
        
        # Remove all non-digit characters except +
        cleaned = re.sub(r'[^\d+]', '', phone)
        
        # Add + if missing and starts with country code
        if not cleaned.startswith('+') and len(cleaned) > 10:
            cleaned = '+' + cleaned
        
        # Validate format
        if not re.match(r'^\+?[1-9]\d{6,14}$', cleaned):
            raise ProfileValidationError(f"Invalid phone number format: {phone}")
        
        logger.debug(f"Normalized phone number: {phone} -> {cleaned}")
        return cleaned
    
    async def _validate_profile_fields(self, fields: Dict[str, Any], profile: Profile) -> Dict[str, Any]:
        """Validate profile fields before updating"""
        validated = {}
        
        for field_name, value in fields.items():
            # Skip None values
            if value is None:
                continue
            
            # Validate specific fields
            if field_name == 'phone':
                validated[field_name] = self._normalize_phone(value)
            elif field_name == 'language':
                if not re.match(r'^[a-z]{2}(-[A-Z]{2})?$', value):
                    raise ProfileValidationError(f"Invalid language code: {value}")
                validated[field_name] = value.lower()
            elif field_name == 'tags':
                if isinstance(value, list) and len(value) <= 20:
                    validated[field_name] = [tag.strip().lower() for tag in value if tag.strip()]
                else:
                    raise ProfileValidationError("Tags must be a list with max 20 items")
            elif field_name == 'attributes':
                if isinstance(value, dict) and len(value) <= 50:
                    validated[field_name] = value
                else:
                    raise ProfileValidationError("Attributes must be a dict with max 50 items")
            else:
                # Generic validation for string fields
                if isinstance(value, str):
                    validated[field_name] = value.strip()
                else:
                    validated[field_name] = value
        
        return validated
    
    async def _check_consent_requirements(self, fields: Dict[str, Any], profile: Profile) -> None:
        """Check if consent is required for the fields being updated"""
        # Handle case when profile is None (database not available)
        if profile is None:
            return  # Skip consent check if database is not available
        
        if profile.consent:
            return  # Consent already granted
        
        consent_required_fields = []
        for field_name in fields.keys():
            if field_name in self.consent_required_fields:
                consent_required_fields.append(field_name)
        
        if consent_required_fields:
            raise ConsentRequiredError(
                f"Consent required to update fields: {', '.join(consent_required_fields)}"
            )
    
    async def _merge_profile_data(
        self, 
        primary: Profile, 
        duplicate: Profile, 
        strategy: str = "prefer_primary"
    ) -> Dict[str, Any]:
        """Merge data from duplicate profile into primary"""
        merged = {}
        
        # Define merge rules for each field
        merge_rules = {
            'prefer_primary': lambda p, d: p if p else d,
            'prefer_duplicate': lambda p, d: d if d else p,
            'prefer_longer': lambda p, d: p if (p and len(str(p)) > len(str(d))) else d,
            'combine': lambda p, d: f"{p}; {d}" if (p and d and p != d) else (p or d)
        }
        
        rule_func = merge_rules.get(strategy, merge_rules['prefer_primary'])
        
        # Merge basic fields
        fields_to_merge = ['name', 'display_name', 'persona', 'description', 'timezone']
        for field in fields_to_merge:
            primary_val = getattr(primary, field, None)
            duplicate_val = getattr(duplicate, field, None)
            merged_val = rule_func(primary_val, duplicate_val)
            if merged_val:
                merged[field] = merged_val
        
        # Merge attributes (combine dictionaries)
        if primary.attributes or duplicate.attributes:
            merged_attrs = primary.attributes.copy()
            merged_attrs.update(duplicate.attributes)
            merged['attributes'] = merged_attrs
        
        # Merge tags (combine lists)
        if primary.tags or duplicate.tags:
            merged_tags = list(set(primary.tags + duplicate.tags))
            merged['tags'] = merged_tags
        
        # Keep most recent last_seen
        if duplicate.last_seen and (not primary.last_seen or duplicate.last_seen > primary.last_seen):
            merged['last_seen'] = duplicate.last_seen
        
        # Prefer granted consent
        if duplicate.consent and not primary.consent:
            merged['consent'] = True
            merged['consent_date'] = duplicate.consent_date
        
        return merged
    
    def _has_field_value(self, profile: Profile, field_name: str) -> bool:
        """Check if profile has a meaningful value for the field"""
        value = getattr(profile, field_name, None)
        
        if value is None:
            return False
        
        if isinstance(value, str):
            return bool(value.strip())
        
        if isinstance(value, (list, dict)):
            return len(value) > 0
        
        if isinstance(value, bool):
            return True  # Boolean fields are considered filled regardless of value
        
        return value is not None
    
    async def _get_recent_inquiries(self, profile_id: uuid.UUID, field_name: str) -> List[Any]:
        """Get recent inquiries for a specific field (placeholder)"""
        # This would query a separate inquiries table
        # For now, return empty list
        return []
    
    def _calculate_quality_score(self, profile: Profile) -> float:
        """Calculate profile quality score (0-100)"""
        score = 0.0
        
        # Essential fields (60 points total)
        essential_weight = 60.0 / len(self.essential_fields)
        for field in self.essential_fields:
            if self._has_field_value(profile, field):
                score += essential_weight
        
        # Optional fields (30 points total)
        if profile.consent:
            optional_weight = 30.0 / len(self.optional_fields)
            for field in self.optional_fields:
                if self._has_field_value(profile, field):
                    score += optional_weight
        
        # Bonus points (10 points total)
        if profile.attributes and len(profile.attributes) > 0:
            score += 5.0
        
        if profile.tags and len(profile.tags) > 0:
            score += 5.0
        
        return round(min(score, 100.0), 1)
    
    def _apply_filters(self, profiles: List[Profile], filters: Dict[str, Any]) -> List[Profile]:
        """Apply additional filters to profile list"""
        filtered = profiles
        
        if 'persona' in filters:
            filtered = [p for p in filtered if p.persona == filters['persona']]
        
        if 'consent' in filters:
            filtered = [p for p in filtered if p.consent == filters['consent']]
        
        if 'language' in filters:
            filtered = [p for p in filtered if p.language == filters['language']]
        
        if 'has_name' in filters:
            filtered = [p for p in filtered if bool(p.name) == filters['has_name']]
        
        return filtered

    async def delete_profile(self, phone: str, deleted_by: str = "system", reason: str = "Profile deleted", hard_delete: bool = False) -> bool:
        """Delete profile by phone number"""
        try:
            # Get the profile first
            profile = await self.get_or_create_profile(phone, auto_create=False)
            
            if not profile:
                return False
            
            # Delete using repository
            success = await self.repository.delete_profile(profile.id, deleted_by, reason, hard_delete)
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete profile for {phone}: {e}")
            return False