"""
Enhanced Contact Database Tools - Comprehensive tools for contact information management
Provides advanced CRUD operations, relationship management, and analytics for contacts
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
import uuid
import json

from .profile_service import ProfileService
from ..repositories.profile_repository import QueryOptions
from ..models.profiles import (
    Profile, ProfileCreate, ProfileUpdate, UpsertProfileRequest,
    ProfileRelationship, ProfileRelationshipCreate,
    ConsentType, ConsentMethod
)
from .audit_logger import audit_logger, ChangeType

logger = logging.getLogger(__name__)

class ContactDatabaseTools:
    """Enhanced tools for contact information management"""
    
    def __init__(self, profile_service: ProfileService = None, db_manager=None):
        self.profile_service = profile_service or ProfileService()
        self.db_manager = db_manager
        # Initialize audit logger with db_manager
        self.audit_logger = audit_logger
        if db_manager:
            self.audit_logger.db_manager = db_manager
    
    async def get_contact_by_phone(self, phone: str, include_deleted: bool = False) -> Optional[Profile]:
        """Get contact profile by phone number"""
        try:
            profile = await self.profile_service.get_or_create_profile(phone, auto_create=False)
            return profile
        except Exception as e:
            logger.error(f"Error getting contact by phone {phone}: {e}")
            return None
    
    async def create_contact(self, phone: str, name: str = None, created_by: str = "system", reason: str = "Contact created", **kwargs) -> Optional[Profile]:
        """Create a new contact profile"""
        try:
            # Normalize phone number
            normalized_phone = self._normalize_phone(phone)
            
            # Prepare profile data
            profile_data = ProfileCreate(
                phone=normalized_phone,
                name=name,
                language=kwargs.get('language', 'en'),
                timezone=kwargs.get('timezone'),
                persona=kwargs.get('persona'),
                display_name=kwargs.get('display_name', name),
                description=kwargs.get('description'),
                consent=kwargs.get('consent', False),
                attributes=kwargs.get('attributes', {}),
                tags=kwargs.get('tags', []),
                updated_by=created_by
            )
            
            # Create profile
            profile = await self.profile_service.repository.create_profile(profile_data, created_by)
            logger.info(f"Created contact profile for {normalized_phone}")
            
            # Log the creation in audit trail
            try:
                await self.audit_logger.log_change(
                    user_id=created_by,
                    change_type=ChangeType.CREATE,
                    table_name="profiles",
                    record_id=str(profile.id),
                    old_values={},
                    new_values=profile.dict(),
                    reason=reason
                )
            except Exception as audit_error:
                logger.warning(f"Failed to log audit entry for contact creation: {audit_error}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error creating contact for {phone}: {e}")
            return None
    
    async def update_contact(self, phone: str, fields: Dict[str, Any], updated_by: str = "system", reason: str = "Contact updated") -> Optional[Profile]:
        """Update contact profile with specified fields"""
        try:
            # Normalize phone number
            normalized_phone = self._normalize_phone(phone)
            
            # Get existing profile
            profile = await self.profile_service.get_or_create_profile(normalized_phone, auto_create=False)
            if not profile:
                logger.warning(f"Contact not found for {normalized_phone}")
                return None
            
            # Store old values for audit trail
            old_values = profile.dict()
            
            # Prepare update request
            update_request = UpsertProfileRequest(
                phone=normalized_phone,
                fields=fields,
                reason=reason,
                expected_version=None,
                session_id=None
            )
            
            # Update profile
            response = await self.profile_service.upsert_profile(update_request, updated_by)
            if response.success:
                logger.info(f"Updated contact profile for {normalized_phone}")
                
                # Log the update in audit trail
                try:
                    new_profile = await self.profile_service.get_or_create_profile(normalized_phone, auto_create=False)
                    if new_profile:
                        await self.audit_logger.log_change(
                            user_id=updated_by,
                            change_type=ChangeType.UPDATE,
                            table_name="profiles",
                            record_id=str(profile.id),
                            old_values=old_values,
                            new_values=new_profile.dict(),
                            reason=reason
                        )
                except Exception as audit_error:
                    logger.warning(f"Failed to log audit entry for contact update: {audit_error}")
                
                return response.profile
            else:
                logger.error(f"Failed to update contact {normalized_phone}: {response.message}")
                return None
                
        except Exception as e:
            logger.error(f"Error updating contact {phone}: {e}")
            return None
    
    async def delete_contact(self, phone: str, deleted_by: str = "system", hard_delete: bool = False, reason: str = "Contact deleted") -> bool:
        """Delete contact profile (soft delete by default)"""
        try:
            # Normalize phone number
            normalized_phone = self._normalize_phone(phone)
            
            # Get existing profile
            profile = await self.profile_service.get_or_create_profile(normalized_phone, auto_create=False)
            if not profile:
                logger.warning(f"Contact not found for {normalized_phone}")
                return False
            
            # Store old values for audit trail
            old_values = profile.dict()
            
            # Delete profile
            success = await self.profile_service.repository.delete_profile(
                profile.id, 
                deleted_by=deleted_by, 
                reason=reason, 
                hard_delete=hard_delete
            )
            
            if success:
                logger.info(f"{'Hard deleted' if hard_delete else 'Soft deleted'} contact profile for {normalized_phone}")
                
                # Log the deletion in audit trail
                try:
                    await self.audit_logger.log_change(
                        user_id=deleted_by,
                        change_type=ChangeType.DELETE if hard_delete else ChangeType.UPDATE,
                        table_name="profiles",
                        record_id=str(profile.id),
                        old_values=old_values,
                        new_values={} if hard_delete else {**old_values, "status": "deleted"},
                        reason=reason
                    )
                except Exception as audit_error:
                    logger.warning(f"Failed to log audit entry for contact deletion: {audit_error}")
                
            else:
                logger.warning(f"Failed to delete contact {normalized_phone}")
                
            return success
                
        except Exception as e:
            logger.error(f"Error deleting contact {phone}: {e}")
            return False
    
    async def search_contacts(self, query: str, fields: List[str] = None, limit: int = 50) -> List[Profile]:
        """Search contacts by text in specified fields"""
        try:
            options = QueryOptions(limit=limit, include_deleted=False)
            profiles = await self.profile_service.repository.search_profiles(query, fields, options)
            logger.info(f"Found {len(profiles)} contacts matching query: {query}")
            return profiles
        except Exception as e:
            logger.error(f"Error searching contacts: {e}")
            return []
    
    async def get_contacts_by_tag(self, tag: str, limit: int = 100) -> List[Profile]:
        """Get all contacts with a specific tag"""
        try:
            # This would require a custom query in the repository
            # For now, we'll do a simple approach
            options = QueryOptions(limit=limit, include_deleted=False)
            profiles, _ = await self.profile_service.repository.list_profiles(options)
            
            # Filter by tag
            tagged_profiles = [p for p in profiles if p.tags and tag.lower() in [t.lower() for t in p.tags]]
            logger.info(f"Found {len(tagged_profiles)} contacts with tag: {tag}")
            return tagged_profiles
        except Exception as e:
            logger.error(f"Error getting contacts by tag {tag}: {e}")
            return []
    
    async def get_contacts_by_persona(self, persona: str, limit: int = 100) -> List[Profile]:
        """Get all contacts with a specific persona"""
        try:
            options = QueryOptions(limit=limit, include_deleted=False)
            profiles = await self.profile_service.repository.get_profiles_by_persona(persona, options)
            logger.info(f"Found {len(profiles)} contacts with persona: {persona}")
            return profiles
        except Exception as e:
            logger.error(f"Error getting contacts by persona {persona}: {e}")
            return []
    
    async def get_vip_contacts(self, limit: int = 100) -> List[Profile]:
        """Get all VIP contacts"""
        try:
            return await self.get_contacts_by_tag("vip", limit)
        except Exception as e:
            logger.error(f"Error getting VIP contacts: {e}")
            return []
    
    async def add_relationship(self, primary_phone: str, related_phone: str, relationship_type: str, strength: float = 0.5, notes: str = None, updated_by: str = "system") -> Optional[ProfileRelationship]:
        """Add a relationship between two contacts"""
        try:
            # Normalize phone numbers
            primary_normalized = self._normalize_phone(primary_phone)
            related_normalized = self._normalize_phone(related_phone)
            
            # Get both profiles
            primary_profile = await self.profile_service.get_or_create_profile(primary_normalized, auto_create=False)
            related_profile = await self.profile_service.get_or_create_profile(related_normalized, auto_create=False)
            
            if not primary_profile or not related_profile:
                logger.warning(f"One or both contacts not found: {primary_phone}, {related_phone}")
                return None
            
            if primary_profile.id == related_profile.id:
                logger.warning("Cannot create relationship with self")
                return None
            
            # Check if relationship already exists
            # This would require a custom repository method
            # For now, we'll assume it doesn't exist
            
            # Create relationship data
            relationship_data = ProfileRelationshipCreate(
                primary_profile_id=primary_profile.id,
                related_profile_id=related_profile.id,
                relationship_type=relationship_type,
                strength=min(max(strength, 0.0), 1.0),  # Clamp to 0-1 range
                notes=notes,
                created_by=updated_by
            )
            
            # This would require a custom repository method to insert the relationship
            # For now, we'll just log the intent
            logger.info(f"Would create relationship: {primary_phone} -> {related_phone} ({relationship_type})")
            return None  # Return None for now since we don't have the full implementation
            
        except Exception as e:
            logger.error(f"Error adding relationship between {primary_phone} and {related_phone}: {e}")
            return None
    
    async def get_contact_relationships(self, phone: str) -> List[ProfileRelationship]:
        """Get all relationships for a contact"""
        try:
            # Normalize phone number
            normalized_phone = self._normalize_phone(phone)
            
            # Get profile
            profile = await self.profile_service.get_or_create_profile(normalized_phone, auto_create=False)
            if not profile:
                logger.warning(f"Contact not found for {normalized_phone}")
                return []
            
            # This would require a custom repository method to fetch relationships
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Error getting relationships for {phone}: {e}")
            return []
    
    async def update_contact_tags(self, phone: str, tags: List[str], updated_by: str = "system", append: bool = True) -> Optional[Profile]:
        """Update contact tags"""
        try:
            # Normalize phone number
            normalized_phone = self._normalize_phone(phone)
            
            # Get profile
            profile = await self.profile_service.get_or_create_profile(normalized_phone, auto_create=False)
            if not profile:
                logger.warning(f"Contact not found for {normalized_phone}")
                return None
            
            # Prepare tags
            if append:
                # Merge with existing tags, removing duplicates
                new_tags = list(set(profile.tags + tags))
            else:
                new_tags = tags
            
            # Update profile
            return await self.update_contact(normalized_phone, {"tags": new_tags}, updated_by, "Tags updated")
            
        except Exception as e:
            logger.error(f"Error updating tags for {phone}: {e}")
            return None
    
    async def get_contact_statistics(self) -> Dict[str, Any]:
        """Get comprehensive contact statistics"""
        try:
            stats = await self.profile_service.repository.get_profile_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting contact statistics: {e}")
            return {}
    
    async def merge_contacts(self, primary_phone: str, duplicate_phone: str, merged_by: str = "system", merge_strategy: str = "prefer_primary") -> bool:
        """Merge two contact profiles"""
        try:
            # This would use the existing merge functionality in ProfileService
            from ..models.profiles import MergeProfilesRequest
            
            merge_request = MergeProfilesRequest(
                primary_phone=primary_phone,
                duplicate_phone=duplicate_phone,
                merge_strategy=merge_strategy,
                reason="Contacts merged via ContactDatabaseTools"
            )
            
            response = await self.profile_service.merge_profiles(merge_request, merged_by)
            
            # Log the merge in audit trail
            if response.success:
                try:
                    await self.audit_logger.log_change(
                        user_id=merged_by,
                        change_type=ChangeType.MERGE,
                        table_name="profiles",
                        record_id="merge_operation",
                        old_values={"primary_phone": primary_phone, "duplicate_phone": duplicate_phone},
                        new_values={"merged_profile_id": str(response.profile.id) if response.profile else "unknown"},
                        reason="Contacts merged"
                    )
                except Exception as audit_error:
                    logger.warning(f"Failed to log audit entry for contact merge: {audit_error}")
            
            return response.success
            
        except Exception as e:
            logger.error(f"Error merging contacts {primary_phone} and {duplicate_phone}: {e}")
            return False
    
    async def get_recent_contacts(self, days: int = 30, limit: int = 50) -> List[Profile]:
        """Get contacts that have been recently active"""
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # This would require a custom query in the repository
            # For now, we'll get all contacts and filter by last_seen
            options = QueryOptions(limit=limit, include_deleted=False)
            profiles, _ = await self.profile_service.repository.list_profiles(options)
            
            # Filter by recent activity
            recent_profiles = [
                p for p in profiles 
                if p.last_seen and p.last_seen >= cutoff_date
            ]
            
            # Sort by last_seen (most recent first)
            recent_profiles.sort(key=lambda p: p.last_seen or datetime.min, reverse=True)
            
            logger.info(f"Found {len(recent_profiles)} recently active contacts")
            return recent_profiles[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent contacts: {e}")
            return []
    
    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number to E.164 format"""
        if not phone:
            return phone
            
        # Remove all non-digit characters except +
        import re
        cleaned = re.sub(r'[^\d+]', '', phone)
        
        # Add + if missing and starts with country code
        if not cleaned.startswith('+') and len(cleaned) > 10:
            cleaned = '+' + cleaned
            
        return cleaned