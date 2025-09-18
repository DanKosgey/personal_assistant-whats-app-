"""
Profile Repository - Data access layer for WhatsApp Profile Management System
Implements CRUD operations, versioning, audit logging, and concurrency handling
"""

import asyncio
import json
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass

from ..db.profile_db import ProfileDatabase, get_profile_db
from ..models.profiles import (
    Profile, ProfileCreate, ProfileUpdate, ProfileStatus, ChangeType,
    ProfileHistory, ProfileHistoryCreate,
    ProfileSummary, ProfileSummaryCreate,
    ProfileRelationship, ProfileRelationshipCreate,
    ProfileConsent, ProfileConsentCreate,
    ConsentType
)

logger = logging.getLogger(__name__)

# Custom JSON encoder for datetime objects
def json_serializer(obj):
    """Custom JSON serializer for datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

@dataclass
class QueryOptions:
    """Options for database queries"""
    limit: int = 50
    offset: int = 0
    include_deleted: bool = False
    include_merged: bool = False
    order_by: str = "updated_at"
    order_desc: bool = True

class ProfileRepository:
    """Repository for profile management with comprehensive CRUD operations"""
    
    def __init__(self, db: Optional[ProfileDatabase] = None):
        self.db = db
        self._db_initialized = False
    
    async def _ensure_db(self):
        """Ensure database is initialized"""
        if not self._db_initialized:
            if self.db is None:
                self.db = await get_profile_db()
            self._db_initialized = True
            # Check if database is healthy
            if self.db and self.db.pool:
                try:
                    health = await self.db.health_check()
                    logger.debug(f"Database health check: {health}")
                    if health.get('status') != 'healthy':
                        logger.warning(f"Database not healthy: {health}")
                except Exception as e:
                    logger.warning(f"Database health check failed: {e}")
    
    # ============================================================================
    # PROFILE CRUD OPERATIONS
    # ============================================================================
    
    async def create_profile(self, profile_data: ProfileCreate, created_by: str = "system") -> Optional[Profile]:
        """Create a new profile with audit logging"""
        await self._ensure_db()
        try:
            if not self.db or not self.db.pool:
                logger.warning("Profile database not available")
                return None
                
            # Generate profile ID
            profile_id = uuid.uuid4()
            current_time = datetime.now().astimezone()
            
            # Prepare profile data
            profile_dict = profile_data.dict()
            logger.debug(f"Creating profile with phone: {profile_dict['phone']}")
            profile_dict.update({
                'id': profile_id,
                'created_at': current_time,
                'updated_at': current_time,
                'version': 1,
                'status': ProfileStatus.ACTIVE.value,
                'updated_by': created_by
            })
            
            # Convert tags list to PostgreSQL array format
            tags_array = profile_dict.pop('tags', [])
            attributes_json = json.dumps(profile_dict.pop('attributes', {}), default=json_serializer)
            
            async with self.db.pool.acquire() as conn:
                async with conn.transaction():
                    # Insert profile
                    query = """
                        INSERT INTO profiles (
                            id, phone, name, display_name, persona, description,
                            language, timezone, consent, consent_date, last_seen,
                            created_at, updated_at, version, attributes, tags,
                            updated_by, status
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18
                        ) RETURNING *
                    """
                    
                    row = await conn.fetchrow(
                        query,
                        profile_id, profile_dict['phone'], profile_dict.get('name'),
                        profile_dict.get('display_name'), profile_dict.get('persona'),
                        profile_dict.get('description'), profile_dict['language'],
                        profile_dict.get('timezone'), profile_dict['consent'],
                        profile_dict.get('consent_date'), profile_dict.get('last_seen'),
                        current_time, current_time, 1, attributes_json, tags_array,
                        created_by, ProfileStatus.ACTIVE.value
                    )
                    
                    # Create audit log entry
                    await self._create_audit_entry(
                        conn, profile_id, created_by, ChangeType.CREATE,
                        {"profile": profile_dict}, "Profile created"
                    )
                    
                    # Convert row to Profile model
                    profile = self._row_to_profile(row)
                    
                    logger.info(f"Created profile {profile_id} for phone {profile_dict['phone']}")
                    return profile
                    
        except Exception as e:
            # Log the specific error but don't fail completely
            error_msg = str(e).lower()
            if "duplicate key" in error_msg or "unique constraint" in error_msg:
                logger.info(f"Profile for phone {profile_data.phone} already exists, will fetch existing profile")
                logger.debug(f"Duplicate key error details: {e}")
                # Re-raise the exception so the caller can handle it properly
                raise
            else:
                logger.error(f"Failed to create profile: {e}")
                return None  # Return None instead of raising exception

    async def get_profile_by_phone(self, phone: str, include_deleted: bool = False) -> Optional[Profile]:
        """Get profile by phone number"""
        await self._ensure_db()
        try:
            if not self.db or not self.db.pool:
                logger.warning("Profile database not available")
                return None
            
            # Log the exact phone number we're searching for, including any hidden characters
            logger.debug(f"Searching for profile with phone: '{phone}' (length: {len(phone)})")
            logger.debug(f"Phone repr: {repr(phone)}")
            for i, char in enumerate(phone):
                logger.debug(f"  char[{i}]: '{char}' (ord: {ord(char)})")
                
            where_clause = "WHERE phone = $1"
            if not include_deleted:
                where_clause += " AND status != 'deleted'"
        
            query = f"""
                SELECT * FROM profiles
                {where_clause}
                ORDER BY created_at DESC
                LIMIT 1
            """
        
            logger.debug(f"Fetching profile by phone: {phone}")
            logger.debug(f"Query: {query}")
            logger.debug(f"Query params: {phone}")
        
            # First, let's see what's actually in the database
            all_profiles_query = "SELECT phone, status, id, created_at FROM profiles ORDER BY created_at DESC LIMIT 20"
            all_profiles = await self.db.fetch(all_profiles_query) if self.db else []
            logger.debug(f"Recent profiles in database (count: {len(all_profiles)}):")
            for p in all_profiles:  # Show all recent profiles
                logger.debug(f"  phone='{p['phone']}' (status={p['status']}, id={p['id']}, created_at={p['created_at']})")
        
            # Also check specifically for the phone number we're looking for
            specific_check_query = "SELECT phone, status, id, created_at FROM profiles WHERE phone = $1"
            specific_profiles = await self.db.fetch(specific_check_query, phone) if self.db else []
            logger.debug(f"Profiles matching phone '{phone}' (count: {len(specific_profiles)}):")
            for p in specific_profiles:
                logger.debug(f"  phone='{p['phone']}' (status={p['status']}, id={p['id']}, created_at={p['created_at']})")
            
            # Check for similar phone numbers that might have hidden characters
            fuzzy_check_query = "SELECT phone, status, id, created_at FROM profiles WHERE phone LIKE $1"
            fuzzy_profiles = await self.db.fetch(fuzzy_check_query, f"%{phone}%") if self.db else []
            logger.debug(f"Profiles fuzzy matching phone '{phone}' (count: {len(fuzzy_profiles)}):")
            for p in fuzzy_profiles:
                logger.debug(f"  phone='{p['phone']}' (status={p['status']}, id={p['id']}, created_at={p['created_at']})")
        
            # Check for phone numbers that are numerically similar but might have different formatting
            numeric_phone = ''.join(filter(str.isdigit, phone))
            if numeric_phone and len(numeric_phone) >= 10:
                numeric_check_query = """
                    SELECT phone, status, id, created_at 
                    FROM profiles 
                    WHERE REPLACE(REPLACE(REPLACE(phone, '+', ''), '-', ''), ' ', '') = $1
                """
                numeric_profiles = await self.db.fetch(numeric_check_query, numeric_phone) if self.db else []
                logger.debug(f"Profiles matching numeric phone '{numeric_phone}' (count: {len(numeric_profiles)}):")
                for p in numeric_profiles:
                    logger.debug(f"  phone='{p['phone']}' (status={p['status']}, id={p['id']}, created_at={p['created_at']})")
        
            row = await self.db.fetchrow(query, phone) if self.db else None
            logger.debug(f"Profile fetch result for '{phone}': {row is not None}")
            if row:
                logger.debug(f"Found profile: phone='{row['phone']}', status={row['status']}, id={row['id']}")
                
                # Double-check the row data
                logger.debug(f"Full row data: {dict(row)}")
            else:
                logger.debug(f"No profile found for phone '{phone}'")
                
                # Try a more permissive query to see if there's any profile with this phone
                permissive_query = "SELECT * FROM profiles WHERE phone = $1"
                permissive_row = await self.db.fetchrow(permissive_query, phone) if self.db else None
                if permissive_row:
                    logger.debug(f"Found profile with permissive query: {dict(permissive_row)}")
                    return self._row_to_profile(permissive_row)
                else:
                    logger.debug(f"Still no profile found with permissive query")
                    
            return self._row_to_profile(row) if row else None
        
        except Exception as e:
            logger.error(f"Failed to get profile by phone '{phone}': {e}")
            logger.exception(e)  # Log full traceback
            return None  # Return None instead of raising exception

    async def get_profile_by_id(self, profile_id: uuid.UUID, include_deleted: bool = False) -> Optional[Profile]:
        """Get profile by ID"""
        await self._ensure_db()
        try:
            if not self.db or not self.db.pool:
                logger.warning("Profile database not available")
                return None
                
            where_clause = "WHERE id = $1"
            if not include_deleted:
                where_clause += " AND status != 'deleted'"
            
            query = f"""
                SELECT * FROM profiles
                {where_clause}
            """
            
            row = await self.db.fetchrow(query, profile_id) if self.db else None
            return self._row_to_profile(row) if row else None
            
        except Exception as e:
            logger.error(f"Failed to get profile by ID {profile_id}: {e}")
            return None

    async def get_profiles_by_persona(self, persona: str, options: Optional[QueryOptions] = None) -> List[Profile]:
        """Get profiles by persona"""
        await self._ensure_db()
        options = options or QueryOptions()
        
        try:
            if not self.db or not self.db.pool:
                logger.warning("Profile database not available")
                return []
                
            where_clause = "WHERE persona = $1"
            if not options.include_deleted:
                where_clause += " AND status != 'deleted'"
            
            query = f"""
                SELECT * FROM profiles
                {where_clause}
                ORDER BY {options.order_by} {'DESC' if options.order_desc else 'ASC'}
                LIMIT {options.limit}
            """
            
            rows = await self.db.fetch(query, persona) if self.db else []
            profiles = [self._row_to_profile(row) for row in rows]
            # Filter out None values
            return [p for p in profiles if p is not None]
            
        except Exception as e:
            logger.error(f"Failed to get profiles by persona {persona}: {e}")
            return []

    async def get_profiles_by_consent(self, has_consent: bool, options: Optional[QueryOptions] = None) -> List[Profile]:
        """Get profiles by consent status"""
        await self._ensure_db()
        options = options or QueryOptions()
        
        try:
            if not self.db or not self.db.pool:
                logger.warning("Profile database not available")
                return []
                
            where_clause = "WHERE consent = $1"
            if not options.include_deleted:
                where_clause += " AND status != 'deleted'"
            
            query = f"""
                SELECT * FROM profiles
                {where_clause}
                ORDER BY {options.order_by} {'DESC' if options.order_desc else 'ASC'}
                LIMIT {options.limit}
            """
            
            rows = await self.db.fetch(query, has_consent) if self.db else []
            profiles = [self._row_to_profile(row) for row in rows]
            # Filter out None values
            return [p for p in profiles if p is not None]
            
        except Exception as e:
            logger.error(f"Failed to get profiles by consent {has_consent}: {e}")
            return []

    async def list_profiles(self, options: Optional[QueryOptions] = None) -> Tuple[List[Profile], int]:
        """List profiles with pagination and filtering"""
        await self._ensure_db()
        options = options or QueryOptions()
        
        try:
            if not self.db or not self.db.pool:
                logger.warning("Profile database not available")
                return [], 0
                
            # Build WHERE clause
            where_conditions = []
            params = []
            
            if not options.include_deleted:
                where_conditions.append("status != 'deleted'")
            
            if not options.include_merged:
                where_conditions.append("status != 'merged'")
            
            where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""
            
            # Count total
            count_query = f"SELECT COUNT(*) FROM profiles {where_clause}"
            total = await self.db.fetchval(count_query, *params) if self.db else 0
            
            # Get profiles
            order_clause = f"ORDER BY {options.order_by} {'DESC' if options.order_desc else 'ASC'}"
            limit_clause = f"LIMIT {options.limit} OFFSET {options.offset}"
            
            query = f"""
                SELECT * FROM profiles
                {where_clause}
                {order_clause}
                {limit_clause}
            """
            
            rows = await self.db.fetch(query, *params) if self.db else []
            profiles = [self._row_to_profile(row) for row in rows]
            # Filter out None values
            filtered_profiles = [p for p in profiles if p is not None]
            return filtered_profiles, total
            
        except Exception as e:
            logger.error(f"Failed to list profiles: {e}")
            return [], 0

    async def search_profiles(
        self, 
        query: str, 
        fields: Optional[List[str]] = None, 
        options: Optional[QueryOptions] = None
    ) -> List[Profile]:
        """Search profiles by text in specified fields"""
        await self._ensure_db()
        fields = fields or ['name', 'display_name', 'phone', 'description']
        options = options or QueryOptions()
        
        try:
            if not self.db or not self.db.pool:
                logger.warning("Profile database not available")
                return []
                
            # Build search conditions
            search_conditions = []
            for field in fields:
                search_conditions.append(f"{field} ILIKE %s")
            
            where_clause = f"WHERE ({' OR '.join(search_conditions)})"
            if not options.include_deleted:
                where_clause += " AND status != 'deleted'"
            
            search_params = [f"%{query}%"] * len(fields)
            
            sql_query = f"""
                SELECT * FROM profiles
                {where_clause}
                ORDER BY {options.order_by} {'DESC' if options.order_desc else 'ASC'}
                LIMIT {options.limit}
            """
            
            rows = await self.db.fetch(sql_query, *search_params) if self.db else []
            profiles = [self._row_to_profile(row) for row in rows]
            # Filter out None values
            return [p for p in profiles if p is not None]
            
        except Exception as e:
            logger.error(f"Failed to search profiles: {e}")
            return []

    async def update_profile(
        self, 
        profile_id: uuid.UUID, 
        update_data: Union[ProfileUpdate, Dict[str, Any]], 
        updated_by: str = "system",
        expected_version: Optional[int] = None,
        reason: str = "Profile updated"
    ) -> Optional[Profile]:
        """Update profile with optimistic locking and audit logging"""
        await self._ensure_db()
        try:
            if not self.db or not self.db.pool:
                logger.warning("Profile database not available")
                return None
                
            # Convert to dict if ProfileUpdate model
            if isinstance(update_data, ProfileUpdate):
                update_dict = update_data.dict(exclude_unset=True)
            else:
                update_dict = update_data
            
            # Remove None values
            update_dict = {k: v for k, v in update_dict.items() if v is not None}
            
            if not update_dict:
                raise ValueError("No fields to update")
            
            async with self.db.pool.acquire() as conn:
                async with conn.transaction():
                    # Get current profile for audit trail
                    current_profile = await conn.fetchrow(
                        "SELECT * FROM profiles WHERE id = $1", profile_id
                    ) if self.db else None
                    
                    if not current_profile:
                        raise ValueError(f"Profile {profile_id} not found")
                    
                    # Check version for optimistic locking
                    if expected_version is not None and current_profile['version'] != expected_version:
                        raise ValueError(
                            f"Version conflict: expected {expected_version}, "
                            f"got {current_profile['version']}"
                        )
                    
                    # Prepare update data
                    new_version = current_profile['version'] + 1
                    update_dict['version'] = new_version
                    update_dict['updated_at'] = datetime.now().astimezone()
                    update_dict['updated_by'] = updated_by
                    
                    # Build dynamic update query
                    set_clauses = []
                    values = []
                    param_num = 1
                    
                    for field, value in update_dict.items():
                        if field == 'attributes':
                            set_clauses.append(f"{field} = ${param_num}")
                            values.append(json.dumps(value, default=json_serializer))
                        elif field == 'tags':
                            set_clauses.append(f"{field} = ${param_num}")
                            values.append(value)
                        else:
                            set_clauses.append(f"{field} = ${param_num}")
                            values.append(value)
                        param_num += 1
                    
                    # Add profile_id parameter
                    values.append(profile_id)
                    
                    query = f"""
                        UPDATE profiles 
                        SET {', '.join(set_clauses)}
                        WHERE id = ${param_num}
                        RETURNING *
                    """
                    
                    updated_row = await conn.fetchrow(query, *values) if self.db else None
                    
                    # Create audit entry
                    if updated_row is not None:
                        await self._create_audit_entry(
                            conn, profile_id, updated_by, ChangeType.UPDATE,
                            {
                                "before": dict(current_profile) if current_profile else {},
                                "after": dict(updated_row) if updated_row else {},
                                "changes": update_dict
                            },
                            reason
                        )
                    
                    profile = self._row_to_profile(updated_row)
                    logger.info(f"Updated profile {profile_id} to version {new_version}")
                    return profile
                    
        except Exception as e:
            logger.error(f"Failed to update profile {profile_id}: {e}")
            return None  # Return None instead of raising exception

    async def delete_profile(
        self,
        profile_id: uuid.UUID,
        deleted_by: str = "system",
        reason: str = "Profile deleted",
        hard_delete: bool = False
    ) -> bool:
        """Delete profile (soft delete by default, hard delete if specified)"""
        await self._ensure_db()
        try:
            if not self.db or not self.db.pool:
                logger.warning("Profile database not available")
                return False

            if hard_delete:
                # Permanent deletion
                query = "DELETE FROM profiles WHERE id = $1"
                result = await self.db.execute(query, profile_id) if self.db else None
                logger.info(f"Permanently deleted profile {profile_id}")
                return result is not None
            else:
                # Soft delete by updating status
                update_data = {
                    "status": "deleted",
                    "deleted_at": datetime.now().astimezone()
                }
                updated_profile = await self.update_profile(
                    profile_id, update_data, deleted_by, reason=reason
                )
                success = updated_profile is not None
                if success:
                    logger.info(f"Soft deleted profile {profile_id}")
                return success
                
        except Exception as e:
            logger.error(f"Failed to delete profile {profile_id}: {e}")
            return False

    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    async def _create_audit_entry(
        self,
        conn,
        profile_id: uuid.UUID,
        changed_by: str,
        change_type: ChangeType,
        change_data: Dict[str, Any],
        reason: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> uuid.UUID:
        """Create audit log entry"""
        audit_id = uuid.uuid4()
        
        await conn.execute("""
            INSERT INTO profile_history (
                id, profile_id, changed_by, change_type, change_data, 
                reason, session_id, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """, audit_id, profile_id, changed_by, change_type.value, 
            json.dumps(change_data, default=json_serializer), reason, session_id, datetime.now().astimezone())
        
        return audit_id
    
    def _row_to_profile(self, row: Optional[Dict[str, Any]]) -> Optional[Profile]:
        """Convert database row to Profile model"""
        if not row:
            return None
        
        # Handle JSONB attributes
        attributes = row.get('attributes', {})
        if isinstance(attributes, str):
            attributes = json.loads(attributes)
        
        # Handle array tags
        tags = row.get('tags', []) or []
        
        return Profile(
            id=row['id'],
            phone=row['phone'],
            name=row.get('name'),
            display_name=row.get('display_name'),
            persona=row.get('persona'),
            description=row.get('description'),
            language=row.get('language', 'en'),
            timezone=row.get('timezone'),
            consent=row.get('consent', False),
            consent_date=row.get('consent_date'),
            last_seen=row.get('last_seen'),
            created_at=row['created_at'],
            updated_at=row.get('updated_at'),
            version=row.get('version', 1),
            attributes=attributes,
            tags=tags,
            updated_by=row.get('updated_by', 'system'),
            status=ProfileStatus(row.get('status', 'active')),
            merged_into=row.get('merged_into'),
            deleted_at=row.get('deleted_at')
        )
    
    def _row_to_profile_history(self, row: Optional[Dict[str, Any]]) -> Optional[ProfileHistory]:
        """Convert database row to ProfileHistory model"""
        if not row:
            return None
            
        # Handle JSONB change_data
        change_data = row.get('change_data', {})
        if isinstance(change_data, str):
            change_data = json.loads(change_data)
        
        return ProfileHistory(
            id=row['id'],
            profile_id=row['profile_id'],
            changed_by=row['changed_by'],
            change_type=ChangeType(row['change_type']),
            change_data=change_data,
            reason=row.get('reason'),
            session_id=row.get('session_id'),
            ip_address=row.get('ip_address'),
            user_agent=row.get('user_agent'),
            created_at=row['created_at']
        )
    
    # ============================================================================
    # STATISTICS AND ANALYTICS
    # ============================================================================
    
    async def get_profile_stats(self) -> Dict[str, Any]:
        """Get profile statistics"""
        await self._ensure_db()
        try:
            if not self.db or not self.db.pool:
                logger.warning("Profile database not available")
                return {}
                
            stats = {}
            
            # Basic counts
            stats['total_profiles'] = await self.db.fetchval(
                "SELECT COUNT(*) FROM profiles WHERE status != 'deleted'"
            ) if self.db else 0
            
            stats['active_profiles'] = await self.db.fetchval(
                "SELECT COUNT(*) FROM profiles WHERE status = 'active'"
            ) if self.db else 0
            
            stats['consented_profiles'] = await self.db.fetchval(
                "SELECT COUNT(*) FROM profiles WHERE consent = true AND status = 'active'"
            ) if self.db else 0
            
            # Recent interactions (last 24 hours)
            yesterday = datetime.now().astimezone() - timedelta(days=1)
            stats['recent_interactions'] = await self.db.fetchval(
                "SELECT COUNT(*) FROM profiles WHERE last_seen > $1", yesterday
            ) if self.db else 0
            
            # Top personas
            persona_stats = await self.db.fetch("""SELECT persona, COUNT(*) as count
                FROM profiles 
                WHERE persona IS NOT NULL AND status = 'active'
                GROUP BY persona
                ORDER BY count DESC
                LIMIT 10
            """) if self.db else []
            stats['top_personas'] = [
                {"persona": row['persona'], "count": row['count']} 
                for row in persona_stats
            ]
            
            # Top languages
            language_stats = await self.db.fetch("""SELECT language, COUNT(*) as count
                FROM profiles 
                WHERE status = 'active'
                GROUP BY language
                ORDER BY count DESC
                LIMIT 10
            """) if self.db else []
            stats['top_languages'] = [
                {"language": row['language'], "count": row['count']} 
                for row in language_stats
            ]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get profile stats: {e}")
            return {}