"""
Advanced privacy controls for the WhatsApp AI Agent
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
from enum import Enum

# Add the project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..database import db_manager
from ..models.profiles import ConsentType, ConsentMethod

logger = logging.getLogger(__name__)

class DataRetentionPeriod(Enum):
    """Data retention period options"""
    ONE_DAY = "1_day"
    ONE_WEEK = "1_week"
    ONE_MONTH = "1_month"
    THREE_MONTHS = "3_months"
    SIX_MONTHS = "6_months"
    ONE_YEAR = "1_year"
    INDEFINITE = "indefinite"

class PrivacyLevel(Enum):
    """Privacy level options"""
    MINIMAL = "minimal"      # Only essential data
    STANDARD = "standard"    # Basic personalization
    ENHANCED = "enhanced"    # Full personalization
    MAXIMUM = "maximum"      # Maximum privacy (no data storage)

class PrivacyController:
    """Advanced privacy controls for the WhatsApp AI Agent"""
    
    def __init__(self):
        self.default_retention_period = DataRetentionPeriod.THREE_MONTHS
        self.default_privacy_level = PrivacyLevel.STANDARD
        self.data_categories = {
            'conversations': 'Conversation transcripts and summaries',
            'profiles': 'User profile information',
            'preferences': 'User preferences and settings',
            'memories': 'Long-term memory entries',
            'analytics': 'Usage analytics and metrics',
            'feedback': 'User feedback and ratings'
        }
    
    async def get_user_privacy_settings(self, user_id: str) -> Dict[str, Any]:
        """
        Get privacy settings for a specific user.
        
        Args:
            user_id: User identifier (phone number)
            
        Returns:
            Dictionary with user privacy settings
        """
        try:
            # Check if user has specific privacy settings
            privacy_settings = await self._get_user_specific_settings(user_id)
            
            if privacy_settings:
                return privacy_settings
            
            # Return default settings
            return {
                'user_id': user_id,
                'privacy_level': self.default_privacy_level.value,
                'retention_period': self.default_retention_period.value,
                'data_categories': {category: True for category in self.data_categories},
                'consent_given': False,
                'consent_date': None,
                'last_updated': datetime.utcnow().isoformat(),
                'auto_delete_date': self._calculate_auto_delete_date(self.default_retention_period).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting user privacy settings: {e}")
            # Return default settings on error
            return {
                'user_id': user_id,
                'privacy_level': self.default_privacy_level.value,
                'retention_period': self.default_retention_period.value,
                'data_categories': {category: True for category in self.data_categories},
                'consent_given': False,
                'consent_date': None,
                'last_updated': datetime.utcnow().isoformat(),
                'auto_delete_date': self._calculate_auto_delete_date(self.default_retention_period).isoformat()
            }
    
    async def _get_user_specific_settings(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user-specific privacy settings from database.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with user settings or None if not found
        """
        try:
            get_col = getattr(db_manager, 'get_collection', None)
            if not callable(get_col):
                return None
            
            privacy_col = get_col('privacy_settings')
            if privacy_col is None:
                return None
            
            def _find_settings():
                return privacy_col.find_one({'user_id': user_id})
            
            settings = await asyncio.to_thread(_find_settings)
            return dict(settings) if settings else None
            
        except Exception as e:
            logger.warning(f"Error retrieving user privacy settings: {e}")
            return None
    
    async def update_user_privacy_settings(self, user_id: str, settings: Dict[str, Any]) -> bool:
        """
        Update privacy settings for a specific user.
        
        Args:
            user_id: User identifier
            settings: Dictionary with new privacy settings
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Validate settings
            validated_settings = self._validate_privacy_settings(settings)
            validated_settings['user_id'] = user_id
            validated_settings['last_updated'] = datetime.utcnow().isoformat()
            
            # Calculate auto-delete date based on retention period
            retention_period = DataRetentionPeriod(validated_settings.get('retention_period', self.default_retention_period.value))
            validated_settings['auto_delete_date'] = self._calculate_auto_delete_date(retention_period).isoformat()
            
            # Save to database
            get_col = getattr(db_manager, 'get_collection', None)
            if not callable(get_col):
                return False
            
            privacy_col = get_col('privacy_settings')
            if privacy_col is None:
                return False
            
            def _update_settings():
                privacy_col.update_one(
                    {'user_id': user_id},
                    {'$set': validated_settings},
                    upsert=True
                )
            
            await asyncio.to_thread(_update_settings)
            
            logger.info(f"Updated privacy settings for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating user privacy settings: {e}")
            return False
    
    def _validate_privacy_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate privacy settings.
        
        Args:
            settings: Dictionary with privacy settings
            
        Returns:
            Validated settings dictionary
        """
        validated = {}
        
        # Validate privacy level
        privacy_level = settings.get('privacy_level', self.default_privacy_level.value)
        try:
            validated['privacy_level'] = PrivacyLevel(privacy_level).value
        except ValueError:
            validated['privacy_level'] = self.default_privacy_level.value
        
        # Validate retention period
        retention_period = settings.get('retention_period', self.default_retention_period.value)
        try:
            validated['retention_period'] = DataRetentionPeriod(retention_period).value
        except ValueError:
            validated['retention_period'] = self.default_retention_period.value
        
        # Validate data categories
        data_categories = settings.get('data_categories', {})
        validated['data_categories'] = {}
        for category, enabled in data_categories.items():
            if category in self.data_categories:
                validated['data_categories'][category] = bool(enabled)
        
        # Ensure all categories are present
        for category in self.data_categories:
            if category not in validated['data_categories']:
                validated['data_categories'][category] = True
        
        # Validate consent
        validated['consent_given'] = bool(settings.get('consent_given', False))
        consent_date = settings.get('consent_date')
        if consent_date and validated['consent_given']:
            validated['consent_date'] = consent_date
        elif validated['consent_given']:
            validated['consent_date'] = datetime.utcnow().isoformat()
        else:
            validated['consent_date'] = None
        
        return validated
    
    def _calculate_auto_delete_date(self, retention_period: DataRetentionPeriod) -> datetime:
        """
        Calculate auto-delete date based on retention period.
        
        Args:
            retention_period: Data retention period
            
        Returns:
            Auto-delete date
        """
        now = datetime.utcnow()
        
        if retention_period == DataRetentionPeriod.ONE_DAY:
            return now + timedelta(days=1)
        elif retention_period == DataRetentionPeriod.ONE_WEEK:
            return now + timedelta(weeks=1)
        elif retention_period == DataRetentionPeriod.ONE_MONTH:
            return now + timedelta(days=30)
        elif retention_period == DataRetentionPeriod.THREE_MONTHS:
            return now + timedelta(days=90)
        elif retention_period == DataRetentionPeriod.SIX_MONTHS:
            return now + timedelta(days=180)
        elif retention_period == DataRetentionPeriod.ONE_YEAR:
            return now + timedelta(days=365)
        else:  # INDEFINITE
            # Set to far future date
            return now + timedelta(days=365*10)
    
    async def anonymize_user_data(self, user_id: str) -> bool:
        """
        Anonymize all data for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if anonymization successful, False otherwise
        """
        try:
            success = True
            
            # Anonymize conversations
            success &= await self._anonymize_conversations(user_id)
            
            # Anonymize messages
            success &= await self._anonymize_messages(user_id)
            
            # Anonymize profiles
            success &= await self._anonymize_profiles(user_id)
            
            # Anonymize memories
            success &= await self._anonymize_memories(user_id)
            
            # Remove privacy settings
            success &= await self._remove_privacy_settings(user_id)
            
            if success:
                logger.info(f"Successfully anonymized data for user {user_id}")
            else:
                logger.warning(f"Partial success anonymizing data for user {user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error anonymizing user data: {e}")
            return False
    
    async def _anonymize_conversations(self, user_id: str) -> bool:
        """Anonymize conversation data for a user"""
        try:
            get_col = getattr(db_manager, 'get_collection', None)
            if not callable(get_col):
                return True  # No database, consider successful
            
            convs = get_col('conversations')
            if convs is None:
                return True
            
            def _anonymize_conv_data():
                convs.update_many(
                    {'phone_number': user_id},
                    {
                        '$set': {
                            'phone_number': f"anon_{hash(user_id) % 1000000}",
                            'transcript': '[ANONYMIZED]',
                            'summary': '[ANONYMIZED]',
                            'participant_names': ['[ANONYMIZED]']
                        }
                    }
                )
            
            await asyncio.to_thread(_anonymize_conv_data)
            return True
            
        except Exception as e:
            logger.warning(f"Error anonymizing conversations for {user_id}: {e}")
            return False
    
    async def _anonymize_messages(self, user_id: str) -> bool:
        """Anonymize message data for a user"""
        try:
            get_col = getattr(db_manager, 'get_collection', None)
            if not callable(get_col):
                return True
            
            msgs = get_col('messages')
            if msgs is None:
                return True
            
            def _anonymize_msg_data():
                msgs.update_many(
                    {'sender': user_id},
                    {
                        '$set': {
                            'sender': f"anon_{hash(user_id) % 1000000}",
                            'text': '[ANONYMIZED]'
                        }
                    }
                )
            
            await asyncio.to_thread(_anonymize_msg_data)
            return True
            
        except Exception as e:
            logger.warning(f"Error anonymizing messages for {user_id}: {e}")
            return False
    
    async def _anonymize_profiles(self, user_id: str) -> bool:
        """Anonymize profile data for a user"""
        try:
            get_col = getattr(db_manager, 'get_collection', None)
            if not callable(get_col):
                return True
            
            profiles = get_col('profiles')
            if profiles is None:
                return True
            
            def _anonymize_profile_data():
                profiles.update_many(
                    {'phone': user_id},
                    {
                        '$set': {
                            'phone': f"anon_{hash(user_id) % 1000000}",
                            'name': '[ANONYMIZED]',
                            'display_name': '[ANONYMIZED]',
                            'description': '[ANONYMIZED]',
                            'attributes': {}
                        }
                    }
                )
            
            await asyncio.to_thread(_anonymize_profile_data)
            return True
            
        except Exception as e:
            logger.warning(f"Error anonymizing profiles for {user_id}: {e}")
            return False
    
    async def _anonymize_memories(self, user_id: str) -> bool:
        """Anonymize memory data for a user"""
        try:
            get_col = getattr(db_manager, 'get_collection', None)
            if not callable(get_col):
                return True
            
            memories = get_col('memories')
            if memories is None:
                return True
            
            def _anonymize_memory_data():
                memories.update_many(
                    {'user_id': user_id},
                    {
                        '$set': {
                            'user_id': f"anon_{hash(user_id) % 1000000}",
                            'content': '[ANONYMIZED]'
                        }
                    }
                )
            
            await asyncio.to_thread(_anonymize_memory_data)
            return True
            
        except Exception as e:
            logger.warning(f"Error anonymizing memories for {user_id}: {e}")
            return False
    
    async def _remove_privacy_settings(self, user_id: str) -> bool:
        """Remove privacy settings for a user"""
        try:
            get_col = getattr(db_manager, 'get_collection', None)
            if not callable(get_col):
                return True
            
            privacy_col = get_col('privacy_settings')
            if privacy_col is None:
                return True
            
            def _remove_settings():
                privacy_col.delete_one({'user_id': user_id})
            
            await asyncio.to_thread(_remove_settings)
            return True
            
        except Exception as e:
            logger.warning(f"Error removing privacy settings for {user_id}: {e}")
            return False
    
    async def delete_expired_data(self) -> Dict[str, int]:
        """
        Delete data that has exceeded its retention period.
        
        Returns:
            Dictionary with counts of deleted records by type
        """
        try:
            deletion_counts = {
                'conversations': 0,
                'messages': 0,
                'profiles': 0,
                'memories': 0,
                'privacy_settings': 0
            }
            
            # Get current date for comparison
            now = datetime.utcnow()
            
            # Get all users with privacy settings
            get_col = getattr(db_manager, 'get_collection', None)
            if not callable(get_col):
                return deletion_counts
            
            privacy_col = get_col('privacy_settings')
            if privacy_col is None:
                return deletion_counts
            
            def _get_all_privacy_settings():
                return list(privacy_col.find({}))
            
            all_settings = await asyncio.to_thread(_get_all_privacy_settings)
            
            for settings in all_settings:
                try:
                    user_id = settings.get('user_id')
                    auto_delete_str = settings.get('auto_delete_date')
                    
                    if not user_id or not auto_delete_str:
                        continue
                    
                    # Parse auto-delete date
                    auto_delete_date = datetime.fromisoformat(auto_delete_str.replace('Z', '+00:00'))
                    
                    # Check if data should be deleted
                    if now >= auto_delete_date:
                        logger.info(f"Deleting expired data for user {user_id}")
                        await self._delete_user_data(user_id, deletion_counts)
                        
                except Exception as e:
                    logger.warning(f"Error processing privacy settings for user: {e}")
                    continue
            
            logger.info(f"Expired data deletion completed: {deletion_counts}")
            return deletion_counts
            
        except Exception as e:
            logger.error(f"Error deleting expired data: {e}")
            return {'error': str(e)}
    
    async def _delete_user_data(self, user_id: str, deletion_counts: Dict[str, int]) -> None:
        """Delete all data for a specific user"""
        try:
            get_col = getattr(db_manager, 'get_collection', None)
            if not callable(get_col):
                return
            
            # Delete conversations
            convs = get_col('conversations')
            if convs is not None:
                def _delete_convs():
                    result = convs.delete_many({'phone_number': user_id})
                    return result.deleted_count
                
                count = await asyncio.to_thread(_delete_convs)
                deletion_counts['conversations'] += count
            
            # Delete messages
            msgs = get_col('messages')
            if msgs is not None:
                def _delete_msgs():
                    result = msgs.delete_many({'sender': user_id})
                    return result.deleted_count
                
                count = await asyncio.to_thread(_delete_msgs)
                deletion_counts['messages'] += count
            
            # Delete profiles
            profiles = get_col('profiles')
            if profiles is not None:
                def _delete_profiles():
                    result = profiles.delete_many({'phone': user_id})
                    return result.deleted_count
                
                count = await asyncio.to_thread(_delete_profiles)
                deletion_counts['profiles'] += count
            
            # Delete memories
            memories = get_col('memories')
            if memories is not None:
                def _delete_memories():
                    result = memories.delete_many({'user_id': user_id})
                    return result.deleted_count
                
                count = await asyncio.to_thread(_delete_memories)
                deletion_counts['memories'] += count
            
            # Delete privacy settings
            privacy_col = get_col('privacy_settings')
            if privacy_col is not None:
                def _delete_privacy():
                    result = privacy_col.delete_one({'user_id': user_id})
                    return result.deleted_count
                
                count = await asyncio.to_thread(_delete_privacy)
                deletion_counts['privacy_settings'] += count
            
            logger.info(f"Deleted data for user {user_id}: {deletion_counts}")
            
        except Exception as e:
            logger.error(f"Error deleting data for user {user_id}: {e}")
    
    async def get_privacy_report(self, user_id: str) -> Dict[str, Any]:
        """
        Generate a privacy report for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with privacy report information
        """
        try:
            # Get user privacy settings
            settings = await self.get_user_privacy_settings(user_id)
            
            # Get data storage information
            data_info = await self._get_user_data_info(user_id)
            
            # Calculate retention info
            retention_info = await self._get_retention_info(user_id, settings)
            
            report = {
                'user_id': user_id,
                'report_generated': datetime.utcnow().isoformat(),
                'privacy_settings': settings,
                'data_storage': data_info,
                'retention_info': retention_info,
                'compliance_status': await self._check_compliance_status(settings)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating privacy report: {e}")
            return {
                'user_id': user_id,
                'error': str(e),
                'report_generated': datetime.utcnow().isoformat()
            }
    
    async def _get_user_data_info(self, user_id: str) -> Dict[str, Any]:
        """Get information about user data storage"""
        try:
            data_info = {}
            
            get_col = getattr(db_manager, 'get_collection', None)
            if not callable(get_col):
                return data_info
            
            # Count conversations
            convs = get_col('conversations')
            if convs is not None:
                def _count_convs():
                    return convs.count_documents({'phone_number': user_id})
                
                data_info['conversations_count'] = await asyncio.to_thread(_count_convs)
            
            # Count messages
            msgs = get_col('messages')
            if msgs is not None:
                def _count_msgs():
                    return msgs.count_documents({'sender': user_id})
                
                data_info['messages_count'] = await asyncio.to_thread(_count_msgs)
            
            # Check profile
            profiles = get_col('profiles')
            if profiles is not None:
                def _check_profile():
                    return profiles.find_one({'phone': user_id}) is not None
                
                data_info['profile_exists'] = await asyncio.to_thread(_check_profile)
            
            # Count memories
            memories = get_col('memories')
            if memories is not None:
                def _count_memories():
                    return memories.count_documents({'user_id': user_id})
                
                data_info['memories_count'] = await asyncio.to_thread(_count_memories)
            
            return data_info
            
        except Exception as e:
            logger.warning(f"Error getting user data info: {e}")
            return {}
    
    async def _get_retention_info(self, user_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Get retention information for user data"""
        try:
            retention_info = {}
            
            # Get auto-delete date
            auto_delete_str = settings.get('auto_delete_date')
            if auto_delete_str:
                auto_delete_date = datetime.fromisoformat(auto_delete_str.replace('Z', '+00:00'))
                retention_info['auto_delete_date'] = auto_delete_str
                retention_info['days_until_deletion'] = (auto_delete_date - datetime.utcnow()).days
            
            # Get retention period
            retention_info['retention_period'] = settings.get('retention_period', self.default_retention_period.value)
            
            return retention_info
            
        except Exception as e:
            logger.warning(f"Error getting retention info: {e}")
            return {}
    
    async def _check_compliance_status(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Check compliance status based on privacy settings"""
        try:
            compliance = {
                'gdpr_compliant': settings.get('consent_given', False),
                'ccpa_compliant': settings.get('consent_given', False),
                'privacy_level': settings.get('privacy_level', self.default_privacy_level.value)
            }
            
            # Check if minimal data collection is enabled
            data_categories = settings.get('data_categories', {})
            compliance['minimal_data_collection'] = all(not enabled for enabled in data_categories.values())
            
            return compliance
            
        except Exception as e:
            logger.warning(f"Error checking compliance status: {e}")
            return {}

# Global privacy controller instance
privacy_controller = PrivacyController()

async def get_user_privacy_settings(user_id: str) -> Dict[str, Any]:
    """
    Get privacy settings for a user.
    
    Args:
        user_id: User identifier
        
    Returns:
        Dictionary with user privacy settings
    """
    return await privacy_controller.get_user_privacy_settings(user_id)

async def update_user_privacy_settings(user_id: str, settings: Dict[str, Any]) -> bool:
    """
    Update privacy settings for a user.
    
    Args:
        user_id: User identifier
        settings: Dictionary with new privacy settings
        
    Returns:
        True if update successful, False otherwise
    """
    return await privacy_controller.update_user_privacy_settings(user_id, settings)

async def anonymize_user_data(user_id: str) -> bool:
    """
    Anonymize all data for a user.
    
    Args:
        user_id: User identifier
        
    Returns:
        True if anonymization successful, False otherwise
    """
    return await privacy_controller.anonymize_user_data(user_id)

async def delete_expired_data() -> Dict[str, int]:
    """
    Delete data that has exceeded its retention period.
    
    Returns:
        Dictionary with counts of deleted records by type
    """
    return await privacy_controller.delete_expired_data()

async def get_privacy_report(user_id: str) -> Dict[str, Any]:
    """
    Generate a privacy report for a user.
    
    Args:
        user_id: User identifier
        
    Returns:
        Dictionary with privacy report information
    """
    return await privacy_controller.get_privacy_report(user_id)

async def schedule_periodic_data_cleanup(interval_hours: int = 24):
    """
    Schedule periodic data cleanup for expired data.
    
    Args:
        interval_hours: How often to run cleanup (default: 24 hours)
    """
    while True:
        try:
            logger.info("Running periodic data cleanup...")
            deletion_counts = await delete_expired_data()
            logger.info(f"Data cleanup completed: {deletion_counts}")
            
            # Wait for next cleanup
            await asyncio.sleep(interval_hours * 3600)
            
        except asyncio.CancelledError:
            logger.info("Periodic data cleanup cancelled")
            break
        except Exception as e:
            logger.error(f"Error in periodic data cleanup: {e}")
            await asyncio.sleep(interval_hours * 3600)

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    async def main():
        user_id = "+1234567890"
        
        # Get current privacy settings
        settings = await get_user_privacy_settings(user_id)
        print(f"Current settings: {settings}")
        
        # Update privacy settings
        new_settings = {
            'privacy_level': 'minimal',
            'retention_period': '1_month',
            'data_categories': {
                'conversations': False,
                'profiles': True,
                'preferences': False,
                'memories': False,
                'analytics': False,
                'feedback': True
            },
            'consent_given': True
        }
        
        success = await update_user_privacy_settings(user_id, new_settings)
        print(f"Settings update successful: {success}")
        
        # Generate privacy report
        report = await get_privacy_report(user_id)
        print(f"Privacy report: {json.dumps(report, indent=2)}")
    
    asyncio.run(main())