"""
Audit Logger - Tracks and manages database changes for undo functionality
Provides comprehensive audit logging and rollback capabilities
"""

import logging
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ChangeType(Enum):
    """Types of database changes"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"

@dataclass
class AuditEntry:
    """Represents a single audit log entry"""
    id: str
    timestamp: datetime
    user_id: str
    change_type: ChangeType
    table_name: str
    record_id: str
    old_values: Dict[str, Any]
    new_values: Dict[str, Any]
    reason: str
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class AuditLogger:
    """Manages audit logging and provides undo functionality"""
    
    def __init__(self, db_manager=None):
        self.db_manager = db_manager
        self.audit_collection = None
        if db_manager:
            self.audit_collection = db_manager.get_collection('audit_log')
    
    async def log_change(self, user_id: str, change_type: ChangeType, table_name: str, 
                        record_id: str, old_values: Dict[str, Any], new_values: Dict[str, Any],
                        reason: str, session_id: Optional[str] = None,
                        ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> str:
        """
        Log a database change to the audit log
        
        Args:
            user_id: ID of the user making the change
            change_type: Type of change (CREATE, UPDATE, DELETE, MERGE)
            table_name: Name of the table being modified
            record_id: ID of the record being modified
            old_values: Values before the change
            new_values: Values after the change
            reason: Reason for the change
            session_id: Session ID (optional)
            ip_address: IP address (optional)
            user_agent: User agent (optional)
            
        Returns:
            ID of the created audit entry
        """
        try:
            audit_id = str(uuid.uuid4())
            timestamp = datetime.utcnow()
            
            # Create audit entry
            audit_entry = AuditEntry(
                id=audit_id,
                timestamp=timestamp,
                user_id=user_id,
                change_type=change_type,
                table_name=table_name,
                record_id=record_id,
                old_values=old_values,
                new_values=new_values,
                reason=reason,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Save to database if available
            if self.audit_collection:
                try:
                    # Convert datetime to string for JSON serialization
                    entry_dict = asdict(audit_entry)
                    entry_dict['timestamp'] = entry_dict['timestamp'].isoformat()
                    entry_dict['change_type'] = entry_dict['change_type'].value
                    
                    # Handle serialization of complex objects
                    entry_dict['old_values'] = self._serialize_values(entry_dict['old_values'])
                    entry_dict['new_values'] = self._serialize_values(entry_dict['new_values'])
                    
                    await self.audit_collection.insert_one(entry_dict)
                    logger.info(f"Audit entry {audit_id} logged for {change_type.value} on {table_name}")
                except Exception as e:
                    logger.error(f"Failed to save audit entry to database: {e}")
            else:
                logger.debug(f"Audit entry {audit_id} created but not saved (no database connection)")
            
            return audit_id
            
        except Exception as e:
            logger.error(f"Error logging audit entry: {e}")
            raise
    
    def _serialize_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize values for JSON storage"""
        serialized = {}
        for key, value in values.items():
            try:
                # Handle datetime objects
                if isinstance(value, datetime):
                    serialized[key] = value.isoformat()
                # Handle other objects that might not be JSON serializable
                elif not isinstance(value, (str, int, float, bool, type(None), list, dict)):
                    serialized[key] = str(value)
                else:
                    serialized[key] = value
            except Exception:
                # If serialization fails, convert to string
                serialized[key] = str(value)
        return serialized
    
    async def get_audit_trail(self, record_id: str, table_name: str, limit: int = 50) -> List[AuditEntry]:
        """
        Get audit trail for a specific record
        
        Args:
            record_id: ID of the record
            table_name: Name of the table
            limit: Maximum number of entries to return
            
        Returns:
            List of audit entries
        """
        try:
            if not self.audit_collection:
                logger.warning("Audit collection not available")
                return []
            
            # Query audit log
            cursor = self.audit_collection.find({
                'record_id': record_id,
                'table_name': table_name
            }).sort('timestamp', -1).limit(limit)
            
            entries = []
            async for entry_dict in cursor:
                try:
                    # Convert back to AuditEntry
                    entry_dict['timestamp'] = datetime.fromisoformat(entry_dict['timestamp'])
                    entry_dict['change_type'] = ChangeType(entry_dict['change_type'])
                    entry = AuditEntry(**entry_dict)
                    entries.append(entry)
                except Exception as e:
                    logger.error(f"Error deserializing audit entry: {e}")
            
            return entries
            
        except Exception as e:
            logger.error(f"Error retrieving audit trail: {e}")
            return []
    
    async def undo_change(self, audit_id: str, user_id: str) -> bool:
        """
        Undo a specific change using its audit ID
        
        Args:
            audit_id: ID of the audit entry to undo
            user_id: ID of the user performing the undo
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.audit_collection:
                logger.error("Audit collection not available")
                return False
            
            # Get the audit entry
            entry_dict = await self.audit_collection.find_one({'id': audit_id})
            if not entry_dict:
                logger.error(f"Audit entry {audit_id} not found")
                return False
            
            # Convert to AuditEntry
            entry_dict['timestamp'] = datetime.fromisoformat(entry_dict['timestamp'])
            entry_dict['change_type'] = ChangeType(entry_dict['change_type'])
            audit_entry = AuditEntry(**entry_dict)
            
            # Verify user has permission to undo this change
            if audit_entry.user_id != user_id:
                logger.warning(f"User {user_id} attempted to undo change by {audit_entry.user_id}")
                return False
            
            # Perform the undo operation based on change type
            success = False
            if audit_entry.change_type == ChangeType.CREATE:
                success = await self._undo_create(audit_entry)
            elif audit_entry.change_type == ChangeType.UPDATE:
                success = await self._undo_update(audit_entry)
            elif audit_entry.change_type == ChangeType.DELETE:
                success = await self._undo_delete(audit_entry)
            elif audit_entry.change_type == ChangeType.MERGE:
                success = await self._undo_merge(audit_entry)
            
            if success:
                # Log the undo operation
                await self.log_change(
                    user_id=user_id,
                    change_type=ChangeType.UPDATE,
                    table_name=f"{audit_entry.table_name}_audit",
                    record_id=audit_entry.id,
                    old_values={"status": "active"},
                    new_values={"status": "undone", "undone_by": user_id, "undone_at": datetime.utcnow().isoformat()},
                    reason=f"Undo of {audit_entry.change_type.value} operation",
                    session_id=audit_entry.session_id
                )
                logger.info(f"Successfully undid change {audit_id}")
            else:
                logger.error(f"Failed to undo change {audit_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error undoing change {audit_id}: {e}")
            return False
    
    async def _undo_create(self, audit_entry: AuditEntry) -> bool:
        """Undo a CREATE operation by deleting the record"""
        try:
            # Get the appropriate collection
            if self.db_manager is None:
                logger.error("Database manager not available")
                return False
                
            collection = self.db_manager.get_collection(audit_entry.table_name)
            if collection is None:
                logger.error(f"Collection {audit_entry.table_name} not found")
                return False
            
            # Delete the record
            result = await collection.delete_one({'id': audit_entry.record_id})
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error undoing CREATE operation: {e}")
            return False
    
    async def _undo_update(self, audit_entry: AuditEntry) -> bool:
        """Undo an UPDATE operation by reverting to old values"""
        try:
            # Get the appropriate collection
            if self.db_manager is None:
                logger.error("Database manager not available")
                return False
                
            collection = self.db_manager.get_collection(audit_entry.table_name)
            if collection is None:
                logger.error(f"Collection {audit_entry.table_name} not found")
                return False
            
            # Update the record with old values
            result = await collection.update_one(
                {'id': audit_entry.record_id},
                {'$set': audit_entry.old_values}
            )
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error undoing UPDATE operation: {e}")
            return False
    
    async def _undo_delete(self, audit_entry: AuditEntry) -> bool:
        """Undo a DELETE operation by recreating the record"""
        try:
            # Get the appropriate collection
            if self.db_manager is None:
                logger.error("Database manager not available")
                return False
                
            collection = self.db_manager.get_collection(audit_entry.table_name)
            if collection is None:
                logger.error(f"Collection {audit_entry.table_name} not found")
                return False
            
            # Recreate the record with old values
            old_values = audit_entry.old_values.copy()
            old_values['id'] = audit_entry.record_id
            result = await collection.insert_one(old_values)
            return result.inserted_id is not None
            
        except Exception as e:
            logger.error(f"Error undoing DELETE operation: {e}")
            return False
    
    async def _undo_merge(self, audit_entry: AuditEntry) -> bool:
        """Undo a MERGE operation (complex - placeholder implementation)"""
        try:
            # Merging is complex and would require detailed tracking of the merge operation
            # This is a placeholder implementation
            logger.warning(f"Undo merge operation not fully implemented for {audit_entry.id}")
            return False
            
        except Exception as e:
            logger.error(f"Error undoing MERGE operation: {e}")
            return False
    
    async def get_user_audit_trail(self, user_id: str, limit: int = 100) -> List[AuditEntry]:
        """
        Get audit trail for all changes made by a specific user
        
        Args:
            user_id: ID of the user
            limit: Maximum number of entries to return
            
        Returns:
            List of audit entries
        """
        try:
            if not self.audit_collection:
                logger.warning("Audit collection not available")
                return []
            
            # Query audit log
            cursor = self.audit_collection.find({
                'user_id': user_id
            }).sort('timestamp', -1).limit(limit)
            
            entries = []
            async for entry_dict in cursor:
                try:
                    # Convert back to AuditEntry
                    entry_dict['timestamp'] = datetime.fromisoformat(entry_dict['timestamp'])
                    entry_dict['change_type'] = ChangeType(entry_dict['change_type'])
                    entry = AuditEntry(**entry_dict)
                    entries.append(entry)
                except Exception as e:
                    logger.error(f"Error deserializing audit entry: {e}")
            
            return entries
            
        except Exception as e:
            logger.error(f"Error retrieving user audit trail: {e}")
            return []

# Global audit logger instance
audit_logger = AuditLogger()