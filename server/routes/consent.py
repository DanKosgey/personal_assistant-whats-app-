"""
API endpoints for consent management and GDPR compliance
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

from ..services.profile_service import ProfileService
from ..models.profiles import ConsentType, ConsentMethod

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/consent", tags=["consent"])

@router.get("/{phone}/status")
async def get_consent_status(phone: str):
    """Get consent status for a user"""
    try:
        profile_service = ProfileService()
        profile = await profile_service.get_or_create_profile(phone, auto_create=False)
        
        return {
            "success": True,
            "phone": phone,
            "profile_exists": profile is not None,
            "memory_consent": profile.consent if profile else False,
            "consent_date": profile.consent_date.isoformat() if profile and profile.consent_date else None
        }
    except Exception as e:
        logger.error(f"Error getting consent status for {phone}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{phone}/grant")
async def grant_consent(
    phone: str,
    consent_type: str = Query("memory_storage"),
    method: str = Query("api_call")
):
    """Grant consent for a user"""
    try:
        consent_type_enum = ConsentType(consent_type)
        consent_method_enum = ConsentMethod(method)
        
        profile_service = ProfileService()
        response = await profile_service.update_consent(
            phone=phone,
            consent_type=consent_type_enum,
            granted=True,
            method=consent_method_enum,
            actor="admin_api"
        )
        
        if not response.success:
            raise HTTPException(status_code=400, detail=response.message)
        
        return {
            "success": True,
            "message": f"Consent {consent_type} granted for {phone}"
        }
    except Exception as e:
        logger.error(f"Error granting consent for {phone}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{phone}/data-export")
async def export_user_data(phone: str):
    """Export all stored data for a user (GDPR Article 20)"""
    try:
        profile_service = ProfileService()
        profile = await profile_service.get_or_create_profile(phone, auto_create=False)
        
        if not profile:
            raise HTTPException(status_code=404, detail="No profile found")
        
        return {
            "success": True,
            "data_export": {
                "export_timestamp": datetime.now().astimezone().isoformat(),
                "phone_number": phone,
                "profile": profile.dict()
            }
        }
    except Exception as e:
        logger.error(f"Error exporting data for {phone}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{phone}/data")
async def delete_user_data(
    phone: str,
    confirm_deletion: bool = Query(False)
):
    """Delete all stored data for a user (GDPR Right to be forgotten)"""
    try:
        if not confirm_deletion:
            raise HTTPException(status_code=400, detail="Must confirm deletion")
        
        profile_service = ProfileService()
        profile = await profile_service.get_or_create_profile(phone, auto_create=False)
        
        if profile:
            await profile_service.repository.delete_profile(
                profile.id,
                deleted_by="gdpr_api",
                reason="GDPR deletion request",
                hard_delete=True
            )
        
        return {
            "success": True,
            "message": "All personal data has been permanently deleted",
            "deletion_timestamp": datetime.now().astimezone().isoformat()
        }
    except Exception as e:
        logger.error(f"Error deleting data for {phone}: {e}")
        raise HTTPException(status_code=500, detail=str(e))