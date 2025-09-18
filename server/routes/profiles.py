"""
FastAPI endpoints for WhatsApp Profile Management System
RESTful API with proper error handling, versioning, and security
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Request
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import uuid

from ..services.profile_service import ProfileService, ProfileValidationError, ConsentRequiredError
from ..models.profiles import (
    Profile, ProfileCreate, ProfileUpdate, ProfileResponse, ProfileListResponse,
    ProfileStatsResponse, UpsertProfileRequest, MergeProfilesRequest,
    ProfileHistory, VersionConflictError
)
from ..tools.profile_tools import LLMProfileTools

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/profiles", tags=["profiles"])

# Dependencies
async def get_profile_service() -> ProfileService:
    """Dependency to get profile service instance"""
    return ProfileService()

async def get_llm_tools() -> LLMProfileTools:
    """Dependency to get LLM tools instance"""
    return LLMProfileTools()

# ============================================================================
# PROFILE CRUD ENDPOINTS
# ============================================================================

@router.get("/{phone}", response_model=ProfileResponse)
async def get_profile(
    phone: str = Path(..., description="Phone number in E.164 format"),
    include_stats: bool = Query(False, description="Include profile statistics"),
    service: ProfileService = Depends(get_profile_service)
):
    """Get profile by phone number"""
    try:
        profile = await service.get_or_create_profile(phone, auto_create=False)
        
        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Profile not found for phone number: {phone}"
            )
        
        response_data = {"profile": profile}
        
        if include_stats:
            completeness = await service.analyze_profile_completeness(profile)
            missing_fields = await service.get_missing_fields(phone)
            suggestions = await service.suggest_profile_improvements(profile)
            
            response_data.update({
                "completeness": completeness,
                "missing_fields": missing_fields,
                "suggestions": suggestions
            })
        
        return ProfileResponse(
            success=True,
            profile=profile,
            message="Profile retrieved successfully",
            **response_data
        )
        
    except Exception as e:
        logger.error(f"Failed to get profile {phone}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/", response_model=ProfileResponse)
async def create_profile(
    profile_data: ProfileCreate,
    request: Request,
    service: ProfileService = Depends(get_profile_service)
):
    """Create a new profile"""
    try:
        # Check if profile already exists
        existing_profile = await service.get_or_create_profile(
            profile_data.phone, auto_create=False
        )
        
        if existing_profile:
            raise HTTPException(
                status_code=409,
                detail=f"Profile already exists for phone number: {profile_data.phone}"
            )
        
        # Create profile
        profile = await service.repository.create_profile(
            profile_data, created_by=f"api_{request.client.host}"
        )
        
        return ProfileResponse(
            success=True,
            profile=profile,
            message="Profile created successfully"
        )
        
    except ProfileValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/{phone}", response_model=ProfileResponse)
async def update_profile(
    phone: str = Path(..., description="Phone number in E.164 format"),
    update_data: ProfileUpdate = None,
    expected_version: Optional[int] = Query(None, description="Expected version for optimistic locking"),
    reason: str = Query("Profile updated via API", description="Reason for update"),
    request: Request = None,
    service: ProfileService = Depends(get_profile_service)
):
    """Update profile with optimistic locking"""
    try:
        # Get existing profile
        profile = await service.get_or_create_profile(phone)
        
        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Profile not found for phone number: {phone}"
            )
        
        # Create upsert request
        upsert_request = UpsertProfileRequest(
            phone=phone,
            fields=update_data.dict(exclude_unset=True),
            reason=reason,
            expected_version=expected_version
        )
        
        # Update profile
        response = await service.upsert_profile(
            upsert_request, actor=f"api_{request.client.host}"
        )
        
        if not response.success:
            if "Version conflict" in response.message:
                raise HTTPException(status_code=409, detail=response.message)
            elif "Consent required" in response.message:
                raise HTTPException(status_code=403, detail=response.message)
            else:
                raise HTTPException(status_code=422, detail=response.message)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update profile {phone}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{phone}")
async def delete_profile(
    phone: str = Path(..., description="Phone number in E.164 format"),
    hard_delete: bool = Query(False, description="Permanently delete (cannot be undone)"),
    reason: str = Query("Profile deleted via API", description="Reason for deletion"),
    request: Request = None,
    service: ProfileService = Depends(get_profile_service)
):
    """Delete profile (soft delete by default)"""
    try:
        profile = await service.get_or_create_profile(phone, auto_create=False)
        
        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Profile not found for phone number: {phone}"
            )
        
        success = await service.repository.delete_profile(
            profile.id,
            deleted_by=f"api_{request.client.host}",
            reason=reason,
            hard_delete=hard_delete
        )
        
        if success:
            return {
                "success": True,
                "message": f"Profile {'permanently deleted' if hard_delete else 'deleted'} successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete profile")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete profile {phone}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PROFILE MANAGEMENT ENDPOINTS
# ============================================================================

@router.post("/upsert", response_model=ProfileResponse)
async def upsert_profile(
    request_data: UpsertProfileRequest,
    request: Request,
    service: ProfileService = Depends(get_profile_service)
):
    """Create or update profile with business logic"""
    try:
        response = await service.upsert_profile(
            request_data, actor=f"api_{request.client.host}"
        )
        
        if not response.success:
            if "Version conflict" in response.message:
                raise HTTPException(status_code=409, detail=response.message)
            elif "Consent required" in response.message:
                raise HTTPException(status_code=403, detail=response.message)
            elif "Validation error" in response.message:
                raise HTTPException(status_code=422, detail=response.message)
            else:
                raise HTTPException(status_code=400, detail=response.message)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upsert profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/merge", response_model=ProfileResponse)
async def merge_profiles(
    merge_request: MergeProfilesRequest,
    request: Request,
    service: ProfileService = Depends(get_profile_service)
):
    """Merge duplicate profiles"""
    try:
        response = await service.merge_profiles(
            merge_request, actor=f"api_{request.client.host}"
        )
        
        if not response.success:
            raise HTTPException(status_code=400, detail=response.message)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to merge profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{phone}/consent")
async def update_consent(
    phone: str = Path(..., description="Phone number in E.164 format"),
    granted: bool = Query(..., description="Whether consent is granted"),
    consent_type: str = Query("memory_storage", description="Type of consent"),
    method: str = Query("api_call", description="Method of consent collection"),
    request: Request = None,
    service: ProfileService = Depends(get_profile_service)
):
    """Update user consent"""
    try:
        from ..models.profiles import ConsentType, ConsentMethod
        
        # Map string to enum
        consent_type_enum = ConsentType(consent_type)
        consent_method_enum = ConsentMethod(method)
        
        response = await service.update_consent(
            phone=phone,
            consent_type=consent_type_enum,
            granted=granted,
            method=consent_method_enum,
            actor=f"api_{request.client.host}"
        )
        
        if not response.success:
            raise HTTPException(status_code=400, detail=response.message)
        
        return {
            "success": True,
            "message": response.message,
            "consent_granted": granted
        }
        
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid enum value: {e}")
    except Exception as e:
        logger.error(f"Failed to update consent for {phone}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PROFILE SEARCH AND LISTING
# ============================================================================

@router.get("/", response_model=ProfileListResponse)
async def list_profiles(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(50, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search query"),
    persona: Optional[str] = Query(None, description="Filter by persona"),
    consent: Optional[bool] = Query(None, description="Filter by consent status"),
    language: Optional[str] = Query(None, description="Filter by language"),
    include_deleted: bool = Query(False, description="Include deleted profiles"),
    service: ProfileService = Depends(get_profile_service)
):
    """List profiles with search and filtering"""
    try:
        filters = {}
        if persona:
            filters['persona'] = persona
        if consent is not None:
            filters['consent'] = consent
        if language:
            filters['language'] = language
        if include_deleted:
            filters['include_deleted'] = include_deleted
        
        response = await service.search_profiles(
            query=search,
            filters=filters,
            page=page,
            per_page=per_page
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to list profiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/advanced")
async def advanced_search(
    q: str = Query(..., description="Search query"),
    fields: List[str] = Query(["name", "phone", "description"], description="Fields to search"),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    service: ProfileService = Depends(get_profile_service)
):
    """Advanced profile search"""
    try:
        from ..repositories.profile_repository import QueryOptions
        
        options = QueryOptions(
            limit=per_page,
            offset=(page - 1) * per_page
        )
        
        profiles = await service.repository.search_profiles(
            query=q,
            fields=fields,
            options=options
        )
        
        return ProfileListResponse(
            success=True,
            profiles=profiles,
            total=len(profiles),
            page=page,
            per_page=per_page,
            message=f"Found {len(profiles)} profiles matching '{q}'"
        )
        
    except Exception as e:
        logger.error(f"Failed advanced search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PROFILE ANALYTICS
# ============================================================================

@router.get("/analytics/stats", response_model=ProfileStatsResponse)
async def get_profile_statistics(
    service: ProfileService = Depends(get_profile_service)
):
    """Get comprehensive profile statistics"""
    try:
        stats = await service.get_profile_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get profile statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{phone}/completeness")
async def get_profile_completeness(
    phone: str = Path(..., description="Phone number in E.164 format"),
    service: ProfileService = Depends(get_profile_service)
):
    """Analyze profile completeness"""
    try:
        profile = await service.get_or_create_profile(phone, auto_create=False)
        
        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Profile not found for phone number: {phone}"
            )
        
        completeness = await service.analyze_profile_completeness(profile)
        missing_fields = await service.get_missing_fields(phone)
        suggestions = await service.suggest_profile_improvements(profile)
        
        return {
            "success": True,
            "phone": phone,
            "completeness": completeness,
            "missing_fields": missing_fields,
            "suggestions": suggestions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze completeness for {phone}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PROFILE HISTORY AND AUDIT
# ============================================================================

@router.get("/{phone}/history")
async def get_profile_history(
    phone: str = Path(..., description="Phone number in E.164 format"),
    limit: int = Query(50, ge=1, le=100, description="Number of history entries"),
    change_type: Optional[str] = Query(None, description="Filter by change type"),
    service: ProfileService = Depends(get_profile_service)
):
    """Get profile change history"""
    try:
        profile = await service.get_or_create_profile(phone, auto_create=False)
        
        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Profile not found for phone number: {phone}"
            )
        
        # Filter by change type if specified
        change_types = None
        if change_type:
            from ..models.profiles import ChangeType
            try:
                change_types = [ChangeType(change_type)]
            except ValueError:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid change type: {change_type}"
                )
        
        history = await service.repository.get_profile_history(
            profile.id, limit=limit, change_types=change_types
        )
        
        return {
            "success": True,
            "phone": phone,
            "profile_id": str(profile.id),
            "history": [h.dict() for h in history],
            "total_entries": len(history)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get history for {phone}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{phone}/revert/{history_id}")
async def revert_profile(
    phone: str = Path(..., description="Phone number in E.164 format"),
    history_id: str = Path(..., description="History entry ID to revert to"),
    request: Request = None,
    service: ProfileService = Depends(get_profile_service)
):
    """Revert profile to previous state"""
    try:
        profile = await service.get_or_create_profile(phone, auto_create=False)
        
        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Profile not found for phone number: {phone}"
            )
        
        try:
            history_uuid = uuid.UUID(history_id)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="Invalid history ID format"
            )
        
        reverted_profile = await service.repository.revert_profile(
            profile.id, history_uuid, reverted_by=f"api_{request.client.host}"
        )
        
        return ProfileResponse(
            success=True,
            profile=reverted_profile,
            message=f"Profile reverted to history entry {history_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to revert profile {phone}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# LLM INTEGRATION ENDPOINTS
# ============================================================================

@router.post("/llm/get-profile")
async def llm_get_profile(
    phone: str,
    tools: LLMProfileTools = Depends(get_llm_tools)
):
    """LLM tool: Get profile information"""
    try:
        result = await tools.get_profile(phone)
        return result.dict()
    except Exception as e:
        logger.error(f"LLM get_profile failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/llm/upsert-profile")
async def llm_upsert_profile(
    phone: str,
    fields: Dict[str, Any],
    reason: str,
    expected_version: Optional[int] = None,
    tools: LLMProfileTools = Depends(get_llm_tools)
):
    """LLM tool: Upsert profile"""
    try:
        result = await tools.upsert_profile(
            phone=phone,
            fields=fields,
            reason=reason,
            expected_version=expected_version
        )
        return result.dict()
    except Exception as e:
        logger.error(f"LLM upsert_profile failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/llm/missing-fields")
async def llm_query_missing_fields(
    phone: str,
    tools: LLMProfileTools = Depends(get_llm_tools)
):
    """LLM tool: Query missing fields"""
    try:
        result = await tools.query_missing_fields(phone)
        return result.dict()
    except Exception as e:
        logger.error(f"LLM query_missing_fields failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health")
async def health_check(service: ProfileService = Depends(get_profile_service)):
    """Health check for profile system"""
    try:
        # Check database connectivity
        health = await service.repository.db.health_check()
        
        return {
            "status": "healthy" if health["connected"] else "unhealthy",
            "timestamp": datetime.now().astimezone().isoformat(),
            "database": health,
            "service": "profile_management"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().astimezone().isoformat(),
            "error": str(e),
            "service": "profile_management"
        }