"""
Pydantic models for the WhatsApp Profile Management System
Comprehensive type-safe models with validation and serialization
"""

from pydantic import BaseModel, Field, validator, EmailStr
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum
import re
import uuid

class ProfileStatus(str, Enum):
    """Profile status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MERGED = "merged"
    DELETED = "deleted"

class ChangeType(str, Enum):
    """Audit log change type enumeration"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    CONSENT = "consent"
    REVERT = "revert"

class SummaryType(str, Enum):
    """Profile summary type enumeration"""
    CONVERSATION = "conversation"
    PERSONALITY = "personality"
    PREFERENCES = "preferences"
    BEHAVIOR = "behavior"
    INTERESTS = "interests"

class ConversationState(str, Enum):
    """Conversation state enumeration"""
    ACTIVE = "active"
    PENDING_END = "pending_end"
    ENDED = "ended"
    REOPENED = "reopened"

class RelationshipType(str, Enum):
    """Profile relationship type enumeration"""
    FAMILY = "family"
    COLLEAGUE = "colleague"
    DUPLICATE = "duplicate"
    ALIAS = "alias"
    BUSINESS = "business"
    FRIEND = "friend"

class ConsentType(str, Enum):
    """Consent type enumeration for GDPR compliance"""
    MEMORY_STORAGE = "memory_storage"
    DATA_PROCESSING = "data_processing"
    MARKETING = "marketing"
    ANALYTICS = "analytics"
    SHARING = "sharing"

class ConsentMethod(str, Enum):
    """Method by which consent was obtained"""
    WHATSAPP_MESSAGE = "whatsapp_message"
    API_CALL = "api_call"
    WEB_FORM = "web_form"
    PHONE_CALL = "phone_call"
    EMAIL = "email"

# ============================================================================
# BASE MODELS
# ============================================================================

class TimestampMixin(BaseModel):
    """Mixin for models with timestamps"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

class UUIDMixin(BaseModel):
    """Mixin for models with UUID primary keys"""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)

# ============================================================================
# CORE PROFILE MODELS
# ============================================================================

class ProfileBase(BaseModel):
    """Base profile model with common fields"""
    phone: str = Field(..., min_length=7, max_length=32, description="Phone number in E.164 format")
    name: Optional[str] = Field(None, max_length=100, description="User's display name")
    display_name: Optional[str] = Field(None, max_length=100, description="Name used by bot")
    persona: Optional[str] = Field(None, max_length=64, description="User archetype")
    description: Optional[str] = Field(None, max_length=1000, description="AI-generated user summary")
    language: str = Field(default="en", max_length=8, description="ISO 639-1 language code")
    timezone: Optional[str] = Field(None, max_length=64, description="IANA timezone identifier")
    consent: bool = Field(default=False, description="Memory storage consent")
    consent_date: Optional[datetime] = Field(None, description="When consent was given/revoked")
    last_seen: Optional[datetime] = Field(None, description="Last interaction timestamp")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Custom key-value attributes")
    tags: List[str] = Field(default_factory=list, description="Profile tags")
    updated_by: str = Field(default="system", description="Who updated this profile")

    @validator('phone')
    def validate_phone(cls, v):
        """Validate phone number format (E.164)"""
        if not re.match(r'^\+?[1-9]\d{6,14}$', v):
            raise ValueError('Phone number must be in valid E.164 format')
        return v.strip()

    @validator('language')
    def validate_language(cls, v):
        """Validate language code format"""
        if not re.match(r'^[a-z]{2}(-[A-Z]{2})?$', v):
            raise ValueError('Language must be valid ISO 639-1 code')
        return v.lower()

    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags list"""
        if len(v) > 20:
            raise ValueError('Maximum 20 tags allowed')
        return [tag.strip().lower() for tag in v if tag.strip()]

    @validator('attributes')
    def validate_attributes(cls, v):
        """Validate attributes dictionary"""
        if len(v) > 50:
            raise ValueError('Maximum 50 attributes allowed')
        return v

class ProfileCreate(ProfileBase):
    """Model for creating new profiles"""
    pass

class ProfileUpdate(BaseModel):
    """Model for updating existing profiles"""
    name: Optional[str] = Field(None, max_length=100)
    display_name: Optional[str] = Field(None, max_length=100)
    persona: Optional[str] = Field(None, max_length=64)
    description: Optional[str] = Field(None, max_length=1000)
    language: Optional[str] = Field(None, max_length=8)
    timezone: Optional[str] = Field(None, max_length=64)
    consent: Optional[bool] = None
    attributes: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    updated_by: Optional[str] = Field(None, max_length=100)

    @validator('language')
    def validate_language(cls, v):
        if v and not re.match(r'^[a-z]{2}(-[A-Z]{2})?$', v):
            raise ValueError('Language must be valid ISO 639-1 code')
        return v.lower() if v else v

    @validator('tags')
    def validate_tags(cls, v):
        if v and len(v) > 20:
            raise ValueError('Maximum 20 tags allowed')
        return [tag.strip().lower() for tag in v if tag.strip()] if v else v

class Profile(ProfileBase, UUIDMixin, TimestampMixin):
    """Complete profile model with all fields"""
    status: ProfileStatus = Field(default=ProfileStatus.ACTIVE)
    version: int = Field(default=1, description="Optimistic locking version")
    merged_into: Optional[uuid.UUID] = Field(None, description="Profile this was merged into")
    deleted_at: Optional[datetime] = Field(None, description="Soft delete timestamp")

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }

# ============================================================================
# PROFILE HISTORY MODELS
# ============================================================================

class ProfileHistoryBase(BaseModel):
    """Base model for profile history entries"""
    profile_id: uuid.UUID = Field(..., description="Reference to profile")
    changed_by: str = Field(..., max_length=100, description="Who made the change")
    change_type: ChangeType = Field(..., description="Type of change")
    change_data: Dict[str, Any] = Field(..., description="Change details")
    reason: Optional[str] = Field(None, max_length=500, description="Reason for change")
    session_id: Optional[str] = Field(None, max_length=100, description="Session identifier")
    ip_address: Optional[str] = Field(None, description="IP address of request")
    user_agent: Optional[str] = Field(None, max_length=500, description="User agent string")

class ProfileHistoryCreate(ProfileHistoryBase):
    """Model for creating history entries"""
    pass

class ProfileHistory(ProfileHistoryBase, UUIDMixin):
    """Complete profile history model"""
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }

# ============================================================================
# PROFILE SUMMARY MODELS
# ============================================================================

class ProfileSummaryBase(BaseModel):
    """Base model for profile summaries"""
    profile_id: uuid.UUID = Field(..., description="Reference to profile")
    summary: str = Field(..., min_length=10, max_length=2000, description="AI-generated summary")
    summary_type: SummaryType = Field(default=SummaryType.CONVERSATION)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="AI confidence")
    source_messages: List[str] = Field(default_factory=list, description="Source message IDs")
    created_by: str = Field(default="ai_summarizer", max_length=100)
    expires_at: Optional[datetime] = Field(None, description="Summary expiry")

class ProfileSummaryCreate(ProfileSummaryBase):
    """Model for creating summaries"""
    pass

class ProfileSummary(ProfileSummaryBase, UUIDMixin):
    """Complete profile summary model"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }

# ============================================================================
# PROFILE RELATIONSHIP MODELS
# ============================================================================

class ProfileRelationshipBase(BaseModel):
    """Base model for profile relationships"""
    primary_profile_id: uuid.UUID = Field(..., description="Primary profile")
    related_profile_id: uuid.UUID = Field(..., description="Related profile")
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Relationship strength")
    notes: Optional[str] = Field(None, max_length=500, description="Additional context")
    created_by: str = Field(default="system", max_length=100)

    @validator('related_profile_id')
    def no_self_reference(cls, v, values):
        """Prevent self-referencing relationships"""
        if 'primary_profile_id' in values and v == values['primary_profile_id']:
            raise ValueError('Cannot create relationship with self')
        return v

class ProfileRelationshipCreate(ProfileRelationshipBase):
    """Model for creating relationships"""
    pass

class ProfileRelationship(ProfileRelationshipBase, UUIDMixin):
    """Complete profile relationship model"""
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }

# ============================================================================
# CONSENT MODELS
# ============================================================================

class ProfileConsentBase(BaseModel):
    """Base model for consent tracking"""
    profile_id: uuid.UUID = Field(..., description="Reference to profile")
    consent_type: ConsentType = Field(..., description="Type of consent")
    granted: bool = Field(..., description="Whether consent was granted")
    expires_at: Optional[datetime] = Field(None, description="Consent expiry")
    consent_method: Optional[ConsentMethod] = Field(None, description="How consent was obtained")
    ip_address: Optional[str] = Field(None, description="IP address for legal compliance")
    user_agent: Optional[str] = Field(None, max_length=500, description="User agent")
    legal_basis: Optional[str] = Field(None, max_length=64, description="GDPR legal basis")
    notes: Optional[str] = Field(None, max_length=500, description="Additional context")

class ProfileConsentCreate(ProfileConsentBase):
    """Model for creating consent records"""
    pass

class ProfileConsent(ProfileConsentBase, UUIDMixin):
    """Complete consent model"""
    granted_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ProfileResponse(BaseModel):
    """Standard profile API response"""
    success: bool = True
    profile: Optional[Profile] = None
    message: Optional[str] = None
    audit_id: Optional[uuid.UUID] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }

class ProfileListResponse(BaseModel):
    """Response for profile list endpoints"""
    success: bool = True
    profiles: List[Profile] = Field(default_factory=list)
    total: int = 0
    page: int = 1
    per_page: int = 50
    message: Optional[str] = None

class ProfileStatsResponse(BaseModel):
    """Response for profile statistics"""
    total_profiles: int = 0
    active_profiles: int = 0
    consented_profiles: int = 0
    recent_interactions: int = 0
    top_personas: List[Dict[str, Union[str, int]]] = Field(default_factory=list)
    top_languages: List[Dict[str, Union[str, int]]] = Field(default_factory=list)

class UpsertProfileRequest(BaseModel):
    """Request model for upserting profiles"""
    phone: str = Field(..., description="Phone number")
    fields: Dict[str, Any] = Field(..., description="Fields to update")
    reason: str = Field(..., min_length=5, max_length=200, description="Reason for change")
    expected_version: Optional[int] = Field(None, description="Expected version for optimistic locking")
    session_id: Optional[str] = Field(None, description="Session identifier")

class MergeProfilesRequest(BaseModel):
    """Request model for merging profiles"""
    primary_phone: str = Field(..., description="Phone number of primary profile")
    duplicate_phone: str = Field(..., description="Phone number of duplicate profile")
    reason: str = Field(..., min_length=5, max_length=200, description="Reason for merge")
    merge_strategy: str = Field(default="prefer_primary", description="How to handle conflicts")

class InquireFieldRequest(BaseModel):
    """Request model for field inquiries"""
    phone: str = Field(..., description="Phone number")
    field_name: str = Field(..., description="Field to inquire about")
    prompt_template: Optional[str] = Field(None, description="Custom prompt template")
    urgency: str = Field(default="normal", description="Inquiry urgency level")

# ============================================================================
# VALIDATION MODELS
# ============================================================================

class VersionConflictError(BaseModel):
    """Model for version conflict errors"""
    error: str = "version_conflict"
    expected_version: int
    current_version: int
    profile_id: uuid.UUID
    message: str = "Profile was modified by another process"

class ValidationError(BaseModel):
    """Model for validation errors"""
    error: str = "validation_error"
    field: str
    message: str
    value: Optional[Any] = None

# ============================================================================
# LLM TOOL MODELS
# ============================================================================

class LLMToolResponse(BaseModel):
    """Base response model for LLM tools"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class GetProfileToolResponse(LLMToolResponse):
    """Response for get_profile LLM tool"""
    profile: Optional[Profile] = None

class UpsertProfileToolResponse(LLMToolResponse):
    """Response for upsert_profile LLM tool"""
    profile: Optional[Profile] = None
    version: Optional[int] = None
    audit_id: Optional[uuid.UUID] = None

class MissingFieldsToolResponse(LLMToolResponse):
    """Response for query_missing_fields LLM tool"""
    missing_fields: List[str] = Field(default_factory=list)
    required_fields: List[str] = Field(default_factory=list)
    optional_fields: List[str] = Field(default_factory=list)

# Export all models
__all__ = [
    # Enums
    'ProfileStatus', 'ChangeType', 'SummaryType', 'RelationshipType', 
    'ConsentType', 'ConsentMethod',
    
    # Core models
    'Profile', 'ProfileCreate', 'ProfileUpdate',
    'ProfileHistory', 'ProfileHistoryCreate',
    'ProfileSummary', 'ProfileSummaryCreate',
    'ProfileRelationship', 'ProfileRelationshipCreate',
    'ProfileConsent', 'ProfileConsentCreate',
    
    # Request/Response models
    'ProfileResponse', 'ProfileListResponse', 'ProfileStatsResponse',
    'UpsertProfileRequest', 'MergeProfilesRequest', 'InquireFieldRequest',
    
    # Error models
    'VersionConflictError', 'ValidationError',
    
    # LLM tool models
    'LLMToolResponse', 'GetProfileToolResponse', 'UpsertProfileToolResponse',
    'MissingFieldsToolResponse'
]