"""
LLM Tools System - Safe interface for AI to interact with profile management
Provides structured tools for get_profile, upsert_profile, query_missing_fields, 
inquire_field, and merge_profiles with proper validation and safety
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import uuid

# Use absolute imports instead of relative imports
from server.services.profile_service import ProfileService, ProfileValidationError, ConsentRequiredError
from server.models.profiles import (
    UpsertProfileRequest, MergeProfilesRequest, InquireFieldRequest,
    GetProfileToolResponse, UpsertProfileToolResponse, MissingFieldsToolResponse,
    LLMToolResponse, ConsentType, ConsentMethod
)

logger = logging.getLogger(__name__)

class LLMProfileTools:
    """LLM-safe tools for profile management with comprehensive validation"""
    
    def __init__(self, profile_service: Optional[ProfileService] = None):
        self.profile_service = profile_service or ProfileService()
        
        # Tool definitions for OpenAI function calling
        self.tool_definitions = self._get_tool_definitions()
    
    def _get_tool_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get OpenAI-compatible tool definitions"""
        return {
            "get_profile": {
                "name": "get_profile",
                "description": "Retrieve complete profile information for a phone number. Use this to check current profile state before making changes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone": {
                            "type": "string",
                            "description": "Phone number to lookup (E.164 format preferred)"
                        }
                    },
                    "required": ["phone"]
                }
            },
            "upsert_profile": {
                "name": "upsert_profile",
                "description": "Create or update profile fields. Always provide a clear reason. Check consent requirements for sensitive fields.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone": {
                            "type": "string",
                            "description": "Phone number (E.164 format preferred)"
                        },
                        "fields": {
                            "type": "object",
                            "description": "Fields to update (name, display_name, persona, description, language, timezone, attributes, tags)",
                            "properties": {
                                "name": {"type": "string", "maxLength": 100},
                                "display_name": {"type": "string", "maxLength": 100},
                                "persona": {"type": "string", "maxLength": 64},
                                "description": {"type": "string", "maxLength": 1000},
                                "language": {"type": "string", "pattern": "^[a-z]{2}(-[A-Z]{2})?$"},
                                "timezone": {"type": "string", "maxLength": 64},
                                "attributes": {"type": "object"},
                                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 20}
                            }
                        },
                        "reason": {
                            "type": "string",
                            "description": "Clear reason for the change (required for audit)",
                            "minLength": 5,
                            "maxLength": 200
                        },
                        "expected_version": {
                            "type": "integer",
                            "description": "Expected profile version for optimistic locking (optional)"
                        }
                    },
                    "required": ["phone", "fields", "reason"]
                }
            },
            "query_missing_fields": {
                "name": "query_missing_fields",
                "description": "Check which essential and optional fields are missing from a profile",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone": {
                            "type": "string",
                            "description": "Phone number to check"
                        }
                    },
                    "required": ["phone"]
                }
            },
            "inquire_field": {
                "name": "inquire_field",
                "description": "Send an inquiry message to user about a missing field. Use sparingly and only for essential fields.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone": {
                            "type": "string",
                            "description": "Phone number of user"
                        },
                        "field_name": {
                            "type": "string",
                            "description": "Name of field to inquire about",
                            "enum": ["name", "consent", "timezone", "language", "persona"]
                        },
                        "prompt_template": {
                            "type": "string",
                            "description": "Custom message template (optional)",
                            "maxLength": 500
                        },
                        "urgency": {
                            "type": "string",
                            "description": "Urgency level",
                            "enum": ["low", "normal", "high"],
                            "default": "normal"
                        }
                    },
                    "required": ["phone", "field_name"]
                }
            },
            "merge_profiles": {
                "name": "merge_profiles",
                "description": "Merge duplicate profiles. Use when you detect the same person has multiple profiles.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "primary_phone": {
                            "type": "string",
                            "description": "Phone number of primary profile to keep"
                        },
                        "duplicate_phone": {
                            "type": "string",
                            "description": "Phone number of duplicate profile to merge"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Reason for merging",
                            "minLength": 5,
                            "maxLength": 200
                        },
                        "merge_strategy": {
                            "type": "string",
                            "description": "How to handle conflicts",
                            "enum": ["prefer_primary", "prefer_duplicate", "prefer_longer", "combine"],
                            "default": "prefer_primary"
                        }
                    },
                    "required": ["primary_phone", "duplicate_phone", "reason"]
                }
            },
            "update_consent": {
                "name": "update_consent",
                "description": "Update user consent for memory storage. Critical for GDPR compliance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone": {
                            "type": "string",
                            "description": "Phone number of user"
                        },
                        "granted": {
                            "type": "boolean",
                            "description": "Whether consent is granted or revoked"
                        },
                        "method": {
                            "type": "string",
                            "description": "How consent was obtained",
                            "enum": ["whatsapp_message", "api_call", "web_form"],
                            "default": "whatsapp_message"
                        }
                    },
                    "required": ["phone", "granted"]
                }
            }
        }
    
    # ============================================================================
    # TOOL IMPLEMENTATIONS
    # ============================================================================
    
    async def get_profile(self, phone: str) -> GetProfileToolResponse:
        """Get profile information for a phone number"""
        try:
            logger.info(f"LLM tool: get_profile for {phone}")
            
            # Validate input
            if not phone or not isinstance(phone, str):
                return GetProfileToolResponse(
                    success=False,
                    message="Phone number is required and must be a string",
                    error="invalid_phone"
                )
            
            # Get profile
            profile = await self.profile_service.get_or_create_profile(phone, auto_create=False)
            
            if not profile:
                return GetProfileToolResponse(
                    success=True,
                    message=f"No profile found for {phone}",
                    profile=None,
                    data={"exists": False}
                )
            
            # Analyze profile completeness
            completeness = await self.profile_service.analyze_profile_completeness(profile)
            missing_fields = await self.profile_service.get_missing_fields(phone)
            
            return GetProfileToolResponse(
                success=True,
                message=f"Profile retrieved successfully",
                profile=profile,
                data={
                    "exists": True,
                    "completeness": completeness,
                    "missing_fields": missing_fields
                }
            )
            
        except Exception as e:
            logger.error(f"LLM tool get_profile failed: {e}")
            return GetProfileToolResponse(
                success=False,
                message=f"Failed to get profile: {str(e)}",
                error="get_profile_failed"
            )
    
    async def upsert_profile(
        self, 
        phone: str, 
        fields: Dict[str, Any], 
        reason: str,
        expected_version: Optional[int] = None,
        session_id: Optional[str] = None
    ) -> UpsertProfileToolResponse:
        """Create or update profile with validation"""
        try:
            logger.info(f"LLM tool: upsert_profile for {phone} with fields {list(fields.keys())}")
            
            # Validate inputs
            if not phone or not isinstance(phone, str):
                return UpsertProfileToolResponse(
                    success=False,
                    message="Phone number is required and must be a string",
                    error="invalid_phone"
                )
            
            if not fields or not isinstance(fields, dict):
                return UpsertProfileToolResponse(
                    success=False,
                    message="Fields must be a non-empty dictionary",
                    error="invalid_fields"
                )
            
            if not reason or len(reason.strip()) < 5:
                return UpsertProfileToolResponse(
                    success=False,
                    message="Reason is required and must be at least 5 characters",
                    error="invalid_reason"
                )
            
            # Create upsert request
            request = UpsertProfileRequest(
                phone=phone,
                fields=fields,
                reason=reason.strip(),
                expected_version=expected_version,
                session_id=session_id
            )
            
            # Execute upsert
            response = await self.profile_service.upsert_profile(request, actor="llm_agent")
            
            if response.success:
                return UpsertProfileToolResponse(
                    success=True,
                    message=response.message or "Profile updated successfully",
                    profile=response.profile,
                    version=response.profile.version if response.profile else None,
                    data={"updated_fields": list(fields.keys())}
                )
            else:
                return UpsertProfileToolResponse(
                    success=False,
                    message=response.message or "Failed to update profile",
                    error="upsert_failed",
                    profile=response.profile
                )
                
        except Exception as e:
            logger.error(f"LLM tool upsert_profile failed: {e}")
            return UpsertProfileToolResponse(
                success=False,
                message=f"Failed to upsert profile: {str(e)}",
                error="upsert_profile_failed"
            )
    
    async def query_missing_fields(self, phone: str) -> MissingFieldsToolResponse:
        """Query which fields are missing from a profile"""
        try:
            logger.info(f"LLM tool: query_missing_fields for {phone}")
            
            if not phone or not isinstance(phone, str):
                return MissingFieldsToolResponse(
                    success=False,
                    message="Phone number is required and must be a string",
                    error="invalid_phone"
                )
            
            # Get missing fields
            missing_fields = await self.profile_service.get_missing_fields(phone)
            
            # Get profile for additional context
            profile = await self.profile_service.get_or_create_profile(phone, auto_create=False)
            
            return MissingFieldsToolResponse(
                success=True,
                message=f"Found {missing_fields['total_missing']} missing fields",
                missing_fields=missing_fields['essential'] + missing_fields['optional'],
                required_fields=missing_fields['essential'],
                optional_fields=missing_fields['optional'],
                data={
                    "profile_exists": profile is not None,
                    "has_consent": profile.consent if profile else False,
                    "total_missing": missing_fields['total_missing']
                }
            )
            
        except Exception as e:
            logger.error(f"LLM tool query_missing_fields failed: {e}")
            return MissingFieldsToolResponse(
                success=False,
                message=f"Failed to query missing fields: {str(e)}",
                error="query_missing_fields_failed"
            )
    
    async def inquire_field(
        self, 
        phone: str, 
        field_name: str,
        prompt_template: Optional[str] = None,
        urgency: str = "normal"
    ) -> LLMToolResponse:
        """Send inquiry message to user about missing field"""
        try:
            logger.info(f"LLM tool: inquire_field {field_name} for {phone}")
            
            # Validate inputs
            if not phone or not isinstance(phone, str):
                return LLMToolResponse(
                    success=False,
                    message="Phone number is required and must be a string",
                    error="invalid_phone"
                )
            
            allowed_fields = ["name", "consent", "timezone", "language", "persona"]
            if field_name not in allowed_fields:
                return LLMToolResponse(
                    success=False,
                    message=f"Field '{field_name}' not allowed for inquiry. Allowed: {allowed_fields}",
                    error="invalid_field"
                )
            
            # Get profile to check if inquiry is needed
            profile = await self.profile_service.get_or_create_profile(phone)
            
            # Check if field should be inquired about
            if profile is not None:
                should_inquire = await self.profile_service.should_inquire_field(profile, field_name)
            else:
                should_inquire = False
            
            if not should_inquire:
                return LLMToolResponse(
                    success=False,
                    message=f"Inquiry not needed for field '{field_name}' (already filled or recently asked)",
                    error="inquiry_not_needed",
                    data={"field_value": getattr(profile, field_name, None)}
                )
            
            # Generate appropriate inquiry message
            inquiry_message = self._generate_inquiry_message(field_name, prompt_template, profile)
            
            # This would integrate with WhatsApp messaging system
            # For now, we'll return the message that should be sent
            return LLMToolResponse(
                success=True,
                message=f"Inquiry message generated for field '{field_name}'",
                data={
                    "field_name": field_name,
                    "inquiry_message": inquiry_message,
                    "urgency": urgency,
                    "should_send": True
                }
            )
            
        except Exception as e:
            logger.error(f"LLM tool inquire_field failed: {e}")
            return LLMToolResponse(
                success=False,
                message=f"Failed to create field inquiry: {str(e)}",
                error="inquire_field_failed"
            )
    
    async def merge_profiles(
        self,
        primary_phone: str,
        duplicate_phone: str,
        reason: str,
        merge_strategy: str = "prefer_primary"
    ) -> LLMToolResponse:
        """Merge duplicate profiles"""
        try:
            logger.info(f"LLM tool: merge_profiles {duplicate_phone} -> {primary_phone}")
            
            # Validate inputs
            if not all([primary_phone, duplicate_phone, reason]):
                return LLMToolResponse(
                    success=False,
                    message="Primary phone, duplicate phone, and reason are all required",
                    error="invalid_inputs"
                )
            
            if primary_phone == duplicate_phone:
                return LLMToolResponse(
                    success=False,
                    message="Cannot merge profile with itself",
                    error="same_profile"
                )
            
            if len(reason.strip()) < 5:
                return LLMToolResponse(
                    success=False,
                    message="Reason must be at least 5 characters",
                    error="invalid_reason"
                )
            
            # Create merge request
            request = MergeProfilesRequest(
                primary_phone=primary_phone,
                duplicate_phone=duplicate_phone,
                reason=reason.strip(),
                merge_strategy=merge_strategy
            )
            
            # Execute merge
            response = await self.profile_service.merge_profiles(request, actor="llm_agent")
            
            return LLMToolResponse(
                success=response.success,
                message=response.message or "Profile merge completed",
                data={
                    "primary_phone": primary_phone,
                    "duplicate_phone": duplicate_phone,
                    "merge_strategy": merge_strategy,
                    "merged_profile_id": str(response.profile.id) if response.profile else None
                },
                error=None if response.success else "merge_failed"
            )
            
        except Exception as e:
            logger.error(f"LLM tool merge_profiles failed: {e}")
            return LLMToolResponse(
                success=False,
                message=f"Failed to merge profiles: {str(e)}",
                error="merge_profiles_failed"
            )
    
    async def update_consent(
        self,
        phone: str,
        granted: bool,
        method: str = "whatsapp_message"
    ) -> LLMToolResponse:
        """Update user consent for memory storage"""
        try:
            logger.info(f"LLM tool: update_consent for {phone}, granted={granted}")
            
            if not phone or not isinstance(phone, str):
                return LLMToolResponse(
                    success=False,
                    message="Phone number is required and must be a string",
                    error="invalid_phone"
                )
            
            # Map method string to enum
            method_mapping = {
                "whatsapp_message": ConsentMethod.WHATSAPP_MESSAGE,
                "api_call": ConsentMethod.API_CALL,
                "web_form": ConsentMethod.WEB_FORM
            }
            
            consent_method = method_mapping.get(method, ConsentMethod.WHATSAPP_MESSAGE)
            
            # Update consent
            response = await self.profile_service.update_consent(
                phone=phone,
                consent_type=ConsentType.MEMORY_STORAGE,
                granted=granted,
                method=consent_method,
                actor="llm_agent"
            )
            
            return LLMToolResponse(
                success=response.success,
                message=response.message or "Consent updated successfully",
                data={
                    "phone": phone,
                    "consent_granted": granted,
                    "method": method,
                    "consent_type": "memory_storage"
                },
                error=None if response.success else "consent_update_failed"
            )
            
        except Exception as e:
            logger.error(f"LLM tool update_consent failed: {e}")
            return LLMToolResponse(
                success=False,
                message=f"Failed to update consent: {str(e)}",
                error="update_consent_failed"
            )
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _generate_inquiry_message(
        self, 
        field_name: str, 
        custom_template: Optional[str], 
        profile
    ) -> str:
        """Generate appropriate inquiry message for a field"""
        if custom_template:
            return custom_template
        
        # Default templates for different fields
        templates = {
            "name": "Hi! I'd love to know your name so I can address you properly. What should I call you?",
            "consent": "To provide you with a more personalized experience and remember our conversations, I'd like to ask for your permission to store our chat history. This helps me understand your preferences better. May I have your consent to remember our conversations?",
            "timezone": "What timezone are you in? This helps me provide better assistance with scheduling and time-sensitive information.",
            "language": "What's your preferred language for our conversations?",
            "persona": "To better assist you, could you tell me a bit about yourself? For example, are you a business owner, developer, trader, or something else?"
        }
        
        base_message = templates.get(field_name, f"Could you please provide your {field_name}?")
        
        # Personalize if we have a name
        if profile and profile.name and field_name != "name":
            base_message = f"Hi {profile.name}! {base_message}"
        
        return base_message
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Union[LLMToolResponse, GetProfileToolResponse, UpsertProfileToolResponse, MissingFieldsToolResponse]:
        """Execute a tool by name with parameters"""
        try:
            if tool_name == "get_profile":
                return await self.get_profile(**kwargs)
            elif tool_name == "upsert_profile":
                return await self.upsert_profile(**kwargs)
            elif tool_name == "query_missing_fields":
                return await self.query_missing_fields(**kwargs)
            elif tool_name == "inquire_field":
                return await self.inquire_field(**kwargs)
            elif tool_name == "merge_profiles":
                return await self.merge_profiles(**kwargs)
            elif tool_name == "update_consent":
                return await self.update_consent(**kwargs)
            else:
                return LLMToolResponse(
                    success=False,
                    message=f"Unknown tool: {tool_name}",
                    error="unknown_tool"
                )
                
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return LLMToolResponse(
                success=False,
                message=f"Tool execution failed: {str(e)}",
                error="tool_execution_failed"
            )
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get OpenAI function calling schema for a specific tool"""
        return self.tool_definitions.get(tool_name)
    
    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas for OpenAI function calling"""
        return list(self.tool_definitions.values())

# Global instance for easy access
llm_profile_tools = LLMProfileTools()