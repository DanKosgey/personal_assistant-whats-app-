from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from ..persona_manager import PersonaManager
import os

router = APIRouter()

# Initialize persona manager with environment configuration
personas_dir = os.getenv("PERSONAS_DIR", "server/personas")
default_personality = os.getenv("USER_PERSONALITY")
pm = PersonaManager(personas_dir=personas_dir, default_prompt=default_personality)


class PersonaCreate(BaseModel):
    name: str
    system_prompt: str
    description: Optional[str] = ""
    personality_traits: Optional[List[str]] = None
    capabilities: Optional[List[str]] = None
    tone: Optional[str] = "professional"
    language: Optional[str] = "English"
    response_style: Optional[str] = "concise"
    context_awareness: Optional[bool] = True
    custom_fields: Optional[Dict[str, Any]] = None


class PersonaUpdate(BaseModel):
    system_prompt: Optional[str] = None
    description: Optional[str] = None
    personality_traits: Optional[List[str]] = None
    capabilities: Optional[List[str]] = None
    tone: Optional[str] = None
    language: Optional[str] = None
    response_style: Optional[str] = None
    context_awareness: Optional[bool] = None
    custom_fields: Optional[Dict[str, Any]] = None


class PersonaSelect(BaseModel):
    name: str


@router.get("/personas")
async def list_personas(detailed: bool = Query(False, description="Include full persona details")):
    """List all available personas"""
    if detailed:
        return {
            "personas": pm.get_all_personas(),
            "current": pm.get_current_persona(),
            "stats": pm.get_persona_stats()
        }
    else:
        return {
            "personas": pm.list_personas(),
            "current": pm.get_current_persona(),
            "stats": pm.get_persona_stats()
        }


@router.get("/personas/{name}")
async def get_persona(name: str):
    """Get details of a specific persona"""
    persona = pm.get_persona_details(name)
    if not persona:
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")
    return {"persona": persona}


@router.post("/personas/select")
async def select_persona(payload: PersonaSelect):
    """Select a persona to use"""
    success = pm.select_persona(payload.name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Persona '{payload.name}' not found")
    return {
        "message": f"Selected persona: {payload.name}",
        "selected": payload.name,
        "current": pm.get_current_persona()
    }


@router.post("/personas")
async def create_persona(persona: PersonaCreate):
    """Create a new persona"""
    # Check if persona already exists
    if persona.name in pm.list_personas():
        raise HTTPException(status_code=409, detail=f"Persona '{persona.name}' already exists")
    
    # Prepare kwargs for additional fields
    kwargs = persona.custom_fields or {}
    
    success = pm.add_persona(
        name=persona.name,
        system_prompt=persona.system_prompt,
        description=persona.description or "",
        personality_traits=persona.personality_traits,
        capabilities=persona.capabilities,
        tone=persona.tone or "professional",
        language=persona.language or "English",
        response_style=persona.response_style or "concise",
        context_awareness=persona.context_awareness if persona.context_awareness is not None else True,
        **kwargs
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to create persona")
    
    return {
        "message": f"Created persona: {persona.name}",
        "persona": pm.get_persona_details(persona.name)
    }


@router.put("/personas/{name}")
async def update_persona(name: str, updates: PersonaUpdate):
    """Update an existing persona"""
    if name not in pm.list_personas():
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")
    
    # Prepare update dict, excluding None values
    update_data = {k: v for k, v in updates.dict().items() if v is not None}
    
    # Handle custom fields
    if updates.custom_fields:
        update_data.update(updates.custom_fields)
        del update_data['custom_fields']
    
    success = pm.update_persona(name, **update_data)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update persona")
    
    return {
        "message": f"Updated persona: {name}",
        "persona": pm.get_persona_details(name)
    }


@router.delete("/personas/{name}")
async def delete_persona(name: str):
    """Delete a persona"""
    if name not in pm.list_personas():
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")
    
    success = pm.delete_persona(name)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete persona")
    
    return {"message": f"Deleted persona: {name}"}


@router.post("/personas/reload")
async def reload_personas():
    """Reload all personas from disk"""
    try:
        pm.reload_personas()
        return {
            "message": "Personas reloaded successfully",
            "stats": pm.get_persona_stats()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload personas: {str(e)}")


@router.get("/personas/current/prompt")
async def get_current_system_prompt():
    """Get the system prompt of the current persona"""
    prompt = pm.get_system_prompt()
    current = pm.get_current_persona()
    
    if not prompt:
        # Fallback to environment variable
        prompt = os.getenv("USER_PERSONALITY", "You are a helpful AI assistant.")
        current = {"name": "default", "system_prompt": prompt}
    
    return {
        "system_prompt": prompt,
        "current_persona": current
    }
