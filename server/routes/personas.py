from fastapi import APIRouter, HTTPException
from ..persona_manager import PersonaManager

router = APIRouter()

pm = PersonaManager()


@router.get("/personas")
async def list_personas():
    return {"personas": pm.list_personas(), "current": pm.current}


@router.post("/personas/select")
async def select_persona(payload: dict):
    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="name required")
    ok = pm.select_persona(name)
    if not ok:
        raise HTTPException(status_code=404, detail="persona not found")
    return {"selected": name}
