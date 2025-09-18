from fastapi import APIRouter

router = APIRouter()

from .webhook import router as webhook_router
from .personas import router as personas_router
from .messages import router as messages_router
from .memory import router as memory_router
from .profiles import router as profiles_router
from .consent import router as consent_router

router.include_router(webhook_router)
router.include_router(personas_router)
router.include_router(messages_router)
router.include_router(memory_router)
router.include_router(profiles_router)
router.include_router(consent_router)

__all__ = ["router"]
