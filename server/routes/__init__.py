from fastapi import APIRouter

router = APIRouter()

from .webhook import router as webhook_router
from .personas import router as personas_router
from .messages import router as messages_router

router.include_router(webhook_router)
router.include_router(personas_router)
router.include_router(messages_router)

__all__ = ["router"]
