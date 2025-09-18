"""
Memory Management API Routes
Provides endpoints for accessing and managing the advanced memory system
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# This will be injected by the main application
_message_processor = None

def set_message_processor(processor):
    """Set the message processor instance"""
    global _message_processor
    _message_processor = processor

@router.get("/memory/stats/{user_id}")
async def get_user_memory_stats(user_id: str):
    """Get comprehensive memory statistics for a user"""
    if not _message_processor:
        raise HTTPException(status_code=503, detail="Memory system not available")
    
    # Validate user_id
    if not user_id or len(user_id.strip()) == 0:
        raise HTTPException(status_code=400, detail="User ID cannot be empty")
    
    if len(user_id) > 50:
        raise HTTPException(status_code=400, detail="User ID too long (max 50 characters)")
    
    try:
        stats = await _message_processor.get_memory_stats(user_id)
        return {
            "status": "success",
            "user_id": user_id,
            "memory_stats": stats
        }
    except Exception as e:
        logger.exception(f"Error getting memory stats for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve memory stats")

@router.get("/memory/context/{user_id}")
async def get_user_context(user_id: str, limit: int = Query(default=10, ge=1, le=50)):
    """Get recent conversation context for a user"""
    if not _message_processor:
        raise HTTPException(status_code=503, detail="Memory system not available")
    
    try:
        # Get context from both memory systems
        legacy_context = await _message_processor.memory.get_recent_context(user_id, minutes=None)
        advanced_context = await _message_processor.advanced_memory.get_recent_context(user_id, limit=limit)
        
        # Get user summary
        user_summary = await _message_processor.advanced_memory.get_user_summary(user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "user_summary": user_summary,
            "recent_context": advanced_context[-limit:] if advanced_context else [],
            "context_count": len(advanced_context) if advanced_context else 0,
            "legacy_context_count": len(legacy_context) if legacy_context else 0
        }
    except Exception as e:
        logger.exception(f"Error getting context for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user context")

@router.get("/memory/search/{user_id}")
async def search_user_memories(
    user_id: str, 
    query: str = Query(min_length=2, max_length=200),
    limit: int = Query(default=5, ge=1, le=20),
    memory_types: Optional[str] = Query(default=None, description="Comma-separated list of memory types")
):
    """Search through a user's long-term memories"""
    if not _message_processor:
        raise HTTPException(status_code=503, detail="Memory system not available")
    
    try:
        from ..services.advanced_memory import MemoryType
        
        # Parse memory types filter
        type_filter = None
        if memory_types:
            try:
                type_names = [t.strip().upper() for t in memory_types.split(",")]
                type_filter = [MemoryType(name.lower()) for name in type_names if hasattr(MemoryType, name)]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid memory type: {e}")
        
        # Search memories
        memories = await _message_processor.advanced_memory.search_memories(
            user_id=user_id,
            query=query,
            memory_types=type_filter,
            limit=limit
        )
        
        # Convert to JSON-serializable format
        memory_results = []
        for memory in memories:
            memory_results.append({
                "id": memory.id,
                "type": memory.memory_type.value,
                "content": memory.content,
                "importance": memory.importance,
                "timestamp": memory.timestamp.isoformat(),
                "metadata": memory.metadata
            })
        
        return {
            "status": "success",
            "user_id": user_id,
            "query": query,
            "results_count": len(memory_results),
            "memories": memory_results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error searching memories for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to search memories")

@router.get("/memory/profile/{user_id}")
async def get_user_profile(user_id: str):
    """Get user profile and preferences"""
    if not _message_processor:
        raise HTTPException(status_code=503, detail="Memory system not available")
    
    try:
        profile = await _message_processor.advanced_memory.get_or_create_user_profile(user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "profile": profile
        }
    except Exception as e:
        logger.exception(f"Error getting profile for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user profile")

@router.get("/memory/system/status")
async def get_memory_system_status():
    """Get the status of the memory system"""
    if not _message_processor:
        return {
            "status": "unavailable",
            "message": "Memory system not initialized"
        }
    
    try:
        # Get some basic statistics
        total_users = len(_message_processor.advanced_memory.user_profiles)
        total_short_term = sum(len(messages) for messages in _message_processor.advanced_memory.short_term_memory.values())
        total_long_term = sum(len(memories) for memories in _message_processor.advanced_memory.long_term_memory.values())
        
        return {
            "status": "active",
            "memory_system": "advanced_dual_layer",
            "statistics": {
                "total_users": total_users,
                "total_short_term_messages": total_short_term,
                "total_long_term_memories": total_long_term,
                "embeddings_cached": len(_message_processor.advanced_memory.embeddings_cache)
            },
            "features": [
                "short_term_context",
                "long_term_persistent_memory", 
                "vector_search",
                "conversation_summarization",
                "entity_extraction",
                "cross_session_context",
                "user_profiling"
            ]
        }
    except Exception as e:
        logger.exception(f"Error getting memory system status: {e}")
        return {
            "status": "error",
            "message": "Failed to retrieve system status"
        }