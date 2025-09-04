"""Utility functions for the WhatsApp AI Agent."""

from functools import wraps
from typing import Optional, Callable, Any, TypeVar, Union, Awaitable, cast
from fastapi.responses import JSONResponse
from pymongo.collection import Collection

T = TypeVar('T')
AsyncFunc = TypeVar('AsyncFunc', bound=Callable[..., Awaitable[Any]])

def check_collection(collection: Optional[Collection]) -> bool:
    """Safely check if a MongoDB collection is available."""
    return collection is not None

def safe_collection(collection_name: str) -> Callable[[AsyncFunc], AsyncFunc]:
    """Decorator to safely handle collection access."""
    def decorator(func: AsyncFunc) -> AsyncFunc:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            from server import db_manager
            collection = db_manager.collections.get(collection_name)
            if not check_collection(collection):
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "Database collection not available",
                        "collection": collection_name
                    }
                )
            # Pass the collection to the wrapped function
            return await func(*args, **{**kwargs, 'collection': collection})
        return cast(AsyncFunc, wrapper)
    return decorator
