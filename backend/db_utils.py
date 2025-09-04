"""Database utility functions for better type safety and error handling."""

from typing import Optional, Any, Dict, List, TypeVar, Generic
from pymongo.collection import Collection
from pymongo.results import InsertOneResult, UpdateResult, DeleteResult
from datetime import datetime

T = TypeVar('T')

class CollectionAccessor(Generic[T]):
    """Safe collection accessor with proper typing."""
    
    def __init__(self, collection: Optional[Collection]):
        self._collection: Optional[Collection] = collection
        
    def _check_collection(self) -> bool:
        """Check if collection is available and valid."""
        return bool(self._collection is not None)
    
    @property
    def is_available(self) -> bool:
        """Check if collection is available."""
        return self._check_collection()
    
    @property
    def collection(self) -> Optional[Collection]:
        """Get the underlying collection safely."""
        return self._collection if self._check_collection() else None
    
    async def insert_one(self, document: Dict[str, Any]) -> Optional[str]:
        """Safely insert one document."""
        if not self._check_collection():
            return None
            
        from server import asyncio
        result: InsertOneResult = await asyncio.to_thread(
            self._collection.insert_one,  # type: ignore
            document
        )
        return str(result.inserted_id)
    
    async def update_one(self, 
                        query: Dict[str, Any],
                        update: Dict[str, Any],
                        upsert: bool = False) -> Optional[UpdateResult]:
        """Safely update one document."""
        if not self._check_collection():
            return None
            
        from server import asyncio
        return await asyncio.to_thread(
            self._collection.update_one,  # type: ignore
            query,
            update,
            upsert=upsert
        )
    
    async def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Safely find one document."""
        if not self._check_collection():
            return None
            
        from server import asyncio
        return await asyncio.to_thread(
            self._collection.find_one,  # type: ignore
            query
        )
    
    async def find_many(self, 
                       query: Dict[str, Any],
                       sort_by: Optional[List[tuple]] = None,
                       limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Safely find multiple documents."""
        if not self._check_collection():
            return []
            
        from server import asyncio
        # We know collection exists because of _check_collection
        cursor = self._collection.find(query)  # type: ignore
        
        if sort_by:
            cursor = cursor.sort(sort_by)
        if limit:
            cursor = cursor.limit(limit)
            
        return await asyncio.to_thread(list, cursor)
    
    async def delete_many(self, query: Dict[str, Any]) -> Optional[DeleteResult]:
        """Safely delete multiple documents."""
        if not self._check_collection():
            return None
            
        from server import asyncio
        return await asyncio.to_thread(
            self._collection.delete_many,  # type: ignore
            query
        )
    
    async def count_documents(self, query: Dict[str, Any]) -> int:
        """Safely count documents."""
        if not self._check_collection():
            return 0
            
        from server import asyncio
        return await asyncio.to_thread(
            self._collection.count_documents,  # type: ignore
            query
        )
