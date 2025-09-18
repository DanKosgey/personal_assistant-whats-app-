from typing import Any, Dict


def call_search_tool(query: str, top_k: int = 3) -> Dict[str, Any]:
    # lightweight stub for a search tool
    return {"query": query, "results": []}


def call_calendar_tool(user_id: str, query: str) -> Dict[str, Any]:
    return {"available": True, "next_slot": "2025-09-01T10:00:00Z"}
