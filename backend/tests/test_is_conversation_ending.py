import asyncio
import sys
from datetime import datetime, timezone

# Ensure backend package is importable
sys.path.insert(0, "..")

from server import ai_handler, ContactInfo


async def run_test():
    # Simulate a short conversation history
    conversation = [
        {"conversation_id": "conv1", "is_from_user": True, "content": "Hello, I want to meet", "timestamp": datetime.now(timezone.utc).isoformat()},
        {"conversation_id": "conv1", "is_from_user": False, "content": "Sure — when works for you?", "timestamp": datetime.now(timezone.utc).isoformat()},
        {"conversation_id": "conv1", "is_from_user": True, "content": "Tomorrow at 10 am", "timestamp": datetime.now(timezone.utc).isoformat()},
        {"conversation_id": "conv1", "is_from_user": False, "content": "Got it. See you then.", "timestamp": datetime.now(timezone.utc).isoformat()},
        {"conversation_id": "conv1", "is_from_user": True, "content": "Ok bye", "timestamp": datetime.now(timezone.utc).isoformat()},
    ]

    contact = ContactInfo(phone_number="+254700000000", name="Dan")

    print("--- Heuristics (AI forced off) ---")
    # Force heuristics
    ai_available_backup = ai_handler.ai_available
    ai_handler.ai_available = False
    ending, reason = await ai_handler.is_conversation_ending(conversation, contact)
    print(f"ending={ending}, reason={reason}")

    # Restore AI flag and try real AI decision if available
    ai_handler.ai_available = ai_available_backup
    if ai_handler.ai_available:
        print("--- AI decision (may call external API) ---")
        try:
            ending_ai, reason_ai = await asyncio.wait_for(ai_handler.is_conversation_ending(conversation, contact), timeout=30)
            print(f"ending={ending_ai}, reason={reason_ai}")
        except Exception as e:
            print(f"AI decision failed or timed out: {e}")
    else:
        print("AI not available; skipping AI decision")


if __name__ == '__main__':
    asyncio.run(run_test())
