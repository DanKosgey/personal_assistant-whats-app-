import asyncio
import os
import sys
from datetime import datetime

# Ensure backend path is importable
sys.path.insert(0, os.path.dirname(__file__))

from server import MessageProcessor, ai_handler, config

async def run_sim():
    # Simulate Gemini exhaustion
    ai_handler.ai_available = False
    # Set next_retry_time so code path considers keys exhausted and will attempt OpenRouter fallback
    ai_handler.next_retry_time = datetime.now()
    # Build a fake webhook payload
    webhook = {
        "entry": [{
            "changes": [{
                "value": {
                    "messages": [{
                        "id": "msg_123",
                        "from": "+254700000000",
                        "timestamp": "2025-08-22T00:00:00Z",
                        "type": "text",
                        "text": {"body": "Habari, nina swali kuhusu ushauri wa biashara."}
                    }]
                }
            }]
        }]
    }

    proc = MessageProcessor()
    ok = await proc.process_message(webhook)
    print('Process result:', ok)

if __name__ == '__main__':
    asyncio.run(run_sim())
