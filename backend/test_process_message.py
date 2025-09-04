import asyncio
import os
import json
from datetime import datetime, timezone

# Ensure OWNER_WHATSAPP_NUMBER is set to a dummy to test notify path
os.environ['OWNER_WHATSAPP_NUMBER'] = os.environ.get('OWNER_WHATSAPP_NUMBER','+10000000000')

# Monkeypatch whatsapp_client.send_message to avoid external HTTP calls
from server import whatsapp_client, process_whatsapp_message, messages_collection, contacts_collection, conversations_collection

async def noop_send(to, message):
    print(f"MOCK send_message called -> to: {to}, message: {message[:80]}...")
    return {'mock': True}

whatsapp_client.send_message = noop_send

# Build a sample webhook payload
payload = {
    'entry': [{
        'changes': [{
            'value': {
                'messages': [{
                    'from': '15551234567',
                    'id': 'testmsg1',
                    'type': 'text',
                    'text': {'body': 'Hi, my name is Alice and I have a business proposal'},
                    'timestamp': str(int(datetime.now(timezone.utc).timestamp()))
                }]
            }
        }]
    }]
}

async def run_test():
    await process_whatsapp_message(payload)

if __name__ == '__main__':
    asyncio.run(run_test())
    print('process_whatsapp_message finished (OK)')
