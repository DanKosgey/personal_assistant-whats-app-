#!/usr/bin/env python3
"""
Force generation of a conversation summary for the most recent conversation and trigger owner notification.
"""
import os
import sys
import asyncio
from datetime import datetime, timezone

# Insert backend path
sys.path.insert(0, os.path.dirname(__file__))

from server import generate_and_log_summary, conversations_collection, contacts_collection

async def main():
    # Find a recent conversation
    conv = None
    try:
        if conversations_collection is None:
            print('No MongoDB connection available; cannot force summary')
            return
        conv = conversations_collection.find_one(sort=[('last_message_time', -1)])
        if not conv:
            print('No conversations found to summarize')
            return
        conv_id = conv['conversation_id']
        phone = conv['phone_number']
        contact = contacts_collection.find_one({'phone_number': phone}) or {}
        print(f'Forcing summary for conversation {conv_id} / {phone}')
        await generate_and_log_summary(conv_id, phone, contact)
        print('Done')
    except Exception as e:
        print('Error:', e)

if __name__ == '__main__':
    asyncio.run(main())
