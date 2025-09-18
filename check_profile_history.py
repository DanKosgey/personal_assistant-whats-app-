import asyncio
import logging
import sys
import os
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.db.profile_db import get_profile_db

async def check_profile_history():
    try:
        db = await get_profile_db()
        print("Database connection established")
        
        # Get the profile ID for the phone number with name "Hi"
        profiles = await db.fetch("SELECT * FROM profiles WHERE phone = '+254702944890'")
        if not profiles:
            print("No profile found with phone +254702944890")
            return
            
        profile_id = profiles[0]['id']
        print(f"Profile ID: {profile_id}")
        
        # Check the profile history
        history = await db.fetch("SELECT * FROM profile_history WHERE profile_id = $1 ORDER BY created_at ASC", profile_id)
        print(f"Found {len(history)} history records:")
        for record in history:
            print(f"  ID: {record['id']}")
            print(f"  Change type: {record['change_type']}")
            print(f"  Changed by: {record['changed_by']}")
            print(f"  Reason: {record['reason']}")
            print(f"  Created at: {record['created_at']}")
            
            # Try to parse change_data
            try:
                change_data = record['change_data']
                if isinstance(change_data, str):
                    change_data = json.loads(change_data)
                print(f"  Change data: {change_data}")
            except Exception as e:
                print(f"  Change data: {record['change_data']} (Error parsing: {e})")
            print()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_profile_history())