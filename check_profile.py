import asyncio
import logging
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.db.profile_db import get_profile_db

async def check_profile():
    try:
        db = await get_profile_db()
        print("Database connection established")
        
        # Check the specific profile with name "Hi"
        profiles = await db.fetch("SELECT * FROM profiles WHERE phone = '+254702944890'")
        print(f"Found {len(profiles)} profile(s) with phone +254702944890:")
        for profile in profiles:
            print(f"  ID: {profile['id']}")
            print(f"  Name: {profile['name']}")
            print(f"  Phone: {profile['phone']}")
            print(f"  Created at: {profile['created_at']}")
            print(f"  Updated at: {profile['updated_at']}")
            print(f"  Updated by: {profile['updated_by']}")
            print()
            
        # Check all profiles to see if there are others with similar issues
        all_profiles = await db.fetch("SELECT * FROM profiles ORDER BY created_at DESC LIMIT 10")
        print(f"Recent profiles (last 10):")
        for profile in all_profiles:
            print(f"  Phone: {profile['phone']}, Name: {profile['name']}, Updated by: {profile['updated_by']}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(check_profile())