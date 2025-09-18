import asyncio
import logging
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.services.profile_service import ProfileService
from server.models.profiles import UpsertProfileRequest

async def fix_invalid_names():
    """Fix profiles with invalid names like 'Hi'"""
    try:
        # Initialize profile service
        profile_service = ProfileService()
        
        # Get all profiles
        search_result = await profile_service.search_profiles()
        if not search_result.success:
            print("Failed to get profiles")
            return
            
        profiles = search_result.profiles
        print(f"Found {len(profiles)} profiles to check")
        
        # Invalid names to fix
        invalid_names = ['Hi', 'Hello', 'Hey', 'Good']
        
        fixed_count = 0
        for profile in profiles:
            if profile.name in invalid_names:
                print(f"Fixing profile {profile.phone} with invalid name '{profile.name}'")
                
                # Create upsert request to remove the invalid name
                upsert_request = UpsertProfileRequest(
                    phone=profile.phone,
                    fields={'name': None},
                    reason="Fixed invalid name",
                    expected_version=None,
                    session_id=None
                )
                
                # Update the profile
                result = await profile_service.upsert_profile(upsert_request)
                if result.success:
                    print(f"Successfully fixed profile {profile.phone}")
                    fixed_count += 1
                else:
                    print(f"Failed to fix profile {profile.phone}: {result.message}")
        
        print(f"Fixed {fixed_count} profiles with invalid names")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(fix_invalid_names())