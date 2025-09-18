import asyncio
import asyncpg
import os
from server.db.profile_db import profile_db

async def check_user_profile():
    """Check the specific profile for +254702944890"""
    try:
        # Initialize database connection
        await profile_db.initialize()
        
        if not profile_db.pool:
            print("❌ Database not available")
            return
        
        print("✅ Database connection established")
        
        # Check the profile for +254702944890
        phone = "+254702944890"
        profile_query = "SELECT * FROM profiles WHERE phone = $1"
        profile = await profile_db.fetchrow(profile_query, phone)
        
        if not profile:
            print(f"❌ No profile found for {phone}")
            return
            
        print(f"📱 Profile for {phone}")
        print(f"  - ID: {profile['id']}")
        print(f"  - Name: {profile['name']}")
        print(f"  - Status: {profile['status']}")
        print(f"  - Created: {profile['created_at']}")
        print(f"  - Updated: {profile['updated_at']}")
        print(f"  - Last Seen: {profile['last_seen']}")
        print(f"  - Consent: {profile['consent']}")
        print(f"  - Language: {profile['language']}")
        print(f"  - Timezone: {profile['timezone']}")
        print(f"  - Persona: {profile['persona']}")
        print(f"  - Description: {profile['description']}")
        print(f"  - Display Name: {profile['display_name']}")
        print(f"  - Attributes: {profile['attributes']}")
        print(f"  - Tags: {profile['tags']}")
        
        # Also check conversations for this user
        try:
            conv_query = "SELECT * FROM conversations WHERE phone = $1 ORDER BY created_at DESC LIMIT 5"
            conversations = await profile_db.fetch(conv_query, phone)
            print(f"\n💬 Recent conversations ({len(conversations)}):")
            for conv in conversations:
                print(f"  - [{conv['created_at']}] {conv['direction']}: {conv['text'][:50]}...")
        except Exception as e:
            print(f"ℹ️ Conversations table check error: {e}")
            
    except Exception as e:
        print(f"❌ Error checking profile: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if profile_db.pool:
            await profile_db.close()
            print("🔒 Database connection closed")

if __name__ == "__main__":
    asyncio.run(check_user_profile())