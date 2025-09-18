import asyncio
import asyncpg
import os
from server.db.profile_db import profile_db

async def check_profiles():
    """Check what's in the database"""
    try:
        # Initialize database connection
        await profile_db.initialize()
        
        if not profile_db.pool:
            print("❌ Database not available")
            return
        
        print("✅ Database connection established")
        
        # Check tables that exist
        tables = await profile_db.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        print(f"📊 Tables in database: {[t['table_name'] for t in tables]}")
        
        # Check for profiles in both possible tables
        try:
            # Check simplified users table
            users_count = await profile_db.fetchval("SELECT COUNT(*) FROM users")
            print(f"👥 Users table count: {users_count}")
            
            if users_count > 0:
                users = await profile_db.fetch("SELECT * FROM users ORDER BY created_at DESC LIMIT 10")
                print("📋 Recent users:")
                for user in users:
                    print(f"  - Phone: {user['phone']}, Name: {user['name']}, Created: {user['created_at']}")
        except Exception as e:
            print(f"ℹ️ Users table not found or error: {e}")
        
        try:
            # Check full profiles table
            profiles_count = await profile_db.fetchval("SELECT COUNT(*) FROM profiles")
            print(f"👤 Profiles table count: {profiles_count}")
            
            if profiles_count > 0:
                profiles = await profile_db.fetch("SELECT phone, status, id, created_at FROM profiles ORDER BY created_at DESC LIMIT 10")
                print("📋 Recent profiles:")
                for profile in profiles:
                    print(f"  - Phone: {profile['phone']}, Status: {profile['status']}, ID: {profile['id']}, Created: {profile['created_at']}")
                    
                # Check specifically for the problematic phone number
                specific_profiles = await profile_db.fetch("SELECT * FROM profiles WHERE phone = $1", "+254702944890")
                print(f"📱 Profiles matching +254702944890: {len(specific_profiles)}")
                for profile in specific_profiles:
                    print(f"  - Phone: {profile['phone']}, Status: {profile['status']}, ID: {profile['id']}, Created: {profile['created_at']}")
        except Exception as e:
            print(f"ℹ️ Profiles table not found or error: {e}")
            
        # Check migrations
        try:
            migrations = await profile_db.fetch("SELECT * FROM migrations ORDER BY id")
            print(f"💾 Applied migrations ({len(migrations)}):")
            for migration in migrations:
                print(f"  - {migration['filename']} (applied: {migration['applied_at']})")
        except Exception as e:
            print(f"ℹ️ Migrations table not found or error: {e}")
            
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if profile_db.pool:
            await profile_db.close()
            print("🔒 Database connection closed")

if __name__ == "__main__":
    asyncio.run(check_profiles())