#!/usr/bin/env python3
"""
Script to check the current state of the database and profiles table
"""

import asyncio
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def check_database_state():
    """Check the current state of the database"""
    try:
        # Import the database module
        from server.db.profile_db import get_profile_db
        
        # Get database instance
        db = await get_profile_db()
        
        if not db.pool:
            print("❌ No database connection pool available")
            return
            
        print("✅ Database connection pool is available")
        
        # Check if profiles table exists
        try:
            tables = await db.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
            table_names = [row['tablename'] for row in tables]
            print(f"📋 Tables in database: {table_names}")
            
            if 'profiles' in table_names:
                print("✅ Profiles table exists")
            else:
                print("❌ Profiles table does not exist")
                
            if 'users' in table_names:
                print("⚠️  Users table exists (may conflict with profiles)")
                
        except Exception as e:
            print(f"❌ Error checking tables: {e}")
            
        # Check migrations table
        try:
            migrations = await db.fetch(
                "SELECT filename, applied_at FROM migrations ORDER BY applied_at"
            )
            print(f"📋 Applied migrations ({len(migrations)}):")
            for migration in migrations:
                print(f"  - {migration['filename']} (applied: {migration['applied_at']})")
        except Exception as e:
            print(f"❌ Error checking migrations: {e}")
            
        # Check profiles data
        try:
            profile_count = await db.fetchval("SELECT COUNT(*) FROM profiles")
            print(f"📊 Total profiles: {profile_count}")
            
            if profile_count > 0:
                # Show sample profiles
                sample_profiles = await db.fetch(
                    "SELECT phone, status, created_at, name FROM profiles ORDER BY created_at DESC LIMIT 5"
                )
                print("📋 Sample profiles:")
                for profile in sample_profiles:
                    print(f"  - {profile['phone']} ({profile['status']}) - {profile['name'] or 'No name'}")
                    
                # Check for the specific phone number
                target_phone = "+254702944890"
                specific_profile = await db.fetchrow(
                    "SELECT phone, status, created_at, name FROM profiles WHERE phone = $1",
                    target_phone
                )
                if specific_profile:
                    print(f"🔍 Found target phone {target_phone}: {specific_profile}")
                else:
                    print(f"🔍 Target phone {target_phone} not found in database")
                    
        except Exception as e:
            print(f"❌ Error checking profiles: {e}")
            
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")

if __name__ == "__main__":
    print("🔍 Checking database state...")
    asyncio.run(check_database_state())
    print("✅ Database state check complete")