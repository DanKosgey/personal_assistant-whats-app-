#!/usr/bin/env python3
"""
Script to initialize the PostgreSQL profiles database by running migrations.
"""

import sys
import os
import asyncio

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

def main():
    """Initialize the profiles database."""
    print("Initializing profiles database...")
    
    try:
        # Import and initialize the profile database
        from server.db.profile_db import profile_db
        import asyncio
        
        # Run the initialization
        asyncio.run(profile_db.initialize())
        print("✅ Profiles database initialized successfully!")
        
    except Exception as e:
        print(f"❌ Error initializing profiles database: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())