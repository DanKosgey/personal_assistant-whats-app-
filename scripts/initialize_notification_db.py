#!/usr/bin/env python3
"""
Script to initialize the database schema for the autonomous notification system.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.services.processor_modules.notification_db_schema import initialize_notification_db_schema
from server.database import db_manager

def main():
    """Initialize the notification database schema."""
    print("Initializing notification database schema...")
    
    try:
        # Initialize the schema
        success = initialize_notification_db_schema(db_manager)
        
        if success:
            print("✅ Notification database schema initialized successfully!")
            print("\nCollections created:")
            print("  - conversations")
            print("  - summaries")
            print("  - notifications")
            print("  - notification_idempotency")
            print("  - notification_feedback")
            print("  - owner_preferences")
        else:
            print("❌ Failed to initialize notification database schema")
            return 1
            
    except Exception as e:
        print(f"❌ Error initializing notification database schema: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())