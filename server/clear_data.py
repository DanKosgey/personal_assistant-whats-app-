#!/usr/bin/env python3
"""
Simple data clearing script for WhatsApp AI Agent
Clears all data from collections/tables without dropping databases
"""

import os
import sys
import asyncio
import logging

# Add the server directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def clear_mongodb_data() -> bool:
    """Clear all data from MongoDB collections"""
    db_manager = None
    try:
        from server.database import db_manager
        
        # Connect to database
        await db_manager.connect()
        
        if not db_manager.db:
            logger.error("Failed to connect to MongoDB")
            return False
            
        # List of collections to clear
        collections_to_clear = [
            'conversations',
            'messages',
            'profiles',
            'contacts',
            'user_memories',
            'conversation_summaries',
            'feedback',
            'analytics'
        ]
        
        # Clear each collection
        for collection_name in collections_to_clear:
            try:
                if hasattr(db_manager, '_using_motor') and db_manager._using_motor:
                    result = await db_manager.db[collection_name].delete_many({})
                else:
                    import asyncio
                    result = await asyncio.to_thread(db_manager.db[collection_name].delete_many, {})
                
                logger.info(f"Cleared MongoDB collection '{collection_name}': {result.deleted_count} documents deleted")
            except Exception as e:
                logger.warning(f"Failed to clear collection {collection_name}: {e}")
        
        logger.info("MongoDB data clearing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to clear MongoDB data: {e}")
        return False
    finally:
        if db_manager:
            try:
                await db_manager.close()
            except:
                pass

def clear_postgresql_data() -> bool:
    """Clear all data from PostgreSQL tables"""
    try:
        import psycopg2
        
        # Get database connection details from environment
        db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:devpass@localhost:5433/whatsapp_profiles')
        
        # Parse connection details
        import re
        match = re.match(r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', db_url)
        if not match:
            logger.error("Invalid DATABASE_URL format")
            return False
            
        user, password, host, port, database = match.groups()
        
        # Connect to the database
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        cursor = conn.cursor()
        
        # List of tables to clear
        tables_to_clear = [
            'profiles',
            'contacts',
            'conversations',
            'messages'
        ]
        
        # Clear each table
        for table_name in tables_to_clear:
            try:
                cursor.execute(f'DELETE FROM "{table_name}"')
                rows_deleted = cursor.rowcount
                logger.info(f"Cleared PostgreSQL table '{table_name}': {rows_deleted} rows deleted")
            except Exception as e:
                logger.warning(f"Failed to clear table {table_name}: {e}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("PostgreSQL data clearing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to clear PostgreSQL data: {e}")
        return False

async def main():
    """Main clear data function"""
    print(" WhatsApp AI Agent Data Clear Script ")
    print("=" * 40)
    print("This script will clear ALL data from collections/tables.")
    print("This action is IRREVERSIBLE and ALL data will be permanently deleted.")
    print()
    
    # Confirm with user
    response = input("Are you sure you want to proceed? Type 'YES' to confirm: ")
    if response != "YES":
        print("Data clearing cancelled.")
        return 1
    
    print("\nStarting data clearing...")
    
    # Clear MongoDB data
    print("\nClearing MongoDB data...")
    mongo_success = await clear_mongodb_data()
    if mongo_success:
        print("✅ MongoDB data cleared successfully")
    else:
        print("❌ MongoDB data clearing failed")
    
    # Clear PostgreSQL data
    print("\nClearing PostgreSQL data...")
    postgres_success = clear_postgresql_data()
    if postgres_success:
        print("✅ PostgreSQL data cleared successfully")
    else:
        print("❌ PostgreSQL data clearing failed")
    
    # Summary
    print("\n" + "=" * 40)
    if mongo_success and postgres_success:
        print("🎉 All data cleared successfully!")
        print("\nYour WhatsApp AI Agent now has a clean slate.")
        print("All previous conversations and profiles have been removed.")
        return 0
    else:
        print("❌ Some data clearing operations failed!")
        if not mongo_success:
            print("  - MongoDB data clearing failed")
        if not postgres_success:
            print("  - PostgreSQL data clearing failed")
        print("\nPlease check the logs above for details.")
        return 1

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the clear data script
    exit_code = asyncio.run(main())
    sys.exit(exit_code)