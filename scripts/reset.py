#!/usr/bin/env python3
"""
Database reset script for WhatsApp AI Agent
Clears all data from MongoDB and PostgreSQL databases
"""

import os
import sys
import asyncio
import logging
import traceback
from typing import Optional
import types

# Add the root directory to the path
root_path = os.path.dirname(os.path.dirname(__file__))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def reset_mongodb() -> bool:
    """Reset MongoDB database by dropping all collections"""
    db_manager = None
    try:
        logger.info("Starting MongoDB reset...")
        
        # Import DatabaseManager directly
        from server.database import DatabaseManager
        logger.info("Successfully imported DatabaseManager")
        
        db_manager = DatabaseManager()
        logger.info("Created DatabaseManager instance")
        
        # Connect to database
        await db_manager.connect()
        logger.info("Connected to database")
        
        # Check if database connection is valid (different check for MongoDB objects)
        if db_manager.db is None:
            logger.error("Failed to connect to MongoDB")
            return False
            
        # Get list of collections
        if hasattr(db_manager, '_using_motor') and db_manager._using_motor:
            # Motor (async)
            collection_names = await db_manager.db.list_collection_names()
        else:
            # PyMongo (sync wrapper)
            collection_names = await asyncio.to_thread(db_manager.db.list_collection_names)
        
        logger.info(f"Found collections: {collection_names}")
        
        # Drop each collection
        for collection_name in collection_names:
            try:
                if hasattr(db_manager, '_using_motor') and db_manager._using_motor:
                    await db_manager.db[collection_name].drop()
                else:
                    await asyncio.to_thread(db_manager.db[collection_name].drop)
                logger.info(f"Dropped MongoDB collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Failed to drop collection {collection_name}: {e}")
        
        logger.info("MongoDB reset completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to reset MongoDB: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        if db_manager:
            try:
                await db_manager.close()
            except Exception as e:
                logger.warning(f"Failed to close database connection: {e}")
                pass

def reset_postgresql() -> bool:
    """Reset PostgreSQL database by dropping all tables"""
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        
        # Get database connection details from environment
        db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:devpass@localhost:5433/whatsapp_profiles')
        
        # Parse connection details
        # Format: postgresql://user:password@host:port/database
        import re
        match = re.match(r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', db_url)
        if not match:
            logger.error("Invalid DATABASE_URL format")
            return False
            
        user, password, host, port, database = match.groups()
        
        # Connect to PostgreSQL server (not specific database)
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database='postgres'  # Connect to default database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Terminate all connections to the target database
        try:
            cursor.execute("""
                SELECT pg_terminate_backend(pid) 
                FROM pg_stat_activity 
                WHERE datname = %s AND pid <> pg_backend_pid()
            """, (database,))
            logger.info(f"Terminated existing connections to database: {database}")
        except Exception as e:
            logger.warning(f"Failed to terminate connections: {e}")
        
        # Drop and recreate the database
        try:
            cursor.execute(f"DROP DATABASE IF EXISTS {database}")
            cursor.execute(f"CREATE DATABASE {database}")
            logger.info(f"Dropped and recreated PostgreSQL database: {database}")
        except Exception as e:
            logger.error(f"Failed to drop/recreate database {database}: {e}")
            return False
        
        cursor.close()
        conn.close()
        
        # Now connect to the recreated database and drop all tables
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        cursor = conn.cursor()
        
        # Drop all tables
        try:
            cursor.execute("""
                SELECT tablename FROM pg_tables WHERE schemaname = 'public'
            """)
            tables = cursor.fetchall()
            
            for table in tables:
                table_name = table[0]
                cursor.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
                logger.info(f"Dropped PostgreSQL table: {table_name}")
                
            conn.commit()
            logger.info("PostgreSQL tables reset completed successfully")
        except Exception as e:
            logger.error(f"Failed to drop PostgreSQL tables: {e}")
            return False
        finally:
            cursor.close()
            conn.close()
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to reset PostgreSQL: {e}")
        return False

async def main():
    """Main reset function"""
    print(" WhatsApp AI Agent Database Reset Script ")
    print("=" * 50)
    print("This script will clear ALL data from both MongoDB and PostgreSQL databases.")
    print("This action is IRREVERSIBLE and ALL data will be permanently deleted.")
    print()
    
    # Confirm with user
    response = input("Are you sure you want to proceed? Type 'YES' to confirm: ")
    if response != "YES":
        print("Reset cancelled.")
        return 1
    
    print("\nStarting database reset...")
    
    # Reset MongoDB
    print("\nResetting MongoDB...")
    mongo_success = await reset_mongodb()
    if mongo_success:
        print("✅ MongoDB reset completed successfully")
    else:
        print("❌ MongoDB reset failed")
    
    # Reset PostgreSQL
    print("\nResetting PostgreSQL...")
    postgres_success = reset_postgresql()
    if postgres_success:
        print("✅ PostgreSQL reset completed successfully")
    else:
        print("❌ PostgreSQL reset failed")
    
    # Summary
    print("\n" + "=" * 50)
    if mongo_success and postgres_success:
        print("🎉 All databases reset successfully!")
        print("\nYour WhatsApp AI Agent now has a clean slate.")
        print("You can start fresh with no existing conversations or profiles.")
        return 0
    else:
        print("❌ Some database resets failed!")
        if not mongo_success:
            print("  - MongoDB reset failed")
        if not postgres_success:
            print("  - PostgreSQL reset failed")
        print("\nPlease check the logs above for details.")
        return 1

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the reset
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


