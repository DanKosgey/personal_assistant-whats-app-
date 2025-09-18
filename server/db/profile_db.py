"""
Database connection and migration management for profile system
Handles PostgreSQL connections, migrations, and connection pooling
"""

import asyncio
import asyncpg
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ProfileDatabase:
    """PostgreSQL database manager for profile system with migration support"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv('DATABASE_URL', 'postgresql://localhost:5432/whatsapp_profiles')
        self.pool: Optional[asyncpg.Pool] = None
        self.migrations_dir = Path(__file__).parent / 'migrations'
        
        # Ensure migrations directory exists
        self.migrations_dir.mkdir(exist_ok=True)
        
    async def initialize(self) -> None:
        """Initialize database connection pool and run migrations"""
        try:
            # Create connection pool with larger size for better concurrency handling
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60,
                server_settings={
                    'jit': 'off',  # Disable JIT for faster small queries
                    'application_name': 'whatsapp_profiles'
                }
            )
            
            logger.info("✅ Database connection pool created")
            
            # Create migrations table if it doesn't exist
            await self._ensure_migrations_table()
            
            # Run pending migrations
            await self.run_migrations()
            
            logger.info("✅ Database initialization complete")
            
        except Exception as e:
            logger.warning(f"⚠️ Profile database not available: {e}")
            # Don't raise the exception - allow the app to continue without PostgreSQL
            logger.info("ℹ️ Profile database will be disabled - continuing with MongoDB only")
            self.pool = None
    
    async def close(self) -> None:
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
    
    async def _ensure_migrations_table(self) -> None:
        """Create migrations tracking table"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS migrations (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL UNIQUE,
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
                    checksum VARCHAR(64)
                )
            """)
    
    async def run_migrations(self) -> None:
        """Run all pending migrations"""
        if not self.migrations_dir.exists():
            logger.warning("Migrations directory not found")
            return
        
        # Get list of migration files
        migration_files = sorted([
            f for f in self.migrations_dir.glob('*.sql')
            if f.is_file()
        ])
        
        if not migration_files:
            logger.info("No migration files found")
            return
        
        # Get applied migrations
        async with self.pool.acquire() as conn:
            applied_migrations = await conn.fetch(
                "SELECT filename FROM migrations ORDER BY id"
            )
            applied_set = {row['filename'] for row in applied_migrations}
        
        # Run pending migrations
        for migration_file in migration_files:
            filename = migration_file.name
            
            if filename in applied_set:
                logger.debug(f"Migration {filename} already applied")
                continue
            
            logger.info(f"Running migration: {filename}")
            
            try:
                # Read migration file
                migration_sql = migration_file.read_text(encoding='utf-8')
                
                # Calculate checksum
                import hashlib
                checksum = hashlib.sha256(migration_sql.encode()).hexdigest()
                
                # Run migration in transaction
                async with self.pool.acquire() as conn:
                    async with conn.transaction():
                        # Execute migration
                        await conn.execute(migration_sql)
                        
                        # Record migration
                        await conn.execute(
                            "INSERT INTO migrations (filename, checksum) VALUES ($1, $2)",
                            filename, checksum
                        )
                
                logger.info(f"✅ Migration {filename} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ Migration {filename} failed: {e}")
                raise
    
    async def execute(self, query: str, *args) -> Any:
        """Execute a query with parameters"""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fetch multiple rows"""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        try:
            async with self.pool.acquire() as conn:
                logger.debug(f"Executing query: {query} with args: {args}")
                rows = await conn.fetch(query, *args)
                result = [dict(row) for row in rows]
                logger.debug(f"Query returned {len(result)} rows")
                return result
        except Exception as e:
            logger.error(f"Failed to execute query: {query} with args: {args}")
            logger.error(f"Error: {e}")
            raise

    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetch single row"""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        try:
            async with self.pool.acquire() as conn:
                logger.debug(f"Executing fetchrow query: {query} with args: {args}")
                row = await conn.fetchrow(query, *args)
                result = dict(row) if row else None
                logger.debug(f"Fetchrow returned: {result is not None}")
                return result
        except Exception as e:
            logger.error(f"Failed to execute fetchrow query: {query} with args: {args}")
            logger.error(f"Error: {e}")
            raise

    async def fetchval(self, query: str, *args) -> Any:
        """Fetch single value"""
        if not self.pool:
            raise RuntimeError("Database not initialized")
        try:
            async with self.pool.acquire() as conn:
                logger.debug(f"Executing fetchval query: {query} with args: {args}")
                result = await conn.fetchval(query, *args)
                logger.debug(f"Fetchval returned: {result}")
                return result
        except Exception as e:
            logger.error(f"Failed to execute fetchval query: {query} with args: {args}")
            logger.error(f"Error: {e}")
            raise
    
    async def transaction(self):
        """Create database transaction context"""
        return self.pool.acquire()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check database health and return status"""
        try:
            start_time = datetime.now()
            
            # Simple connectivity test
            result = await self.fetchval("SELECT 1")
            
            # Pool status
            pool_size = self.pool.get_size()
            pool_idle = self.pool.get_idle_size()
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "status": "healthy",
                "connected": result == 1,
                "response_time_ms": round(response_time, 2),
                "pool_size": pool_size,
                "pool_idle": pool_idle,
                "pool_active": pool_size - pool_idle
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False
            }

# Global database instance
profile_db = ProfileDatabase()

async def get_profile_db() -> ProfileDatabase:
    """Get database instance (dependency injection helper)"""
    if not profile_db.pool:
        await profile_db.initialize()
    return profile_db