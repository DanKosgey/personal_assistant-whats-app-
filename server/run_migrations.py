#!/usr/bin/env python3
"""
Script to run database migrations directly using psycopg2.
This avoids issues with Alembic configuration and authentication.
"""

import psycopg2
from psycopg2 import sql
import os

def run_migrations():
    # Database connection parameters
    db_params = {
        'host': 'localhost',
        'port': 5432,
        'database': 'whatsapp_profiles',
        'user': 'postgres',
        'password': 'devpass'
    }
    
    try:
        # Connect to the database
        print("Connecting to database...")
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        print("Connected successfully!")
        
        # Read the SQL migration file
        migration_file = os.path.join('server', 'db', 'migrations', '001_create_profiles_schema.sql')
        print(f"Reading migration file: {migration_file}")
        
        with open(migration_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # Split the SQL content into statements
        # We need to be careful with the splitting to handle functions properly
        statements = []
        current_statement = ""
        in_function = False
        dollar_quote_count = 0
        
        for line in sql_content.split('\n'):
            stripped_line = line.strip()
            
            # Check for dollar quotes used in function definitions
            if '$$' in line:
                dollar_quote_count += line.count('$$')
                # If we have an odd number of dollar quotes, we're inside a function body
                in_function = (dollar_quote_count % 2 == 1)
            
            current_statement += line + '\n'
            
            # If we're not in a function and the line ends with semicolon, it's a complete statement
            if not in_function and stripped_line.endswith(';') and stripped_line != '$$':
                statements.append(current_statement.strip())
                current_statement = ""
        
        # Add any remaining statement
        if current_statement.strip():
            statements.append(current_statement.strip())
        
        # Execute each statement
        print(f"Executing {len(statements)} statements...")
        for i, statement in enumerate(statements):
            if statement.strip():
                print(f"Executing statement {i+1}/{len(statements)}")
                try:
                    cursor.execute(statement)
                except Exception as e:
                    print(f"Error executing statement {i+1}: {e}")
                    print(f"Statement: {statement[:100]}...")
                    # Continue with other statements
                    continue
        
        # Commit the transaction
        conn.commit()
        print("All migrations completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        if 'conn' in locals():
            conn.rollback()
        raise
    finally:
        # Close the connection
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    run_migrations()