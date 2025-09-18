"""create_profiles_schema

Revision ID: abca67650f20
Revises: 
Create Date: 2025-09-12 17:31:39.480331

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'abca67650f20'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Read the SQL file and execute it
    with open('server/db/migrations/001_create_profiles_schema.sql', 'r') as f:
        sql_content = f.read()
        # Split by semicolon and execute each statement
        statements = sql_content.split(';')
        for statement in statements:
            statement = statement.strip()
            if statement:
                op.execute(statement)


def downgrade() -> None:
    # Drop all tables in reverse order
    op.execute("DROP TABLE IF EXISTS profile_consents")
    op.execute("DROP TABLE IF EXISTS profile_relationships")
    op.execute("DROP TABLE IF EXISTS profile_summaries")
    op.execute("DROP TABLE IF EXISTS profile_tags")
    op.execute("DROP TABLE IF EXISTS profile_history")
    op.execute("DROP TABLE IF EXISTS profiles")
    op.execute("DROP EXTENSION IF EXISTS vector")
    op.execute("DROP EXTENSION IF EXISTS pgcrypto")
    op.execute("DROP EXTENSION IF EXISTS \"uuid-ossp\"")