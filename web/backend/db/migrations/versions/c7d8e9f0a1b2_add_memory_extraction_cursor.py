"""add memory extraction cursor to sessions

Revision ID: c7d8e9f0a1b2
Revises: f1a2b3c4d5e6
Create Date: 2026-07-25 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c7d8e9f0a1b2'
down_revision: Union[str, Sequence[str], None] = 'f1a2b3c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add last_extracted_message_id and last_memory_extracted_at to sessions."""
    op.add_column('sessions', sa.Column('last_extracted_message_id', sa.Integer(), nullable=True))
    op.add_column('sessions', sa.Column('last_memory_extracted_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    """Remove memory extraction cursor columns from sessions."""
    op.drop_column('sessions', 'last_memory_extracted_at')
    op.drop_column('sessions', 'last_extracted_message_id')
