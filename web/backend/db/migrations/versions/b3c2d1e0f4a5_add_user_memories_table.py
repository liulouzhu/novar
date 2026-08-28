"""add user_memories table

Revision ID: b3c2d1e0f4a5
Revises: abe8fb90d84c
Create Date: 2026-06-09 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'b3c2d1e0f4a5'
down_revision: Union[str, Sequence[str], None] = 'abe8fb90d84c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create user_memories table for long-term user memory."""
    op.create_table(
        'user_memories',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('key', sa.String(length=100), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), server_default='1.0', nullable=True),
        sa.Column('pinned', sa.Boolean(), server_default='false', nullable=True),
        sa.Column('tags', postgresql.JSONB(astext_type=sa.Text()), server_default='[]', nullable=True),
        sa.Column('source', sa.String(length=50), server_default='auto', nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'category', 'key', name='uq_user_memory_key'),
    )
    op.create_index('idx_user_memories_user', 'user_memories', ['user_id'])
    op.create_index('idx_user_memories_category', 'user_memories', ['user_id', 'category'])


def downgrade() -> None:
    """Drop user_memories table."""
    op.drop_index('idx_user_memories_category', table_name='user_memories')
    op.drop_index('idx_user_memories_user', table_name='user_memories')
    op.drop_table('user_memories')
