"""add episodic_memories table

Revision ID: f1a2b3c4d5e6
Revises: d4e5ce7fe17f
Create Date: 2026-07-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, Sequence[str], None] = 'd4e5ce7fe17f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create episodic_memories table for episodic memory."""
    op.create_table(
        'episodic_memories',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('session_id', sa.String(length=64), nullable=True),

        sa.Column('memory_type', sa.String(length=50), nullable=False),
        sa.Column('summary', sa.String(length=500), nullable=False),
        sa.Column('context', sa.Text(), server_default='', nullable=True),
        sa.Column('action', sa.Text(), server_default='', nullable=True),
        sa.Column('outcome', sa.Text(), server_default='', nullable=True),

        sa.Column('topics', postgresql.JSONB(astext_type=sa.Text()), server_default='[]', nullable=True),
        sa.Column('source_message_ids', postgresql.JSONB(astext_type=sa.Text()), server_default='[]', nullable=True),

        sa.Column('importance', sa.Float(), server_default='0.5', nullable=True),
        sa.Column('confidence', sa.Float(), server_default='0.5', nullable=True),

        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('embedding_model', sa.String(length=100), server_default='', nullable=True),
        sa.Column('vector_id', sa.String(length=64), nullable=True),
        sa.Column('index_status', sa.String(length=20), server_default='pending', nullable=False),

        sa.Column('status', sa.String(length=20), server_default='active', nullable=False),
        sa.Column('pinned', sa.Boolean(), server_default='false', nullable=True),

        sa.Column('occurred_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('last_retrieved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('retrieval_count', sa.Integer(), server_default='0', nullable=True),

        sa.UniqueConstraint('user_id', 'content_hash', name='uq_episodic_memory_hash'),
    )
    op.create_index('idx_episodic_memories_user', 'episodic_memories', ['user_id'])
    op.create_index('idx_episodic_memories_session', 'episodic_memories', ['session_id'])
    op.create_index('idx_episodic_memories_status', 'episodic_memories', ['user_id', 'status'])


def downgrade() -> None:
    """Drop episodic_memories table."""
    op.drop_index('idx_episodic_memories_status', table_name='episodic_memories')
    op.drop_index('idx_episodic_memories_session', table_name='episodic_memories')
    op.drop_index('idx_episodic_memories_user', table_name='episodic_memories')
    op.drop_table('episodic_memories')
