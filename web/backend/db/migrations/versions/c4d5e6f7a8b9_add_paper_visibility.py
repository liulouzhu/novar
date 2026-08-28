"""add paper visibility and created_by_user_id

Revision ID: c4d5e6f7a8b9
Revises: b3c2d1e0f4a5
Create Date: 2026-06-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'c4d5e6f7a8b9'
down_revision: Union[str, Sequence[str], None] = 'b3c2d1e0f4a5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add visibility and created_by_user_id to papers table."""
    op.add_column('papers', sa.Column(
        'visibility', sa.String(length=10), nullable=False,
        server_default='public',
    ))
    op.add_column('papers', sa.Column(
        'created_by_user_id', postgresql.UUID(as_uuid=True),
        sa.ForeignKey('users.id'), nullable=True,
    ))
    op.create_check_constraint(
        'ck_paper_visibility', 'papers',
        "visibility IN ('public', 'private')",
    )
    op.create_index('idx_papers_visibility', 'papers', ['visibility'])
    op.create_index('idx_papers_creator', 'papers', ['created_by_user_id'])


def downgrade() -> None:
    """Remove visibility and created_by_user_id from papers table."""
    op.drop_index('idx_papers_creator', table_name='papers')
    op.drop_index('idx_papers_visibility', table_name='papers')
    op.drop_constraint('ck_paper_visibility', 'papers', type_='check')
    op.drop_column('papers', 'created_by_user_id')
    op.drop_column('papers', 'visibility')
