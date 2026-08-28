"""add user_paper relation_type and fulltext_access

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
Create Date: 2026-06-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd5e6f7a8b9c0'
down_revision: Union[str, Sequence[str], None] = 'c4d5e6f7a8b9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add relation_type, has_fulltext_access, source to user_papers."""
    op.add_column('user_papers', sa.Column(
        'relation_type', sa.String(length=20), nullable=False,
        server_default='searched',
    ))
    op.add_column('user_papers', sa.Column(
        'has_fulltext_access', sa.Boolean(), nullable=False,
        server_default='false',
    ))
    op.add_column('user_papers', sa.Column(
        'source', sa.String(length=30), nullable=True,
    ))
    op.create_check_constraint(
        'ck_user_paper_relation_type', 'user_papers',
        "relation_type IN ('searched', 'parsed', 'uploaded', 'shared')",
    )
    op.create_index('idx_user_papers_fulltext', 'user_papers', ['user_id', 'has_fulltext_access'])

    # Backfill existing rows: parsed papers get fulltext access
    op.execute(
        "UPDATE user_papers SET relation_type = 'parsed', has_fulltext_access = true "
        "WHERE relation_type = 'searched'"
    )


def downgrade() -> None:
    """Remove relation_type, has_fulltext_access, source from user_papers."""
    op.drop_index('idx_user_papers_fulltext', table_name='user_papers')
    op.drop_constraint('ck_user_paper_relation_type', 'user_papers', type_='check')
    op.drop_column('user_papers', 'source')
    op.drop_column('user_papers', 'has_fulltext_access')
    op.drop_column('user_papers', 'relation_type')
