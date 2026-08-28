"""merge embeddings and user_paper branches

Revision ID: d4e5ce7fe17f
Revises: 89feba200900, d5e6f7a8b9c0
Create Date: 2026-06-11 09:27:05.553270

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4e5ce7fe17f'
down_revision: Union[str, Sequence[str], None] = ('89feba200900', 'd5e6f7a8b9c0')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
