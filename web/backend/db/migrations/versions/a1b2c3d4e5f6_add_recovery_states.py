"""add recovery states

Revision ID: a1b2c3d4e5f6
Revises: f0a1b2c3d4e5
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "f0a1b2c3d4e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "recovery_states",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("turn_id", sa.String(length=32), nullable=False),
        sa.Column("iteration", sa.Integer(), server_default="0", nullable=False),
        sa.Column("retry_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default="active",
            nullable=False,
        ),
        sa.Column(
            "recovery_data",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "schema_version",
            sa.Integer(),
            server_default="1",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('active', 'completed', 'failed', 'interrupted')",
            name="ck_recovery_state_status",
        ),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("idx_recovery_states_session"),
        "recovery_states",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        op.f("idx_recovery_states_run"),
        "recovery_states",
        ["run_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("idx_recovery_states_run"), table_name="recovery_states")
    op.drop_index(op.f("idx_recovery_states_session"), table_name="recovery_states")
    op.drop_table("recovery_states")
