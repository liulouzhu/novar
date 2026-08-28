"""add recovery events and fix recovery_states constraints

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "b2c3d4e5f6g7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. 修复 recovery_states 表：添加 unique 约束和新列
    # 先添加 run_status 列（如果不存在）
    op.add_column(
        "recovery_states",
        sa.Column("run_status", sa.String(length=20), server_default="running", nullable=False),
    )

    # 添加 unique(session_id, run_id) 约束
    op.create_unique_constraint(
        "uq_recovery_state_session_run",
        "recovery_states",
        ["session_id", "run_id"],
    )

    # 添加 user_id 索引
    op.create_index(
        "idx_recovery_states_user",
        "recovery_states",
        ["user_id"],
        unique=False,
    )

    # 添加 run_status 检查约束
    op.create_check_constraint(
        "ck_recovery_state_run_status",
        "recovery_states",
        "run_status IN ('running', 'completed', 'failed', 'cancelled', 'timed_out', 'interrupted', 'recovered')",
    )

    # 2. 创建 recovery_events 表
    op.create_table(
        "recovery_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_key", sa.String(length=128), nullable=False),
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "sequence", name="uq_recovery_event_run_sequence"),
        sa.UniqueConstraint("run_id", "event_key", name="uq_recovery_event_run_key"),
    )
    op.create_index(
        "idx_recovery_events_run",
        "recovery_events",
        ["run_id"],
        unique=False,
    )
    op.create_index(
        "idx_recovery_events_session",
        "recovery_events",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        "idx_recovery_events_user",
        "recovery_events",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_table("recovery_events")
    op.drop_index("idx_recovery_states_user", table_name="recovery_states")
    op.drop_constraint("uq_recovery_state_session_run", "recovery_states", type_="unique")
    op.drop_constraint("ck_recovery_state_run_status", "recovery_states", type_="check")
    op.drop_column("recovery_states", "run_status")
