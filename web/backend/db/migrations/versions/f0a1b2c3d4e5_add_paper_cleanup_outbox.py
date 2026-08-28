"""add logical paper deletion and cleanup outbox

Revision ID: f0a1b2c3d4e5
Revises: e9f0a1b2c3d4
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "f0a1b2c3d4e5"
down_revision: Union[str, None] = "e9f0a1b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("papers", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index(op.f("ix_papers_deleted_at"), "papers", ["deleted_at"], unique=False)
    op.add_column("user_papers", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index(op.f("ix_user_papers_deleted_at"), "user_papers", ["deleted_at"], unique=False)
    op.add_column("user_uploads", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index(op.f("ix_user_uploads_deleted_at"), "user_uploads", ["deleted_at"], unique=False)

    op.create_table(
        "paper_cleanup_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", sa.String(length=255), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("scope", sa.String(length=10), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="pending", nullable=False),
        sa.Column("steps", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("scope IN ('user', 'paper')", name="ck_paper_cleanup_scope"),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'failed', 'completed')",
            name="ck_paper_cleanup_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_paper_cleanup_ready", "paper_cleanup_jobs", ["status", "next_retry_at"], unique=False)
    op.create_index("idx_paper_cleanup_paper", "paper_cleanup_jobs", ["paper_id"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_paper_cleanup_paper", table_name="paper_cleanup_jobs")
    op.drop_index("idx_paper_cleanup_ready", table_name="paper_cleanup_jobs")
    op.drop_table("paper_cleanup_jobs")
    op.drop_index(op.f("ix_user_uploads_deleted_at"), table_name="user_uploads")
    op.drop_column("user_uploads", "deleted_at")
    op.drop_index(op.f("ix_user_papers_deleted_at"), table_name="user_papers")
    op.drop_column("user_papers", "deleted_at")
    op.drop_index(op.f("ix_papers_deleted_at"), table_name="papers")
    op.drop_column("papers", "deleted_at")
