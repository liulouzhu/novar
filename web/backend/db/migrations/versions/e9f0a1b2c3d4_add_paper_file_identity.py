"""add paper identifiers and content-addressed uploads

Revision ID: e9f0a1b2c3d4
Revises: d8e9f0a1b2c3
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "e9f0a1b2c3d4"
down_revision: Union[str, None] = "d8e9f0a1b2c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "file_blobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("sha256", name="uq_file_blob_sha256"),
    )
    op.create_index(op.f("ix_file_blobs_sha256"), "file_blobs", ["sha256"], unique=False)

    op.create_table(
        "paper_identifiers",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", sa.String(length=255), nullable=False),
        sa.Column("identifier_type", sa.String(length=20), nullable=False),
        sa.Column("identifier", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("identifier", name="uq_paper_identifier"),
    )
    op.create_index("idx_paper_identifiers_paper", "paper_identifiers", ["paper_id"], unique=False)

    op.create_table(
        "user_uploads",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("blob_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("original_filename", sa.String(length=512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["blob_id"], ["file_blobs.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "blob_id", name="uq_user_upload_blob"),
    )
    op.create_index("idx_user_uploads_user", "user_uploads", ["user_id"], unique=False)
    op.create_index(op.f("ix_user_uploads_blob_id"), "user_uploads", ["blob_id"], unique=False)

    op.create_table(
        "paper_files",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("paper_id", sa.String(length=255), nullable=False),
        sa.Column("blob_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source", sa.String(length=30), server_default="upload", nullable=False),
        sa.Column("version", sa.String(length=50), nullable=True),
        sa.Column("access_scope", sa.String(length=10), server_default="private", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("access_scope IN ('public', 'private')", name="ck_paper_file_access_scope"),
        sa.ForeignKeyConstraint(["blob_id"], ["file_blobs.id"]),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("paper_id", "blob_id", name="uq_paper_file_blob"),
    )
    op.create_index("idx_paper_files_blob", "paper_files", ["blob_id"], unique=False)


def downgrade() -> None:
    op.drop_index("idx_paper_files_blob", table_name="paper_files")
    op.drop_table("paper_files")
    op.drop_index(op.f("ix_user_uploads_blob_id"), table_name="user_uploads")
    op.drop_index("idx_user_uploads_user", table_name="user_uploads")
    op.drop_table("user_uploads")
    op.drop_index("idx_paper_identifiers_paper", table_name="paper_identifiers")
    op.drop_table("paper_identifiers")
    op.drop_index(op.f("ix_file_blobs_sha256"), table_name="file_blobs")
    op.drop_table("file_blobs")
