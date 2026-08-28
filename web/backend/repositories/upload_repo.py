"""Persistence helpers for content-addressed user uploads."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web.backend.db.models import FileBlob, PaperFile, UserUpload


def _insert_for(session: AsyncSession, model):
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
        return insert(model)
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
        return insert(model)
    return None


@dataclass(frozen=True)
class OwnedUpload:
    upload_id: UUID
    blob_id: UUID
    original_filename: str
    sha256: str
    size_bytes: int
    mime_type: str | None
    storage_path: str


class UploadRepository:
    def __init__(self, db: AsyncSession, user_id: UUID):
        self.db = db
        self.user_id = user_id

    async def get_or_create_blob(
        self,
        *,
        sha256: str,
        size_bytes: int,
        mime_type: str | None,
        storage_path: str,
    ) -> FileBlob:
        return await get_or_create_file_blob(
            self.db,
            sha256=sha256,
            size_bytes=size_bytes,
            mime_type=mime_type,
            storage_path=storage_path,
        )

    async def get_or_create_upload(
        self,
        *,
        blob: FileBlob,
        original_filename: str,
    ) -> tuple[UserUpload, bool]:
        before = await self.db.execute(
            select(UserUpload).where(
                UserUpload.user_id == self.user_id,
                UserUpload.blob_id == blob.id,
            )
        )
        existing = before.scalar_one_or_none()
        if existing is not None:
            if existing.deleted_at is not None:
                existing.deleted_at = None
                existing.original_filename = original_filename
                await self.db.flush()
                return existing, True
            return existing, False

        insert = _insert_for(self.db, UserUpload)
        if insert is not None:
            statement = insert.values(
                id=uuid.uuid4(),
                user_id=self.user_id,
                blob_id=blob.id,
                original_filename=original_filename,
            ).on_conflict_do_nothing(index_elements=["user_id", "blob_id"])
            await self.db.execute(statement)
            await self.db.flush()
        else:
            self.db.add(UserUpload(
                user_id=self.user_id,
                blob_id=blob.id,
                original_filename=original_filename,
            ))
            await self.db.flush()

        result = await self.db.execute(
            select(UserUpload).where(
                UserUpload.user_id == self.user_id,
                UserUpload.blob_id == blob.id,
            )
        )
        return result.scalar_one(), True

    async def get_owned(self, upload_id: UUID) -> OwnedUpload | None:
        result = await self.db.execute(
            select(UserUpload, FileBlob)
            .join(FileBlob, FileBlob.id == UserUpload.blob_id)
            .where(
                UserUpload.id == upload_id,
                UserUpload.user_id == self.user_id,
                UserUpload.deleted_at.is_(None),
            )
        )
        row = result.one_or_none()
        if row is None:
            return None
        upload, blob = row
        return OwnedUpload(
            upload_id=upload.id,
            blob_id=blob.id,
            original_filename=upload.original_filename,
            sha256=blob.sha256,
            size_bytes=blob.size_bytes,
            mime_type=blob.mime_type,
            storage_path=blob.storage_path,
        )


async def get_or_create_file_blob(
    db: AsyncSession,
    *,
    sha256: str,
    size_bytes: int,
    mime_type: str | None,
    storage_path: str,
) -> FileBlob:
    insert = _insert_for(db, FileBlob)
    if insert is not None:
        statement = insert.values(
            id=uuid.uuid4(),
            sha256=sha256,
            size_bytes=size_bytes,
            mime_type=mime_type,
            storage_path=storage_path,
        ).on_conflict_do_nothing(index_elements=["sha256"])
        await db.execute(statement)
        await db.flush()
    else:
        existing = await db.execute(select(FileBlob).where(FileBlob.sha256 == sha256))
        if existing.scalar_one_or_none() is None:
            db.add(FileBlob(
                sha256=sha256,
                size_bytes=size_bytes,
                mime_type=mime_type,
                storage_path=storage_path,
            ))
            await db.flush()

    result = await db.execute(select(FileBlob).where(FileBlob.sha256 == sha256))
    return result.scalar_one()


async def get_paper_files_for_blob(db: AsyncSession, blob_id: UUID) -> list[PaperFile]:
    result = await db.execute(
        select(PaperFile)
        .where(PaperFile.blob_id == blob_id)
        .order_by(PaperFile.created_at, PaperFile.id)
    )
    return list(result.scalars().all())


async def link_paper_file(
    db: AsyncSession,
    *,
    paper_id: str,
    blob_id: UUID,
    source: str,
    access_scope: str,
    version: str | None = None,
) -> PaperFile:
    insert = _insert_for(db, PaperFile)
    if insert is not None:
        statement = insert.values(
            id=uuid.uuid4(),
            paper_id=paper_id,
            blob_id=blob_id,
            source=source,
            access_scope=access_scope,
            version=version,
        ).on_conflict_do_nothing(index_elements=["paper_id", "blob_id"])
        await db.execute(statement)
        await db.flush()
    else:
        existing = await db.execute(
            select(PaperFile).where(PaperFile.paper_id == paper_id, PaperFile.blob_id == blob_id)
        )
        if existing.scalar_one_or_none() is None:
            db.add(PaperFile(
                paper_id=paper_id,
                blob_id=blob_id,
                source=source,
                access_scope=access_scope,
                version=version,
            ))
            await db.flush()

    result = await db.execute(
        select(PaperFile).where(PaperFile.paper_id == paper_id, PaperFile.blob_id == blob_id)
    )
    paper_file = result.scalar_one()
    if access_scope == "public" and paper_file.access_scope != "public":
        paper_file.access_scope = "public"
    return paper_file
