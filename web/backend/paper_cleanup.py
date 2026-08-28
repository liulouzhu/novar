"""Durable, idempotent cleanup for user-paper deletion across all stores."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from uuid import UUID

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from web.backend.db.base import get_session_factory
from web.backend.db.models import (
    Chunk,
    Citation,
    Embedding,
    FileBlob,
    Paper,
    PaperCleanupJob,
    PaperFile,
    PaperIdentifier,
    UserPaper,
    UserUpload,
    utcnow,
)

logger = logging.getLogger("novare.web.paper_cleanup")

_MCP_SERVER_ROOT = Path(__file__).resolve().parents[2] / "mcp-server"
if str(_MCP_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_MCP_SERVER_ROOT))

_USER_STEPS = ("user_milvus", "user_cache")
_GLOBAL_STEPS = (
    "global_milvus",
    "elasticsearch",
    "postgresql",
    "files",
    "metadata",
    "global_cache",
)


def _initial_steps(purge_shared: bool, has_fulltext: bool) -> dict[str, str]:
    steps = {name: "pending" for name in _USER_STEPS}
    for name in _GLOBAL_STEPS:
        steps[name] = "pending" if purge_shared else "skipped"
    if purge_shared and not has_fulltext:
        steps["global_milvus"] = "skipped"
        steps["elasticsearch"] = "skipped"
    return steps


async def schedule_paper_cleanup(
    db: AsyncSession,
    *,
    paper_id: str,
    user_id: UUID,
) -> PaperCleanupJob | None:
    """Logically unlink a paper and enqueue cleanup in the same transaction."""
    paper_result = await db.execute(
        select(Paper).where(Paper.id == paper_id, Paper.deleted_at.is_(None)).with_for_update()
    )
    paper = paper_result.scalar_one_or_none()
    if paper is None:
        return None

    relation_result = await db.execute(
        select(UserPaper).where(
            UserPaper.paper_id == paper_id,
            UserPaper.user_id == user_id,
            UserPaper.deleted_at.is_(None),
        ).with_for_update()
    )
    relation = relation_result.scalar_one_or_none()
    if relation is None:
        return None

    now = utcnow()
    relation.deleted_at = now

    blob_result = await db.execute(
        select(PaperFile.blob_id).where(PaperFile.paper_id == paper_id)
    )
    blob_ids = list(dict.fromkeys(str(row[0]) for row in blob_result.all()))
    if blob_ids:
        await db.execute(
            update(UserUpload)
            .where(
                UserUpload.user_id == user_id,
                UserUpload.blob_id.in_([UUID(value) for value in blob_ids]),
                UserUpload.deleted_at.is_(None),
            )
            .values(deleted_at=now)
        )

    # Sessions use autoflush=False; reference counting must see this unlink.
    await db.flush()

    remaining_result = await db.execute(
        select(func.count()).select_from(UserPaper).where(
            UserPaper.paper_id == paper_id,
            UserPaper.deleted_at.is_(None),
        )
    )
    purge_shared = remaining_result.scalar_one() == 0
    chunk_count = await db.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.paper_id == paper_id)
    )

    if purge_shared and paper.visibility == "private":
        paper.deleted_at = now

    job = PaperCleanupJob(
        paper_id=paper_id,
        user_id=user_id,
        scope="paper" if purge_shared else "user",
        status="pending",
        steps=_initial_steps(purge_shared, bool(chunk_count)),
        payload={
            "blob_ids": blob_ids,
            "legacy_pdf_path": paper.pdf_path,
            "delete_private_metadata": bool(purge_shared and paper.visibility == "private"),
        },
        next_retry_at=now,
    )
    db.add(job)
    await db.flush()
    return job


async def _delete_milvus(paper_id: str, user_id: str | None) -> None:
    from core.vector_store import delete_vectors

    await asyncio.to_thread(delete_vectors, paper_id, user_id)


async def _delete_elasticsearch(paper_id: str) -> None:
    from core.elasticsearch_store import delete_paper_chunks

    await delete_paper_chunks(paper_id)


async def _invalidate_cache(user_id: UUID | None) -> None:
    from web.backend.redis_service import redis_service

    if user_id is None:
        prefixes = ("cache:paper_search:user:", "cache:rag_query:user:")
    else:
        prefixes = (
            f"cache:paper_search:user:{user_id}:",
            f"cache:rag_query:user:{user_id}:",
        )
    for prefix in prefixes:
        if not await redis_service.delete_prefix(prefix):
            raise RuntimeError(f"Redis unavailable while invalidating {prefix}")


async def _cleanup_postgresql(db: AsyncSession, job: PaperCleanupJob) -> None:
    chunk_ids_result = await db.execute(
        select(Chunk.id).where(Chunk.paper_id == job.paper_id)
    )
    chunk_ids = [row[0] for row in chunk_ids_result.all()]
    if chunk_ids:
        await db.execute(delete(Embedding).where(Embedding.chunk_id.in_(chunk_ids)))
    await db.execute(delete(Chunk).where(Chunk.paper_id == job.paper_id))
    await db.execute(
        delete(Citation).where(
            or_(Citation.source_id == job.paper_id, Citation.target_id == job.paper_id)
        )
    )
    await db.execute(delete(PaperFile).where(PaperFile.paper_id == job.paper_id))
    paper = await db.get(Paper, job.paper_id)
    if paper is not None:
        paper.pdf_path = None
    await db.flush()


def _delete_legacy_file(storage_path: str) -> None:
    root = Path(os.environ.get("RESEARCH_DATA_DIR", "./data")).resolve()
    path = Path(storage_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing to delete file outside managed data: {path}") from exc
    path.unlink(missing_ok=True)


async def _cleanup_files(db: AsyncSession, job: PaperCleanupJob) -> None:
    from novare.file_storage import delete_stored_blob

    payload = dict(job.payload or {})
    for raw_blob_id in payload.get("blob_ids", []):
        blob_id = UUID(raw_blob_id)
        blob = await db.get(FileBlob, blob_id)
        if blob is None:
            continue
        paper_refs = await db.scalar(
            select(func.count()).select_from(PaperFile).where(PaperFile.blob_id == blob_id)
        )
        active_uploads = await db.scalar(
            select(func.count()).select_from(UserUpload).where(
                UserUpload.blob_id == blob_id,
                UserUpload.deleted_at.is_(None),
            )
        )
        if paper_refs or active_uploads:
            continue
        delete_stored_blob(blob.storage_path)
        await db.execute(delete(UserUpload).where(UserUpload.blob_id == blob_id))
        await db.delete(blob)

    legacy_path = payload.get("legacy_pdf_path")
    if legacy_path:
        paper_refs = await db.scalar(
            select(func.count()).select_from(Paper).where(Paper.pdf_path == legacy_path)
        )
        blob_refs = await db.scalar(
            select(func.count()).select_from(FileBlob).where(FileBlob.storage_path == legacy_path)
        )
        if not paper_refs and not blob_refs:
            _delete_legacy_file(legacy_path)
    await db.flush()


async def _cleanup_private_metadata(db: AsyncSession, job: PaperCleanupJob) -> None:
    if not (job.payload or {}).get("delete_private_metadata"):
        return
    await db.execute(delete(UserPaper).where(UserPaper.paper_id == job.paper_id))
    await db.execute(delete(PaperIdentifier).where(PaperIdentifier.paper_id == job.paper_id))
    paper = await db.get(Paper, job.paper_id)
    if paper is not None:
        await db.delete(paper)
    await db.flush()


async def _active_reference_count(db: AsyncSession, paper_id: str) -> int:
    value = await db.scalar(
        select(func.count()).select_from(UserPaper).where(
            UserPaper.paper_id == paper_id,
            UserPaper.deleted_at.is_(None),
        )
    )
    return int(value or 0)


async def process_cleanup_job(db: AsyncSession, job_id: UUID) -> PaperCleanupJob | None:
    """Run one job under a paper row lock; all operations are retry-safe."""
    result = await db.execute(
        select(PaperCleanupJob).where(PaperCleanupJob.id == job_id).with_for_update()
    )
    job = result.scalar_one_or_none()
    if job is None or job.status == "completed":
        return job

    paper_result = await db.execute(
        select(Paper).where(Paper.id == job.paper_id).with_for_update()
    )
    paper = paper_result.scalar_one_or_none()
    steps = dict(job.steps or {})
    job.status = "running"
    job.attempts = int(job.attempts or 0) + 1
    job.last_error = None
    current_step = ""

    async def run_step(name: str, action) -> None:
        nonlocal current_step
        if steps.get(name) in {"completed", "skipped"}:
            return
        current_step = name
        steps[name] = "running"
        await action()
        steps[name] = "completed"
        job.steps = dict(steps)

    try:
        user_reassociated = await db.scalar(
            select(func.count()).select_from(UserPaper).where(
                UserPaper.paper_id == job.paper_id,
                UserPaper.user_id == job.user_id,
                UserPaper.deleted_at.is_(None),
            )
        )
        if user_reassociated:
            steps["user_milvus"] = "skipped"
        else:
            await run_step(
                "user_milvus",
                lambda: _delete_milvus(job.paper_id, str(job.user_id)),
            )
        await run_step("user_cache", lambda: _invalidate_cache(job.user_id))

        if job.scope == "paper":
            if await _active_reference_count(db, job.paper_id) > 0:
                for name in _GLOBAL_STEPS:
                    if steps.get(name) not in {"completed", "skipped"}:
                        steps[name] = "skipped"
                if paper is not None:
                    paper.deleted_at = None
            else:
                await run_step("global_milvus", lambda: _delete_milvus(job.paper_id, None))
                await run_step("elasticsearch", lambda: _delete_elasticsearch(job.paper_id))
                await run_step("postgresql", lambda: _cleanup_postgresql(db, job))
                await run_step("files", lambda: _cleanup_files(db, job))
                await run_step("metadata", lambda: _cleanup_private_metadata(db, job))
                await run_step("global_cache", lambda: _invalidate_cache(None))

        job.steps = dict(steps)
        job.status = "completed"
        job.completed_at = utcnow()
        job.next_retry_at = utcnow()
        await db.commit()
        return job
    except Exception as exc:
        if current_step:
            steps[current_step] = "failed"
        job.steps = dict(steps)
        job.status = "failed"
        job.last_error = f"{type(exc).__name__}: {exc}"[:2000]
        delay = min(3600, 30 * (2 ** min(job.attempts - 1, 7)))
        attempt_count = int(job.attempts or 0)
        job.next_retry_at = utcnow() + timedelta(seconds=delay)
        try:
            await db.commit()
        except Exception:
            # A database error can invalidate the transaction. Re-open the
            # outbox row so the failure itself is never lost.
            await db.rollback()
            retry_job = await db.get(PaperCleanupJob, job_id)
            if retry_job is not None:
                retry_job.steps = dict(steps)
                retry_job.status = "failed"
                retry_job.last_error = f"{type(exc).__name__}: {exc}"[:2000]
                retry_job.attempts = max(int(retry_job.attempts or 0), attempt_count)
                retry_job.next_retry_at = utcnow() + timedelta(seconds=delay)
                await db.commit()
                job = retry_job
        logger.warning(
            "Paper cleanup job %s failed at %s: %s",
            job.id,
            current_step,
            exc,
            exc_info=True,
        )
        return job


async def run_cleanup_job(job_id: UUID) -> PaperCleanupJob | None:
    factory = get_session_factory()
    async with factory() as db:
        return await process_cleanup_job(db, job_id)


async def process_pending_cleanup_jobs(limit: int = 20) -> int:
    """Process ready outbox jobs and return how many were attempted."""
    factory = get_session_factory()
    async with factory() as db:
        result = await db.execute(
            select(PaperCleanupJob.id)
            .where(
                PaperCleanupJob.status.in_(("pending", "failed")),
                PaperCleanupJob.next_retry_at <= utcnow(),
            )
            .order_by(PaperCleanupJob.created_at)
            .limit(limit)
        )
        job_ids = list(result.scalars().all())

    for job_id in job_ids:
        await run_cleanup_job(job_id)
    return len(job_ids)
