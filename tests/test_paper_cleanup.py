"""Cross-store paper deletion, garbage collection, and retry tests."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, select

from web.backend.db.models import (
    Chunk,
    Embedding,
    FileBlob,
    Paper,
    PaperCleanupJob,
    PaperFile,
    User,
    UserPaper,
    UserUpload,
)
from web.backend.paper_cleanup import process_cleanup_job, schedule_paper_cleanup
from web.backend.repositories.user_paper_repo import UserPaperRepository


async def _user(db, label: str) -> User:
    row = User(
        id=uuid.uuid4(),
        username=f"cleanup-{label}-{uuid.uuid4().hex[:6]}",
        email=f"cleanup-{label}-{uuid.uuid4().hex[:6]}@example.com",
        password_hash="test",
    )
    db.add(row)
    await db.flush()
    return row


async def _seed_parsed_paper(
    db,
    tmp_path,
    monkeypatch,
    *,
    visibility: str,
    users: list[User],
    paper_id: str,
    with_blob: bool = True,
):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(data_dir))
    paper = Paper(
        id=paper_id,
        title="Cleanup test paper",
        visibility=visibility,
        created_by_user_id=users[0].id,
    )
    db.add(paper)
    for user in users:
        db.add(UserPaper(
            user_id=user.id,
            paper_id=paper_id,
            relation_type="uploaded",
            has_fulltext_access=True,
            source="upload",
        ))
    await db.flush()

    chunk = Chunk(paper_id=paper_id, section="Body", ordinal=0, text="content")
    db.add(chunk)
    await db.flush()
    db.add(Embedding(chunk_id=chunk.id, dim=1, vec=b"\x00\x00\x00\x00"))

    blob = None
    path = None
    if with_blob:
        digest = "a" * 64
        path = data_dir / "file_blobs" / digest[:2] / digest
        path.parent.mkdir(parents=True)
        path.write_bytes(b"pdf")
        blob = FileBlob(
            id=uuid.uuid4(),
            sha256=digest,
            size_bytes=3,
            mime_type="application/pdf",
            storage_path=str(path.resolve()),
        )
        db.add(blob)
        await db.flush()
        db.add(PaperFile(
            paper_id=paper_id,
            blob_id=blob.id,
            source="upload",
            access_scope=visibility,
        ))
        for user in users:
            db.add(UserUpload(
                user_id=user.id,
                blob_id=blob.id,
                original_filename="paper.pdf",
            ))
        paper.pdf_path = str(path.resolve())
    await db.flush()
    return paper, chunk, blob, path


def _patch_external_success(monkeypatch):
    import web.backend.paper_cleanup as cleanup

    milvus_calls = []
    cache_calls = []
    es_calls = []

    async def milvus(paper_id, user_id):
        milvus_calls.append((paper_id, user_id))

    async def cache(user_id):
        cache_calls.append(user_id)

    async def elasticsearch(paper_id):
        es_calls.append(paper_id)

    monkeypatch.setattr(cleanup, "_delete_milvus", milvus)
    monkeypatch.setattr(cleanup, "_invalidate_cache", cache)
    monkeypatch.setattr(cleanup, "_delete_elasticsearch", elasticsearch)
    return milvus_calls, cache_calls, es_calls


@pytest.mark.asyncio
async def test_shared_delete_only_removes_requesting_user(db_session, tmp_path, monkeypatch):
    user_a = await _user(db_session, "a")
    user_b = await _user(db_session, "b")
    paper, chunk, blob, path = await _seed_parsed_paper(
        db_session,
        tmp_path,
        monkeypatch,
        visibility="private",
        users=[user_a, user_b],
        paper_id="upload:sha256:shared",
    )
    milvus, caches, es = _patch_external_success(monkeypatch)

    job = await schedule_paper_cleanup(
        db_session, paper_id=paper.id, user_id=user_a.id
    )
    assert job is not None and job.scope == "user"
    await db_session.commit()
    job = await process_cleanup_job(db_session, job.id)

    assert job.status == "completed"
    assert milvus == [(paper.id, str(user_a.id))]
    assert caches == [user_a.id]
    assert es == []
    assert await db_session.get(Chunk, chunk.id) is not None
    assert await db_session.get(FileBlob, blob.id) is not None
    assert path.exists()
    active_b = await db_session.scalar(
        select(func.count()).select_from(UserPaper).where(
            UserPaper.user_id == user_b.id,
            UserPaper.deleted_at.is_(None),
        )
    )
    assert active_b == 1


@pytest.mark.asyncio
async def test_last_public_reference_cleans_fulltext_but_keeps_metadata(
    db_session, tmp_path, monkeypatch
):
    user = await _user(db_session, "public")
    paper, chunk, blob, path = await _seed_parsed_paper(
        db_session,
        tmp_path,
        monkeypatch,
        visibility="public",
        users=[user],
        paper_id="doi:10.1000/cleanup",
    )
    milvus, caches, es = _patch_external_success(monkeypatch)

    job = await schedule_paper_cleanup(db_session, paper_id=paper.id, user_id=user.id)
    assert job is not None and job.scope == "paper"
    await db_session.commit()
    job = await process_cleanup_job(db_session, job.id)

    kept = await db_session.get(Paper, paper.id)
    assert job.status == "completed"
    assert kept is not None and kept.deleted_at is None and kept.pdf_path is None
    assert await db_session.get(Chunk, chunk.id) is None
    assert await db_session.get(FileBlob, blob.id) is None
    assert not path.exists()
    assert (paper.id, str(user.id)) in milvus
    assert (paper.id, None) in milvus
    assert es == [paper.id]
    assert user.id in caches and None in caches


@pytest.mark.asyncio
async def test_last_private_reference_removes_metadata(db_session, tmp_path, monkeypatch):
    user = await _user(db_session, "private")
    paper, _, _, _ = await _seed_parsed_paper(
        db_session,
        tmp_path,
        monkeypatch,
        visibility="private",
        users=[user],
        paper_id="upload:sha256:private",
    )
    _patch_external_success(monkeypatch)

    job = await schedule_paper_cleanup(db_session, paper_id=paper.id, user_id=user.id)
    await db_session.commit()
    job = await process_cleanup_job(db_session, job.id)

    assert job.status == "completed"
    assert await db_session.get(Paper, paper.id) is None
    assert await db_session.get(PaperCleanupJob, job.id) is not None


@pytest.mark.asyncio
async def test_cleanup_failure_is_persisted_and_retry_completes(
    db_session, tmp_path, monkeypatch
):
    import web.backend.paper_cleanup as cleanup

    user = await _user(db_session, "retry")
    paper, chunk, _, _ = await _seed_parsed_paper(
        db_session,
        tmp_path,
        monkeypatch,
        visibility="public",
        users=[user],
        paper_id="arxiv:2601.00001",
        with_blob=False,
    )
    attempts = 0

    async def flaky_milvus(_paper_id, _user_id):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Milvus unavailable")

    async def success(*_args):
        return None

    monkeypatch.setattr(cleanup, "_delete_milvus", flaky_milvus)
    monkeypatch.setattr(cleanup, "_delete_elasticsearch", success)
    monkeypatch.setattr(cleanup, "_invalidate_cache", success)

    job = await schedule_paper_cleanup(db_session, paper_id=paper.id, user_id=user.id)
    await db_session.commit()
    job = await process_cleanup_job(db_session, job.id)
    assert job.status == "failed"
    assert job.steps["user_milvus"] == "failed"
    assert "Milvus unavailable" in job.last_error
    assert await db_session.get(Chunk, chunk.id) is not None

    job = await process_cleanup_job(db_session, job.id)
    assert job.status == "completed"
    assert job.attempts == 2
    assert await db_session.get(Chunk, chunk.id) is None


@pytest.mark.asyncio
async def test_reassociation_cancels_stale_global_cleanup(db_session, tmp_path, monkeypatch):
    user = await _user(db_session, "restore")
    paper, chunk, _, _ = await _seed_parsed_paper(
        db_session,
        tmp_path,
        monkeypatch,
        visibility="public",
        users=[user],
        paper_id="doi:10.1000/restored",
        with_blob=False,
    )
    milvus, _, es = _patch_external_success(monkeypatch)

    job = await schedule_paper_cleanup(db_session, paper_id=paper.id, user_id=user.id)
    await UserPaperRepository(db_session, user.id).associate(
        paper.id,
        relation_type="parsed",
        has_fulltext_access=True,
        source="paper_parse",
    )
    await db_session.commit()
    job = await process_cleanup_job(db_session, job.id)

    assert job.status == "completed"
    assert job.steps["user_milvus"] == "skipped"
    assert job.steps["postgresql"] == "skipped"
    assert milvus == []
    assert es == []
    assert await db_session.get(Chunk, chunk.id) is not None


def test_milvus_delete_uses_scoped_expression(monkeypatch):
    from core import vector_store

    expressions = []

    class Result:
        delete_count = 3

    class Collection:
        def __init__(self, _name):
            pass

        def delete(self, expression):
            expressions.append(expression)
            return Result()

        def flush(self):
            pass

    monkeypatch.setattr(vector_store, "_ensure_connected", lambda: None)
    monkeypatch.setattr(vector_store.utility, "has_collection", lambda _name: True)
    monkeypatch.setattr(vector_store, "Collection", Collection)

    assert vector_store.delete_vectors('doi:10.1/"quoted"', "user-a") == 3
    assert expressions == ['paper_id == "doi:10.1/\\\"quoted\\\"" and user_id == "user-a"']


@pytest.mark.asyncio
async def test_elasticsearch_delete_by_paper_id(monkeypatch):
    from core import elasticsearch_store

    calls = []

    class Indices:
        async def exists(self, *, index):
            calls.append(("exists", index))
            return True

    class Client:
        indices = Indices()

        async def delete_by_query(self, **kwargs):
            calls.append(("delete", kwargs))
            return {"deleted": 4, "failures": []}

    async def get_client():
        return Client()

    monkeypatch.setattr(elasticsearch_store, "_AsyncElasticsearch", object)
    monkeypatch.setattr(elasticsearch_store, "_get_client", get_client)

    assert await elasticsearch_store.delete_paper_chunks("doi:10.1/test") == 4
    delete_call = calls[1][1]
    assert delete_call["body"] == {"query": {"term": {"paper_id": "doi:10.1/test"}}}
    assert delete_call["conflicts"] == "proceed"


@pytest.mark.asyncio
async def test_redis_prefix_invalidation_is_nonblocking_scan():
    from web.backend.redis_service import RedisService

    deleted = []

    class Client:
        async def scan_iter(self, *, match, count):
            assert match == "cache:rag_query:user:u-1:*"
            assert count == 200
            for key in ("k1", "k2"):
                yield key

        async def delete(self, *keys):
            deleted.extend(keys)

    service = RedisService()
    service._enabled = True
    service._available = True
    service._client = Client()

    assert await service.delete_prefix("cache:rag_query:user:u-1:") is True
    assert deleted == ["k1", "k2"]
