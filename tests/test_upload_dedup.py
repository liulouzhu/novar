"""Content-addressed upload and shared paper identity tests."""

from __future__ import annotations

import asyncio
from io import BytesIO
import json
import uuid
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import func, select
from starlette.datastructures import Headers, UploadFile

from novare.file_storage import store_upload_stream
from web.backend.db.models import FileBlob, Paper, PaperFile, User, UserPaper, UserUpload
from web.backend.repositories.paper_repo import PaperRepository
from web.backend.repositories.upload_repo import UploadRepository, link_paper_file
from web.backend.routes.upload import upload_file


def _user(name: str) -> User:
    return User(
        id=uuid.uuid4(),
        username=name,
        email=f"{name}@example.com",
        password_hash="not-used",
    )


def _upload(name: str, content: bytes) -> UploadFile:
    return UploadFile(
        file=BytesIO(content),
        filename=name,
        headers=Headers({"content-type": "application/pdf"}),
    )


@pytest.mark.asyncio
async def test_cross_user_uploads_share_one_blob(db_session, tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(tmp_path))
    user_a = _user("dedup-a")
    user_b = _user("dedup-b")
    db_session.add_all([user_a, user_b])
    await db_session.flush()

    content = b"%PDF-1.7\nidentical paper bytes\n%%EOF"
    first = await upload_file(_upload("first.pdf", content), user_a, db_session)
    second = await upload_file(_upload("renamed.pdf", content), user_b, db_session)
    repeated = await upload_file(_upload("again.pdf", content), user_a, db_session)

    blob_count = await db_session.scalar(select(func.count()).select_from(FileBlob))
    upload_count = await db_session.scalar(select(func.count()).select_from(UserUpload))
    assert blob_count == 1
    assert upload_count == 2
    assert first.upload_id != second.upload_id
    assert repeated.upload_id == first.upload_id
    assert repeated.already_uploaded is True
    assert first.file_path is None


@pytest.mark.asyncio
async def test_blob_storage_is_idempotent_under_parallel_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(tmp_path))
    content = b"same bytes written concurrently" * 100

    stored = await asyncio.gather(
        store_upload_stream(_upload("a.pdf", content)),
        store_upload_stream(_upload("b.pdf", content)),
    )

    assert stored[0].sha256 == stored[1].sha256
    assert stored[0].storage_path == stored[1].storage_path
    with open(stored[0].storage_path, "rb") as result:
        assert result.read() == content


@pytest.mark.asyncio
async def test_upload_id_is_user_scoped(db_session, tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(tmp_path))
    user_a = _user("scope-a")
    user_b = _user("scope-b")
    db_session.add_all([user_a, user_b])
    await db_session.flush()

    response = await upload_file(_upload("private.pdf", b"private-content"), user_a, db_session)
    upload_id = uuid.UUID(response.upload_id)

    assert await UploadRepository(db_session, user_a.id).get_owned(upload_id) is not None
    assert await UploadRepository(db_session, user_b.id).get_owned(upload_id) is None


@pytest.mark.asyncio
async def test_associated_user_can_view_shared_private_paper(db_session, tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(tmp_path))
    user_a = _user("owner-a")
    user_b = _user("reader-b")
    db_session.add_all([user_a, user_b])
    await db_session.flush()

    upload = await upload_file(_upload("shared.pdf", b"shared-private-paper"), user_a, db_session)
    owned = await UploadRepository(db_session, user_a.id).get_owned(uuid.UUID(upload.upload_id))
    assert owned is not None

    paper = Paper(
        id=f"upload:sha256:{owned.sha256}",
        title="Shared private paper",
        visibility="private",
        created_by_user_id=user_a.id,
    )
    db_session.add(paper)
    await db_session.flush()
    await link_paper_file(
        db_session,
        paper_id=paper.id,
        blob_id=owned.blob_id,
        source="upload",
        access_scope="private",
    )
    db_session.add(UserPaper(
        user_id=user_b.id,
        paper_id=paper.id,
        relation_type="uploaded",
        has_fulltext_access=True,
        source="upload",
    ))
    await db_session.flush()

    assert await PaperRepository(db_session).get_visible(paper.id, user_b.id) is not None
    paper_files = await db_session.scalar(select(func.count()).select_from(PaperFile))
    assert paper_files == 1


@pytest.mark.asyncio
async def test_external_identifiers_resolve_to_one_paper(db_session):
    from core.database import upsert_paper

    first = {
        "id": "https://doi.org/10.1234/Example.Paper",
        "identifiers": ["arxiv:2401.12345v2"],
        "title": "Unified identity",
        "authors": [],
        "visibility": "public",
    }
    first_id = await upsert_paper(db_session, first)
    second_id = await upsert_paper(db_session, {
        "id": "https://arxiv.org/abs/2401.12345",
        "title": "Unified identity from arXiv",
        "authors": [],
        "visibility": "public",
    })

    assert first_id == "doi:10.1234/example.paper"
    assert second_id == first_id
    assert await db_session.scalar(select(func.count()).select_from(Paper)) == 1


def test_extract_document_identifiers_uses_front_matter_only():
    from core.paper_id import extract_document_identifiers

    text = """# A Paper
DOI: 10.5555/ABC.123
arXiv: 2402.01234v3

# References
Other work https://doi.org/10.9999/not-this-paper
"""
    assert extract_document_identifiers(text) == [
        "doi:10.5555/abc.123",
        "arxiv:2402.01234",
    ]


@pytest.mark.asyncio
async def test_second_user_reuses_parsed_chunks(
    db_session_factory,
    tmp_path,
    monkeypatch,
):
    from core import pdf_parser
    from web.backend.db import base as db_base
    from web.backend.db.models import Chunk
    from tools import knowledge_graph, paper_parse
    from core import elasticsearch_store

    monkeypatch.setenv("RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(db_base, "get_session_factory", lambda: db_session_factory)

    async with db_session_factory() as setup:
        user_a = _user("parse-a")
        user_b = _user("parse-b")
        setup.add_all([user_a, user_b])
        await setup.commit()

    content = b"%PDF-1.7\nsame uploaded paper\n%%EOF"
    async with db_session_factory() as session:
        first_upload = await upload_file(_upload("paper-a.pdf", content), user_a, session)
    async with db_session_factory() as session:
        second_upload = await upload_file(_upload("paper-b.pdf", content), user_b, session)

    markdown = """# A Deduplicated Research Paper
DOI: 10.4242/shared.paper

# Abstract
This paper contains enough text to exercise parsing and cross-user deduplication. """ * 3
    embed = AsyncMock(side_effect=lambda texts: [[0.1, 0.2] for _ in texts])
    monkeypatch.setattr(pdf_parser, "parse_pdf_to_markdown", lambda _: markdown)
    monkeypatch.setattr(paper_parse, "embed_batch_async", embed)
    monkeypatch.setattr(paper_parse, "get_embedding_dim", lambda: 2)
    monkeypatch.setattr(paper_parse, "_milvus_insert", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        elasticsearch_store,
        "bulk_upsert_chunks",
        AsyncMock(side_effect=lambda docs: {"success": len(docs), "errors": []}),
    )
    monkeypatch.setattr(knowledge_graph, "extract_from_abstract_sync", AsyncMock(return_value="{}"))

    first_result = json.loads(await paper_parse.handle_paper_parse(
        {"upload_id": first_upload.upload_id},
        user_id=str(user_a.id),
    ))
    second_result = json.loads(await paper_parse.handle_paper_parse(
        {"upload_id": second_upload.upload_id},
        user_id=str(user_b.id),
    ))

    assert first_result["data"]["paper_id"] == "doi:10.4242/shared.paper"
    assert second_result["data"]["paper_id"] == first_result["data"]["paper_id"]
    assert second_result["data"]["already_parsed"] is True
    assert second_result["data"]["deduplicated"] is True
    assert embed.await_count == 1

    async with db_session_factory() as verify:
        assert await verify.scalar(select(func.count()).select_from(FileBlob)) == 1
        assert await verify.scalar(select(func.count()).select_from(Paper)) == 1
        assert await verify.scalar(select(func.count()).select_from(PaperFile)) == 1
        assert await verify.scalar(select(func.count()).select_from(Chunk)) == second_result["data"]["chunk_count"]
        assert await verify.scalar(select(func.count()).select_from(UserPaper)) == 2
