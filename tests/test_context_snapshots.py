"""Durable context snapshots backed by immutable raw messages."""

from pathlib import Path
from unittest.mock import patch
import uuid

import pytest
from sqlalchemy import func, select

from novare.session import Session
from web.backend.agent_service import AgentService
from web.backend.auth.service import hash_password
from web.backend.db.models import ContextSnapshot, MessageModel, SessionModel, User
from web.backend.repositories import (
    ContextSnapshotRepository,
    MessageRepository,
    SessionRepository,
)


async def _create_user_and_session(factory, session_id="context-session"):
    user_id = uuid.uuid4()
    async with factory() as db:
        db.add(User(
            id=user_id,
            username=f"ctx_{user_id.hex[:8]}",
            email=f"ctx_{user_id.hex[:8]}@test.com",
            password_hash=hash_password("pass"),
        ))
        await db.flush()
        await SessionRepository(db, user_id).create(session_id, title="Context")
        await db.commit()
    return user_id


@pytest.mark.asyncio
async def test_compaction_appends_raw_messages_and_saves_snapshot(db_session_factory):
    user_id = await _create_user_and_session(db_session_factory)
    async with db_session_factory() as db:
        repo = MessageRepository(db, user_id)
        old = await repo.add_message("context-session", "user", "old raw")
        await db.commit()
        old_id = old.id

    working = Session(session_id="context-session", workspace=Path("."))
    working.messages = [
        {
            "role": "assistant",
            "content": "summary",
            "_compacted": True,
            "_compaction_meta": {"schema_version": 2, "strategy": "hybrid_llm"},
        },
        {"role": "user", "content": "new raw"},
        {"role": "assistant", "content": "new answer"},
    ]
    raw_turn = working.messages[1:]
    service = AgentService()

    with patch("web.backend.agent_service.get_session_factory", return_value=db_session_factory):
        await service.persist_web_turn(
            working, str(user_id), raw_turn, compacted=True, title="Context"
        )

    async with db_session_factory() as db:
        messages = await MessageRepository(db, user_id).get_messages("context-session")
        snapshot = await ContextSnapshotRepository(db, user_id).get_by_session("context-session")
        assert [message.content for message in messages] == [
            "old raw",
            "new raw",
            "new answer",
        ]
        assert messages[0].id == old_id
        assert snapshot.snapshot_data == working.messages
        assert snapshot.compacted_through_message_id == messages[-1].id
        assert snapshot.schema_version == 2


@pytest.mark.asyncio
async def test_reload_uses_snapshot_plus_only_messages_after_cursor(db_session_factory):
    user_id = await _create_user_and_session(db_session_factory)
    async with db_session_factory() as db:
        msg_repo = MessageRepository(db, user_id)
        await msg_repo.add_message("context-session", "user", "covered raw")
        covered = await msg_repo.add_message("context-session", "assistant", "covered answer")
        await ContextSnapshotRepository(db, user_id).upsert(
            "context-session",
            [{"role": "assistant", "content": "summary", "_compacted": True}],
            covered.id,
            estimated_tokens=10,
        )
        await msg_repo.add_message("context-session", "user", "after cursor")
        await db.commit()

    service = AgentService()
    with patch("web.backend.agent_service.get_session_factory", return_value=db_session_factory):
        loaded = await service.load_session("context-session", str(user_id))

    assert loaded.messages == [
        {"role": "assistant", "content": "summary", "_compacted": True},
        {"role": "user", "content": "after cursor"},
    ]


@pytest.mark.asyncio
async def test_reload_without_snapshot_uses_all_raw_messages(db_session_factory):
    user_id = await _create_user_and_session(db_session_factory)
    async with db_session_factory() as db:
        repo = MessageRepository(db, user_id)
        await repo.add_message("context-session", "user", "first")
        await repo.add_message("context-session", "assistant", "second")
        await db.commit()

    service = AgentService()
    with patch("web.backend.agent_service.get_session_factory", return_value=db_session_factory):
        loaded = await service.load_session("context-session", str(user_id))

    assert [message["content"] for message in loaded.messages] == ["first", "second"]


@pytest.mark.asyncio
async def test_snapshot_failure_rolls_back_raw_messages_and_cursor(db_session_factory):
    user_id = await _create_user_and_session(db_session_factory)
    async with db_session_factory() as db:
        message = await MessageRepository(db, user_id).add_message(
            "context-session", "user", "old raw"
        )
        old_cursor = message.id
        await ContextSnapshotRepository(db, user_id).upsert(
            "context-session",
            [{"role": "assistant", "content": "old summary", "_compacted": True}],
            old_cursor,
        )
        await db.commit()

    working = Session(session_id="context-session", workspace=Path("."))
    working.messages = [{"role": "assistant", "content": "new summary", "_compacted": True}]
    service = AgentService()
    original_upsert = ContextSnapshotRepository.upsert

    async def fail_after_flush(repo, *args, **kwargs):
        await original_upsert(repo, *args, **kwargs)
        raise RuntimeError("snapshot failed")

    with (
        patch("web.backend.agent_service.get_session_factory", return_value=db_session_factory),
        patch.object(
            ContextSnapshotRepository,
            "upsert",
            new=fail_after_flush,
        ),
    ):
        with pytest.raises(RuntimeError, match="snapshot failed"):
            await service.persist_web_turn(
                working,
                str(user_id),
                [{"role": "user", "content": "must roll back"}],
                compacted=True,
                title="Context",
            )

    async with db_session_factory() as db:
        count = await db.scalar(
            select(func.count()).select_from(MessageModel).where(
                MessageModel.session_id == "context-session"
            )
        )
        assert count == 1
        snapshot = await ContextSnapshotRepository(db, user_id).get_by_session(
            "context-session"
        )
        assert snapshot.compacted_through_message_id == old_cursor
        assert snapshot.snapshot_data[0]["content"] == "old summary"


@pytest.mark.asyncio
async def test_snapshot_is_user_scoped_and_memory_cursor_is_independent(db_session_factory):
    owner_id = await _create_user_and_session(db_session_factory)
    other_id = uuid.uuid4()
    async with db_session_factory() as db:
        db.add(User(
            id=other_id,
            username=f"ctx_{other_id.hex[:8]}",
            email=f"ctx_{other_id.hex[:8]}@test.com",
            password_hash=hash_password("pass"),
        ))
        session = await SessionRepository(db, owner_id).get_by_id("context-session")
        session.last_extracted_message_id = 77
        message = await MessageRepository(db, owner_id).add_message(
            "context-session", "user", "raw"
        )
        await ContextSnapshotRepository(db, owner_id).upsert(
            "context-session",
            [{"role": "assistant", "content": "summary", "_compacted": True}],
            message.id,
        )
        await db.commit()

    async with db_session_factory() as db:
        assert await ContextSnapshotRepository(db, other_id).get_by_session("context-session") is None
        assert await ContextSnapshotRepository(db, other_id).upsert(
            "context-session", [], 1
        ) is None
        session = await SessionRepository(db, owner_id).get_by_id("context-session")
        assert session.last_extracted_message_id == 77


@pytest.mark.asyncio
async def test_session_delete_cascades_snapshot(db_session_factory):
    user_id = await _create_user_and_session(db_session_factory)
    async with db_session_factory() as db:
        message = await MessageRepository(db, user_id).add_message(
            "context-session", "user", "raw"
        )
        await ContextSnapshotRepository(db, user_id).upsert(
            "context-session", [{"role": "assistant", "content": "summary"}], message.id
        )
        await db.commit()

    async with db_session_factory() as db:
        assert await SessionRepository(db, user_id).delete("context-session") is True
        await db.commit()

    async with db_session_factory() as db:
        assert await db.get(ContextSnapshot, "context-session") is None
        assert await db.get(SessionModel, "context-session") is None
