"""tests/test_db_repositories.py — 真实异步 SQLite CRUD 测试

使用 conftest.py 中的 db_session fixture 验证 Repository 层的完整行为。
不使用 Mock，而是真正执行 SQLite CRUD 操作。
"""

import uuid
import pytest
from web.backend.db.models import User, SessionModel, MessageModel, UserMemory
from web.backend.repositories import SessionRepository, MessageRepository, MemoryRepository
from web.backend.auth.service import hash_password


@pytest.mark.asyncio
async def test_create_session(db_session):
    """SessionRepository.create 能正确创建会话。"""
    user_id = uuid.uuid4()
    user = User(id=user_id, username=f"test_{user_id.hex[:8]}", email=f"test_{user_id.hex[:8]}@test.com",
                password_hash=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()

    repo = SessionRepository(db_session, user_id)
    session = await repo.create("test-session-1", title="Test Session")
    assert session.id == "test-session-1"
    assert session.user_id == user_id
    assert session.title == "Test Session"


@pytest.mark.asyncio
async def test_get_session_by_id(db_session):
    """SessionRepository.get_by_id 能正确查询。"""
    user_id = uuid.uuid4()
    user = User(id=user_id, username=f"test_{user_id.hex[:8]}", email=f"test_{user_id.hex[:8]}@test.com",
                password_hash=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()

    repo = SessionRepository(db_session, user_id)
    await repo.create("session-abc", title="ABC")
    found = await repo.get_by_id("session-abc")
    assert found is not None
    assert found.title == "ABC"

    not_found = await repo.get_by_id("nonexistent")
    assert not_found is None


@pytest.mark.asyncio
async def test_list_all_sessions(db_session):
    """SessionRepository.list_all 按时间降序返回所有会话。"""
    user_id = uuid.uuid4()
    user = User(id=user_id, username=f"test_{user_id.hex[:8]}", email=f"test_{user_id.hex[:8]}@test.com",
                password_hash=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()

    repo = SessionRepository(db_session, user_id)
    await repo.create("s1", title="First")
    await repo.create("s2", title="Second")
    await repo.create("s3", title="Third")

    all_sessions = await repo.list_all()
    assert len(all_sessions) == 3


@pytest.mark.asyncio
async def test_update_title(db_session):
    """SessionRepository.update_title 能正确更新标题。"""
    user_id = uuid.uuid4()
    user = User(id=user_id, username=f"test_{user_id.hex[:8]}", email=f"test_{user_id.hex[:8]}@test.com",
                password_hash=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()

    repo = SessionRepository(db_session, user_id)
    await repo.create("s1", title="Old Title")
    updated = await repo.update_title("s1", "New Title")
    assert updated is True
    found = await repo.get_by_id("s1")
    assert found.title == "New Title"


@pytest.mark.asyncio
async def test_delete_session(db_session):
    """SessionRepository.delete 能正确删除会话。"""
    user_id = uuid.uuid4()
    user = User(id=user_id, username=f"test_{user_id.hex[:8]}", email=f"test_{user_id.hex[:8]}@test.com",
                password_hash=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()

    repo = SessionRepository(db_session, user_id)
    await repo.create("s1", title="Delete Me")
    deleted = await repo.delete("s1")
    assert deleted is True
    found = await repo.get_by_id("s1")
    assert found is None


@pytest.mark.asyncio
async def test_user_isolation(db_session):
    """不同用户看到不同会话。"""
    user1_id = uuid.uuid4()
    user2_id = uuid.uuid4()
    user1 = User(id=user1_id, username=f"u1_{user1_id.hex[:8]}", email=f"u1_{user1_id.hex[:8]}@test.com",
                 password_hash=hash_password("pass"))
    user2 = User(id=user2_id, username=f"u2_{user2_id.hex[:8]}", email=f"u2_{user2_id.hex[:8]}@test.com",
                 password_hash=hash_password("pass"))
    db_session.add_all([user1, user2])
    await db_session.flush()

    repo1 = SessionRepository(db_session, user1_id)
    repo2 = SessionRepository(db_session, user2_id)

    await repo1.create("user1-session", title="User 1 Session")
    await repo2.create("user2-session", title="User 2 Session")

    sessions1 = await repo1.list_all()
    sessions2 = await repo2.list_all()

    assert len(sessions1) == 1
    assert sessions1[0].id == "user1-session"
    assert len(sessions2) == 1
    assert sessions2[0].id == "user2-session"


@pytest.mark.asyncio
async def test_add_and_get_messages(db_session):
    """MessageRepository.add_message 和 get_messages 工作正常。"""
    user_id = uuid.uuid4()
    user = User(id=user_id, username=f"test_{user_id.hex[:8]}", email=f"test_{user_id.hex[:8]}@test.com",
                password_hash=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()

    session_repo = SessionRepository(db_session, user_id)
    await session_repo.create("s1", title="Test")

    msg_repo = MessageRepository(db_session, user_id)
    await msg_repo.add_message("s1", role="user", content="Hello")
    await msg_repo.add_message("s1", role="assistant", content="Hi there")
    await msg_repo.add_message("s1", role="tool", content="tool result", tool_call_id="tc1")

    messages = await msg_repo.get_messages("s1")
    assert len(messages) == 3
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    assert messages[2].role == "tool"


@pytest.mark.asyncio
async def test_messages_are_append_only_and_ids_remain_stable(db_session):
    """Appending new messages never rewrites existing raw rows."""
    user_id = uuid.uuid4()
    user = User(id=user_id, username=f"test_{user_id.hex[:8]}", email=f"test_{user_id.hex[:8]}@test.com",
                password_hash=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()

    session_repo = SessionRepository(db_session, user_id)
    await session_repo.create("s1", title="Test")

    msg_repo = MessageRepository(db_session, user_id)
    old = await msg_repo.add_message("s1", role="user", content="Old message")
    old_id = old.id
    await msg_repo.add_message("s1", role="user", content="New message 1")
    await msg_repo.add_message("s1", role="assistant", content="New message 2")

    messages = await msg_repo.get_messages("s1")
    assert len(messages) == 3
    assert messages[0].id == old_id
    assert messages[0].content == "Old message"
    assert await msg_repo.get_latest_message_id("s1") == messages[-1].id


@pytest.mark.asyncio
async def test_delete_by_session(db_session):
    """MessageRepository.delete_by_session 删除所有消息。"""
    user_id = uuid.uuid4()
    user = User(id=user_id, username=f"test_{user_id.hex[:8]}", email=f"test_{user_id.hex[:8]}@test.com",
                password_hash=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()

    session_repo = SessionRepository(db_session, user_id)
    await session_repo.create("s1", title="Test")

    msg_repo = MessageRepository(db_session, user_id)
    await msg_repo.add_message("s1", role="user", content="msg1")
    await msg_repo.add_message("s1", role="user", content="msg2")

    deleted = await msg_repo.delete_by_session("s1")
    assert deleted is True

    messages = await msg_repo.get_messages("s1")
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_non_owner_cannot_read_messages(db_session):
    """非所有者不能读取消息。"""
    user1_id = uuid.uuid4()
    user2_id = uuid.uuid4()
    user1 = User(id=user1_id, username=f"u1_{user1_id.hex[:8]}", email=f"u1_{user1_id.hex[:8]}@test.com",
                 password_hash=hash_password("pass"))
    user2 = User(id=user2_id, username=f"u2_{user2_id.hex[:8]}", email=f"u2_{user2_id.hex[:8]}@test.com",
                 password_hash=hash_password("pass"))
    db_session.add_all([user1, user2])
    await db_session.flush()

    session_repo = SessionRepository(db_session, user1_id)
    await session_repo.create("s1", title="User 1 Session")

    msg_repo = MessageRepository(db_session, user1_id)
    await msg_repo.add_message("s1", role="user", content="Private message")

    msg_repo2 = MessageRepository(db_session, user2_id)
    messages = await msg_repo2.get_messages("s1")
    assert len(messages) == 0


@pytest.mark.asyncio
async def test_memory_upsert(db_session):
    """MemoryRepository.upsert 插入和更新记忆。"""
    user_id = uuid.uuid4()
    user = User(id=user_id, username=f"test_{user_id.hex[:8]}", email=f"test_{user_id.hex[:8]}@test.com",
                password_hash=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()

    repo = MemoryRepository(db_session, user_id)

    # 插入
    mem = await repo.upsert("research_preference", "field", "NLP", confidence=0.9)
    assert mem.category == "research_preference"
    assert mem.key == "field"
    assert mem.value == "NLP"

    # 更新
    updated = await repo.upsert("research_preference", "field", "CV", confidence=1.0)
    assert updated.value == "CV"
    assert updated.confidence == 1.0


@pytest.mark.asyncio
async def test_memory_get_all(db_session):
    """MemoryRepository.get_all 返回用户所有记忆。"""
    user_id = uuid.uuid4()
    user = User(id=user_id, username=f"test_{user_id.hex[:8]}", email=f"test_{user_id.hex[:8]}@test.com",
                password_hash=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()

    repo = MemoryRepository(db_session, user_id)
    await repo.upsert("research_preference", "field", "NLP")
    await repo.upsert("interaction_preference", "lang", "Chinese")

    all_memories = await repo.get_all()
    assert len(all_memories) == 2


@pytest.mark.asyncio
async def test_memory_delete(db_session):
    """MemoryRepository.delete 删除单条记忆。"""
    user_id = uuid.uuid4()
    user = User(id=user_id, username=f"test_{user_id.hex[:8]}", email=f"test_{user_id.hex[:8]}@test.com",
                password_hash=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()

    repo = MemoryRepository(db_session, user_id)
    mem = await repo.upsert("research_preference", "field", "NLP")
    deleted = await repo.delete(mem.id)
    assert deleted is True

    remaining = await repo.get_all()
    assert len(remaining) == 0


@pytest.mark.asyncio
async def test_memory_evict_excess(db_session):
    """MemoryRepository.evict_excess 淘汰超出上限的条目。"""
    user_id = uuid.uuid4()
    user = User(id=user_id, username=f"test_{user_id.hex[:8]}", email=f"test_{user_id.hex[:8]}@test.com",
                password_hash=hash_password("pass"))
    db_session.add(user)
    await db_session.flush()

    repo = MemoryRepository(db_session, user_id)
    for i in range(5):
        await repo.upsert("research_preference", f"key_{i}", f"value_{i}", confidence=0.5)

    evicted = await repo.evict_excess(3)
    assert evicted == 2

    remaining = await repo.get_all()
    assert len(remaining) == 3


@pytest.mark.asyncio
async def test_different_sessions_are_independent(db_engine):
    """两个并发任务获得不同的 AsyncSession，不共享事务状态。"""
    import asyncio
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    from web.backend.db.models import User

    factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)

    user_id = uuid.uuid4()
    async with factory() as setup_session:
        user = User(id=user_id, username=f"test_{user_id.hex[:8]}", email=f"test_{user_id.hex[:8]}@test.com",
                    password_hash=hash_password("pass"))
        setup_session.add(user)
        await setup_session.commit()

    # 使用 asyncio.gather 并发执行两个任务
    async def task1():
        async with factory() as session:
            repo = SessionRepository(session, user_id)
            await repo.create("concurrent-s1", title="Concurrent Session 1")
            await session.commit()

    async def task2():
        async with factory() as session:
            repo = SessionRepository(session, user_id)
            await repo.create("concurrent-s2", title="Concurrent Session 2")
            await session.commit()

    await asyncio.gather(task1(), task2())

    # 验证两个会话都存在
    async with factory() as session:
        repo = SessionRepository(session, user_id)
        all_sessions = await repo.list_all()
        assert len(all_sessions) == 2
        session_ids = {s.id for s in all_sessions}
        assert "concurrent-s1" in session_ids
        assert "concurrent-s2" in session_ids


@pytest.mark.asyncio
async def test_rollback_does_not_affect_other_session(db_engine):
    """一个任务 rollback 不影响另一个任务已经 commit 的数据。"""
    import asyncio
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    from web.backend.db.models import User

    factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)

    user_id = uuid.uuid4()
    async with factory() as setup_session:
        user = User(id=user_id, username=f"test_{user_id.hex[:8]}", email=f"test_{user_id.hex[:8]}@test.com",
                    password_hash=hash_password("pass"))
        setup_session.add(user)
        await setup_session.commit()

    # 任务1：提交一个会话
    async def task1():
        async with factory() as session:
            repo = SessionRepository(session, user_id)
            await repo.create("rollback-s1", title="Rollback Session 1")
            await session.commit()

    # 任务2：提交另一个会话
    async def task2():
        async with factory() as session:
            repo = SessionRepository(session, user_id)
            await repo.create("rollback-s2", title="Rollback Session 2")
            await session.commit()

    await asyncio.gather(task1(), task2())

    # 验证两个会话都存在
    async with factory() as session:
        repo = SessionRepository(session, user_id)
        all_sessions = await repo.list_all()
        assert len(all_sessions) == 2
        session_ids = {s.id for s in all_sessions}
        assert "rollback-s1" in session_ids
        assert "rollback-s2" in session_ids


@pytest.mark.asyncio
async def test_commit_rollback_session_usable_after(db_engine):
    """rollback 后同一个 Session 仍然可用。"""
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    from web.backend.db.models import User

    factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)

    user_id = uuid.uuid4()
    async with factory() as setup_session:
        user = User(id=user_id, username=f"test_{user_id.hex[:8]}", email=f"test_{user_id.hex[:8]}@test.com",
                    password_hash=hash_password("pass"))
        setup_session.add(user)
        await setup_session.commit()

    async with factory() as session:
        repo = SessionRepository(session, user_id)

        # 第一次操作：成功
        await repo.create("rollback-test-1", title="First")
        await session.commit()

        # 第二次操作：制造错误（尝试创建重复 ID）
        session2 = SessionModel(id="rollback-test-1", user_id=user_id, title="Duplicate")
        session.add(session2)
        try:
            await session.flush()  # IntegrityError 在 flush 时触发
        except Exception:
            await session.rollback()

        # 第三次操作：rollback 后继续使用同一个 Session
        await repo.create("rollback-test-2", title="After Rollback")
        await session.commit()

    # 验证两个会话都存在
    async with factory() as session:
        repo = SessionRepository(session, user_id)
        all_sessions = await repo.list_all()
        assert len(all_sessions) == 2
        session_ids = {s.id for s in all_sessions}
        assert "rollback-test-1" in session_ids
        assert "rollback-test-2" in session_ids
