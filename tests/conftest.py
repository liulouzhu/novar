"""共享测试 fixtures — 异步 SQLite 数据库 + FastAPI 测试客户端"""

import asyncio
import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# mcp-server 目录
MCP_ROOT = PROJECT_ROOT / "mcp-server"
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

# 在导入业务模块前设置测试数据库 URL
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

from sqlalchemy import event as _sa_event  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession  # noqa: E402
from web.backend.db.base import Base  # noqa: E402


# ── 防逃逸 fixture：禁止默认测试访问真实 Milvus ──────────────────

# 需要监控的 PyMilvus API 列表
_MILVUS_FORBIDDEN_APIS = [
    ("connections.connect", "pymilvus", "connections", "connect"),
    ("connections.disconnect", "pymilvus", "connections", "disconnect"),
    ("connections.list_connections", "pymilvus", "connections", "list_connections"),
    ("utility.has_collection", "pymilvus", "utility", "has_collection"),
    ("Collection", "pymilvus", "Collection", None),
    ("Collection.load", "pymilvus", "Collection", "load"),
    ("Collection.search", "pymilvus", "Collection", "search"),
    ("Collection.insert", "pymilvus", "Collection", "insert"),
    ("Collection.delete", "pymilvus", "Collection", "delete"),
    ("Collection.flush", "pymilvus", "Collection", "flush"),
    ("Collection.create_index", "pymilvus", "Collection", "create_index"),
]


def _make_recorder(name, attempted_calls):
    """创建一个记录违规调用的函数。"""
    def recorder(*args, **kwargs):
        attempted_calls.append((name, args, kwargs))
        raise RuntimeError(f"Forbidden Milvus call: {name}")
    return recorder


@pytest.fixture(autouse=True)
def forbid_real_milvus_network(monkeypatch):
    """默认单元测试禁止调用真实 Milvus 网络 API。

    记录所有违规调用，teardown 时断言没有违规。
    测试自身需要验证连接行为时，在测试内用 MagicMock 覆盖 recorder。
    fixture teardown 自动恢复所有 patch，不跨测试污染。
    """
    attempted_calls = []

    try:
        import pymilvus as _pymilvus
        from pymilvus import connections as _connections
        from pymilvus import utility as _utility

        # Patch connections API
        monkeypatch.setattr(_connections, "connect",
                            _make_recorder("connections.connect", attempted_calls))
        monkeypatch.setattr(_connections, "disconnect",
                            _make_recorder("connections.disconnect", attempted_calls))
        monkeypatch.setattr(_connections, "list_connections",
                            _make_recorder("connections.list_connections", attempted_calls))

        # Patch utility API
        monkeypatch.setattr(_utility, "has_collection",
                            _make_recorder("utility.has_collection", attempted_calls))

        # Patch Collection class — 需要保留构造能力但记录调用
        _OriginalCollection = _pymilvus.Collection

        class _RecordingCollection:
            """包装 Collection，记录方法调用但不执行真实操作。"""
            def __init__(self, *args, **kwargs):
                attempted_calls.append(("Collection.__init__", args, kwargs))
                raise RuntimeError("Forbidden Milvus call: Collection construction")

        monkeypatch.setattr(_pymilvus, "Collection", _RecordingCollection)

        # Patch 已缓存的调用方引用（core.vector_store）
        try:
            import core.vector_store as _cvs
            monkeypatch.setattr(_cvs, "connections", _connections)
            monkeypatch.setattr(_cvs, "utility", _utility)
            monkeypatch.setattr(_cvs, "Collection", _RecordingCollection)
        except ImportError:
            pass

        # Patch 已缓存的调用方引用（episodic_memory.vector_store）
        try:
            from web.backend.episodic_memory import vector_store as _evs
            # 这些是函数内部 import 的，不需要 patch 模块级引用
        except ImportError:
            pass

    except ImportError:
        # pymilvus 未安装时跳过
        pass

    yield

    # teardown: 断言没有违规调用
    if attempted_calls:
        names = list(dict.fromkeys(name for name, _, _ in attempted_calls))
        pytest.fail(
            "Unit test attempted forbidden Milvus operations: "
            + ", ".join(names)
        )


@pytest.fixture(autouse=True)
def mock_es_search_for_tests():
    """默认单元测试禁止连接真实 Elasticsearch。

    mock _es_search 返回空结果元组，避免测试中触发 ES 连接。
    """
    from unittest.mock import patch, AsyncMock
    with patch("tools.rag_query._es_search", new_callable=AsyncMock, return_value=([], True, None)):
        yield


def _enable_sqlite_foreign_keys(dbapi_conn, connection_record):
    """在 SQLite 连接上启用外键约束。"""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


@pytest.fixture
def tmp_workspace(tmp_path):
    """创建临时工作空间"""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / ".novare").mkdir()
    return ws


@pytest.fixture
def tmp_data_dir(tmp_path):
    """创建临时数据目录"""
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest_asyncio.fixture
async def db_engine():
    """创建异步 SQLite 测试引擎，启用外键约束。"""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    # 启用 SQLite 外键约束
    _sa_event.listen(engine.sync_engine, "connect", _enable_sqlite_foreign_keys)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine):
    """创建异步测试 Session，每个测试自动回滚"""
    session_factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def db_session_factory(db_engine):
    """返回 async_sessionmaker 工厂，供需要创建短生命周期 Session 的测试使用"""
    return async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def test_user(db_session):
    """创建一个测试用户"""
    from web.backend.db.models import User
    from web.backend.auth.service import hash_password
    user_id = uuid.uuid4()
    user = User(
        id=user_id,
        username=f"testuser_{user_id.hex[:8]}",
        email=f"test_{user_id.hex[:8]}@test.com",
        password_hash=hash_password("testpassword"),
    )
    db_session.add(user)
    await db_session.flush()
    return user
