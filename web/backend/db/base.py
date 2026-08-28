"""异步 SQLAlchemy 数据库基础设施

生产环境: postgresql+asyncpg
测试环境: sqlite+aiosqlite

注意：
- 业务运行时必须设置 DATABASE_URL（PostgreSQL）
- 测试通过 tests/conftest.py 在导入前设置 DATABASE_URL
- novare.session 等纯核心模块不依赖此文件
- 所有模块通过 get_session_factory() / get_engine() 获取实例，
  不得在模块级导入 engine 或 async_session_factory
"""

import logging
import os
from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)


def resolve_database_url(raw_url: str) -> str:
    """规范化数据库 URL，使其兼容 async driver。

    - postgresql://  →  postgresql+asyncpg://
    - postgresql+asyncpg://  →  保持不变
    - sqlite:///:memory:  →  sqlite+aiosqlite:///:memory:
    - 已带 +asyncpg 或 +aiosqlite 后缀的不重复添加
    """
    if raw_url.startswith("sqlite://"):
        return raw_url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    if raw_url.startswith("postgresql://") and "+asyncpg" not in raw_url:
        return raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return raw_url


def validate_database_url_for_alembic(raw_url: str) -> str:
    """验证并规范化 Alembic 使用的数据库 URL。

    返回规范化后的 URL。如果不是 PostgreSQL 则抛出 ValueError。
    """
    if not raw_url:
        raise ValueError(
            "DATABASE_URL 环境变量未设置。"
            "Alembic 需要 PostgreSQL 数据库，请设置 DATABASE_URL，"
            "例如：DATABASE_URL=postgresql://postgres:password@localhost:5432/research_agent"
        )
    resolved = resolve_database_url(raw_url)
    parsed = make_url(resolved)
    if parsed.get_backend_name() != "postgresql":
        raise ValueError(
            f"Alembic 仅支持 PostgreSQL 数据库，当前 URL 使用 {parsed.get_backend_name()} 后端。"
        )
    return resolved


DATABASE_URL_RAW = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or ""

# 如果缺少 DATABASE_URL，不创建引擎
# 业务运行时会在首次访问 get_engine()/get_session_factory() 时触发 RuntimeError
# 测试通过 tests/conftest.py 在导入前设置 DATABASE_URL
if DATABASE_URL_RAW:
    DATABASE_URL = resolve_database_url(DATABASE_URL_RAW)
else:
    DATABASE_URL = ""

# 延迟初始化引擎和 Session 工厂
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker] = None


def get_engine() -> AsyncEngine:
    """获取或创建异步引擎。缺少 DATABASE_URL 时抛出 RuntimeError。

    注意：不要在模块级保存返回值。每次需要时都应调用此函数，
    以确保 dispose_engine() 后下次调用能获得新引擎。
    """
    global _engine
    if _engine is None:
        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL 环境变量未设置。"
                "请在 .env 文件中配置，例如：DATABASE_URL=postgresql://postgres:password@localhost:5432/research_agent"
            )
        _engine = create_async_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            echo=False,
        )
    return _engine


def get_session_factory() -> async_sessionmaker:
    """获取或创建异步 Session 工厂。缺少 DATABASE_URL 时抛出 RuntimeError。

    注意：不要在模块级保存返回值。每次需要时都应调用此函数，
    以确保 dispose_engine() 后下次调用能获得新 factory。
    """
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


class Base(DeclarativeBase):
    pass


async def get_db():
    """FastAPI 异步依赖 — yield 一个 AsyncSession，请求结束自动关闭。"""
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def dispose_engine():
    """应用关闭时调用，释放连接池。"""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
    _session_factory = None
