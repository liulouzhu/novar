"""FastAPI 应用入口 — 管理 Agent 生命周期，注册路由"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 在 import 业务模块前加载 .env，确保 DATABASE_URL 等变量可用
from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from web.backend.agent_service import AgentService  # noqa: E402
from web.backend.db.base import Base, dispose_engine, get_engine, get_session_factory  # noqa: E402
from web.backend.redis_service import redis_service  # noqa: E402
from web.backend.sandbox.manager import (  # noqa: E402
    IDLE_TIMEOUT,
    sandbox_manager,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
web_logger = logging.getLogger("novare.web")

# 全局 AgentService 单例
agent_service = AgentService()


async def _idle_cleanup_loop():
    """Periodically evict sandbox containers that exceed IDLE_TIMEOUT."""
    interval = max(60, IDLE_TIMEOUT // 4)
    while True:
        await asyncio.sleep(interval)
        try:
            sandbox_manager.cleanup_idle()
        except Exception:
            web_logger.warning("Sandbox idle cleanup error", exc_info=True)


async def _paper_cleanup_loop():
    """Retry durable paper cleanup outbox jobs."""
    from web.backend.paper_cleanup import process_pending_cleanup_jobs

    while True:
        try:
            attempted = await process_pending_cleanup_jobs()
            if attempted:
                web_logger.info("Processed %d paper cleanup job(s)", attempted)
        except Exception:
            web_logger.warning("Paper cleanup loop error", exc_info=True)
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭生命周期"""
    os.chdir(PROJECT_ROOT)

    # 导入所有模型，确保 Base.metadata 包含所有表定义
    import web.backend.db.models  # noqa: F401

    # 通过 Alembic 管理数据库结构，不再执行 create_all
    # 如果需要自动建表（仅开发用途），取消下面的注释：
    # async with engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.create_all)
    web_logger.info("DB engine ready (use 'alembic upgrade head' to create tables)")

    # Clean up stale sandbox containers, then start idle watcher
    sandbox_manager.startup()
    cleanup_task = asyncio.create_task(_idle_cleanup_loop())
    web_logger.info("Sandbox idle cleanup task started (interval=%ds)", max(60, IDLE_TIMEOUT // 4))

    await agent_service.initialize()

    # ── Redis 初始化（可选，失败不阻止启动） ──
    if agent_service.config:
        try:
            await redis_service.initialize(
                enabled=agent_service.config.redis_enabled,
                url=agent_service.config.redis_url,
            )
            redis_status = "disabled"
            if agent_service.config.redis_enabled:
                redis_status = "ok" if redis_service.is_available else "unavailable"
            web_logger.info(
                "Redis health: enabled=%s available=%s status=%s",
                agent_service.config.redis_enabled,
                redis_service.is_available,
                redis_status,
            )
        except Exception:
            web_logger.warning("Redis init failed (non-fatal), continuing without Redis", exc_info=True)

    paper_cleanup_task = asyncio.create_task(_paper_cleanup_loop())
    web_logger.info("Paper cleanup retry task started")

    # ── 多渠道接入系统 ──
    channel_tasks: list[asyncio.Task] = []
    if agent_service.config and agent_service.config.channels_enabled:
        from novare.channels.bus import MessageBus
        from novare.channels.manager import ChannelManager
        from novare.channels.adapter import AgentAdapter

        bus = MessageBus()
        manager = ChannelManager(agent_service.config.channels, bus)
        adapter = AgentAdapter(
            bus=bus,
            agent_service=agent_service,
            db_session_factory=get_session_factory(),
            default_user_id=agent_service.config.channel_default_user_id or None,
        )

        channel_tasks.append(asyncio.create_task(manager.start_all()))
        channel_tasks.append(asyncio.create_task(adapter.run()))
        web_logger.info("Channel system started: %s", list(manager.channels.keys()))

    try:
        yield
    finally:
        paper_cleanup_task.cancel()
        try:
            await paper_cleanup_task
        except asyncio.CancelledError:
            pass
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        try:
            sandbox_manager.shutdown()
        except Exception:
            web_logger.warning("Sandbox shutdown error (non-fatal)", exc_info=True)

        if channel_tasks:
            await adapter.stop()
            await manager.stop_all()
            for t in channel_tasks:
                t.cancel()
            for t in channel_tasks:
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        await redis_service.close()

        try:
            await agent_service.shutdown()
        except Exception:
            web_logger.warning("Shutdown error (non-fatal)", exc_info=True)

        # 释放异步数据库引擎连接池
        await dispose_engine()


app = FastAPI(
    title="Novare Web API",
    description="智能科研助手 Web 接口",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — 开发时允许前端 dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
from web.backend.auth.router import router as auth_router  # noqa: E402
from web.backend.routes.chat import router as chat_router  # noqa: E402
from web.backend.routes.graph import router as graph_router  # noqa: E402
from web.backend.routes.papers import router as papers_router  # noqa: E402
from web.backend.routes.sessions import router as sessions_router  # noqa: E402
from web.backend.routes.upload import router as upload_router  # noqa: E402
from web.backend.routes.memories import router as memories_router  # noqa: E402
from web.backend.routes.episodic_memories import router as episodic_memories_router  # noqa: E402

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(sessions_router)
app.include_router(papers_router)
app.include_router(graph_router)
app.include_router(upload_router)
app.include_router(memories_router)
app.include_router(episodic_memories_router)


@app.get("/api/health")
async def health():
    result: dict = {
        "status": "ok",
        "model": agent_service.config.model if agent_service.config else "not ready",
        "redis": {"enabled": False, "available": False, "status": "disabled"},
        "database": {"status": "ok"},
        "sandbox": {"available": sandbox_manager.client is not None},
    }
    try:
        if redis_service._enabled:
            result["redis"]["enabled"] = True
            result["redis"]["available"] = redis_service.is_available
            try:
                pong = await redis_service.ping()
                result["redis"]["status"] = "ok" if pong else "unavailable"
            except Exception:
                result["redis"]["status"] = "unavailable"

        # DB 子检查（异步）
        try:
            async with get_session_factory()() as db:
                from sqlalchemy import text
                await db.execute(text("SELECT 1"))
        except Exception:
            result["database"]["status"] = "error"
    except Exception:
        pass
    return result


@app.get("/api/skills")
async def list_skills():
    """返回所有可用 skill 列表（从文件系统动态发现）"""
    from novare.skill import discover_skills

    skill_dirs = []
    if agent_service.config:
        skill_dirs = list(agent_service.config.skill_dirs)

    skills = discover_skills(skill_dirs)
    return [
        {"name": s.name, "description": s.description}
        for s in skills
    ]
