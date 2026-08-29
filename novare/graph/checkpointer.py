"""novare/graph/checkpointer.py — checkpoint 持久化工厂

阶段 2：LangGraph checkpoint 负责执行现场与恢复；用户可见对话仍以
PostgreSQL 消息表为唯一数据源，Redis 继续负责锁与取消。

thread_id 采用 turn 级隔离：

    thread_id = f"{user_id}:{session_id}:{run_id}"

与 RecoveryState 的 run_id 生命周期一一对应（recovery_states 表里的
快照存有 thread_id，显式恢复时据此回到同一个 thread 继续）。
不采用会话级 thread 累积多轮消息，避免与消息表形成双数据源漂移。

按数据库 URL 与平台选择存储后端：
- postgresql://... + Linux/macOS → AsyncPostgresSaver
- postgresql://... + Windows      → 同步 PostgresSaver（LangGraph 在
  async graph 中自动以线程池适配）。原因：psycopg async 要求
  SelectorEventLoop，而本项目 MCP stdio 传输依赖 ProactorEventLoop
  （subprocess 支持），两者在 Windows 上互斥；同步 saver 规避冲突。
- None / 其他 → InMemorySaver（测试 / 单进程回退）
"""

from __future__ import annotations

import asyncio
import logging
import sys

logger = logging.getLogger("novare.graph.checkpointer")


def build_thread_id(user_id: str | None, session_id: str, run_id: str) -> str:
    """构造 turn 级 checkpoint thread_id。

    run_id 由调用方从 RecoveryState.run_id 取（每次 run_turn 新建，
    显式恢复时复用既有 run_id 回到同一 thread）。
    """
    user_part = user_id or "anonymous"
    return f"{user_part}:{session_id}:{run_id}"


def _normalize_postgres_url(url: str) -> str:
    # asyncpg 风格的 +asyncpg 驱动声明对 psycopg 无效，统一剥掉
    return url.replace("+asyncpg://", "://") if "+asyncpg://" in url else url


async def build_checkpointer(database_url: str | None = None):
    """按配置构造 checkpointer 并完成初始化（建表等）。

    生产使用 DATABASE_URL（PostgreSQL）；未配置时 InMemorySaver 仅限
    测试和单进程场景，进程退出即丢。
    """
    if database_url and database_url.startswith(("postgresql://", "postgres://")):
        conn_string = _normalize_postgres_url(database_url)
        try:
            if sys.platform == "win32":
                saver = await _build_sync_postgres_saver(conn_string)
            else:
                saver = await _build_async_postgres_saver(conn_string)
        except ImportError as exc:  # psycopg 未安装时给出可行动的错误
            raise RuntimeError(
                "使用 PostgreSQL checkpoint 需要安装 "
                "langgraph-checkpoint-postgres 和 psycopg："
                "pip install langgraph-checkpoint-postgres 'psycopg[binary]'"
            ) from exc
        logger.info("LangGraph checkpoint: %s ready", type(saver).__name__)
        return saver

    from langgraph.checkpoint.memory import InMemorySaver

    logger.info("LangGraph checkpoint: InMemorySaver (未配置数据库，恢复能力仅限本进程)")
    return InMemorySaver()


async def _build_async_postgres_saver(conn_string: str):
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    saver = AsyncPostgresSaver.from_conn_string(conn_string)
    # from_conn_string 返回 async context manager，进入后才能使用
    await saver.__aenter__()
    await saver.setup()
    return saver


class _SyncSaverAsyncBridge:
    """把同步 BaseCheckpointSaver 的 async 方法面桥接到线程池。

    LangGraph 1.2 在 ainvoke 路径调用 checkpointer 的 a* 方法时，
    同步 saver 不会自动适配（NotImplementedError）。本桥接把全部
    async 请求转发到线程池执行同步方法 —— Windows 上因此可以避开
    psycopg async 的 SelectorEventLoop 约束（见模块 docstring）。

    继承自 saver 实例的动态子类以通过 ensure_valid_checkpointer 的
    isinstance(BaseCheckpointSaver) 校验。
    """

    @staticmethod
    def wrap(sync_saver):
        from langgraph.checkpoint.base import BaseCheckpointSaver

        sync_cls = type(sync_saver)

        class _Bridge(BaseCheckpointSaver):
            def __init__(self):
                super().__init__(serde=sync_saver.serde)
                self._sync = sync_saver

            # ── 同步方法：直接转发 ──
            def get_tuple(self, config):
                return self._sync.get_tuple(config)

            def list(self, config=None, *, filter=None, before=None, limit=None):
                return self._sync.list(config, filter=filter, before=before, limit=limit)

            def put(self, config, checkpoint, metadata, new_versions):
                return self._sync.put(config, checkpoint, metadata, new_versions)

            def put_writes(self, config, writes, task_id, task_path=""):
                return self._sync.put_writes(config, writes, task_id, task_path)

            def delete_thread(self, thread_id):
                return self._sync.delete_thread(thread_id)

            # ── async 方法：线程池桥接 ──
            async def aget_tuple(self, config):
                return await asyncio.to_thread(self._sync.get_tuple, config)

            async def alist(self, config=None, *, filter=None, before=None, limit=None):
                def _collect():
                    return list(self._sync.list(config, filter=filter, before=before, limit=limit))
                for item in await asyncio.to_thread(_collect):
                    yield item

            async def aput(self, config, checkpoint, metadata, new_versions):
                return await asyncio.to_thread(
                    self._sync.put, config, checkpoint, metadata, new_versions,
                )

            async def aput_writes(self, config, writes, task_id, task_path=""):
                return await asyncio.to_thread(
                    self._sync.put_writes, config, writes, task_id, task_path,
                )

            async def adelete_thread(self, thread_id):
                return await asyncio.to_thread(self._sync.delete_thread, thread_id)

        _Bridge.__name__ = f"AsyncBridge[{sync_cls.__name__}]"
        bridge = _Bridge()
        bridge._cm = getattr(sync_saver, "_cm", None)
        return bridge


async def _build_sync_postgres_saver(conn_string: str):
    """Windows 路径：同步 PostgresSaver + 线程池桥接。"""
    from langgraph.checkpoint.postgres import PostgresSaver

    def _open():
        cm = PostgresSaver.from_conn_string(conn_string)
        saver = cm.__enter__()
        saver.setup()
        saver._cm = cm  # dispose_checkpointer 释放连接时使用
        return saver

    sync_saver = await asyncio.to_thread(_open)
    return _SyncSaverAsyncBridge.wrap(sync_saver)


async def dispose_checkpointer(checkpointer) -> None:
    """释放 checkpointer 持有的连接。"""
    if getattr(checkpointer, "__aexit__", None) is not None:
        try:
            await checkpointer.__aexit__(None, None, None)
        except Exception:
            logger.debug("checkpointer dispose failed", exc_info=True)
        return
    cm = getattr(checkpointer, "_cm", None)
    if cm is not None:
        try:
            await asyncio.to_thread(cm.__exit__, None, None, None)
        except Exception:
            logger.debug("checkpointer dispose failed", exc_info=True)
