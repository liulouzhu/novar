"""novare/graph/checkpointer.py — checkpoint 持久化工厂

阶段 2：LangGraph checkpoint 负责执行现场与恢复；用户可见对话仍以
PostgreSQL 消息表为唯一数据源，Redis 继续负责锁与取消。

thread_id 采用 turn 级隔离：

    thread_id = f"{user_id}:{session_id}:{run_id}"

与 RecoveryState 的 run_id 生命周期一一对应（recovery_states 表里的
快照存有 thread_id，显式恢复时据此回到同一个 thread 继续）。
不采用会话级 thread 累积多轮消息，避免与消息表形成双数据源漂移。

按数据库 URL 选择存储后端：
- postgresql://... → AsyncPostgresSaver（生产）
- None / 其他     → InMemorySaver（测试 / 单进程回退）
"""

from __future__ import annotations

import logging

logger = logging.getLogger("novare.graph.checkpointer")


def build_thread_id(user_id: str | None, session_id: str, run_id: str) -> str:
    """构造 turn 级 checkpoint thread_id。

    run_id 由调用方从 RecoveryState.run_id 取（每次 run_turn 新建，
    显式恢复时复用既有 run_id 回到同一 thread）。
    """
    user_part = user_id or "anonymous"
    return f"{user_part}:{session_id}:{run_id}"


async def build_checkpointer(database_url: str | None = None):
    """按配置构造 checkpointer 并完成初始化（建表等）。

    - database_url 以 postgresql 开头 → AsyncPostgresSaver（需 psycopg）
    - 其余情况 → InMemorySaver：仅用于测试和单进程场景，进程退出即丢
    """
    if database_url and database_url.startswith(("postgresql://", "postgres://")):
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as exc:  # psycopg 未安装时给出可行动的错误
            raise RuntimeError(
                "使用 PostgreSQL checkpoint 需要安装 "
                "langgraph-checkpoint-postgres 和 psycopg："
                "pip install langgraph-checkpoint-postgres 'psycopg[binary]'"
            ) from exc

        conn_string = database_url
        # asyncpg 风格的 +asyncpg 驱动声明对 psycopg 无效，统一剥掉
        if "+asyncpg://" in conn_string:
            conn_string = conn_string.replace("+asyncpg://", "://")
        saver = AsyncPostgresSaver.from_conn_string(conn_string)
        # from_conn_string 返回 async context manager，进入后才能使用
        await saver.__aenter__()
        await saver.setup()
        logger.info("LangGraph checkpoint: AsyncPostgresSaver ready")
        return saver

    from langgraph.checkpoint.memory import InMemorySaver

    logger.info("LangGraph checkpoint: InMemorySaver (未配置数据库，恢复能力仅限本进程)")
    return InMemorySaver()


async def dispose_checkpointer(checkpointer) -> None:
    """释放 checkpointer 持有的连接（Postgres saver 以 aenter 打开）。"""
    close = getattr(checkpointer, "__aexit__", None)
    if close is not None:
        try:
            await close(None, None, None)
        except Exception:
            logger.debug("checkpointer dispose failed", exc_info=True)
