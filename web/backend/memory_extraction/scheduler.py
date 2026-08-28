"""批量记忆提取调度器 — 累计轮次触发提取，支持空闲 flush 和会话切换 flush。

职责：
1. 跟踪每个会话的提取游标（PostgreSQL 是唯一事实源）
2. 累计 interval_turns 个完整轮次后触发一次 LLM 提取
3. 空闲 idle_seconds 后自动 flush 未提取的完整轮次
4. 会话切换时 flush 旧会话
5. 同一 session 同时最多运行一个提取任务
6. 跨进程防重复（Redis 可选锁 + PostgreSQL CAS）
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING

from web.backend.db.base import get_session_factory
from web.backend.repositories import SessionRepository, MessageRepository

from .coordinator import ExtractionResult

if TYPE_CHECKING:
    from novare.llm_client import LLMClient
    from web.backend.memory_extraction.coordinator import MemoryExtractionCoordinator
    from web.backend.redis_service import RedisService

logger = logging.getLogger("novare.memory_extraction.scheduler")


class MemoryExtractionScheduler:
    """批量记忆提取调度器。"""

    def __init__(
        self,
        coordinator: MemoryExtractionCoordinator,
        llm_client: LLMClient,
        redis_service: RedisService | None = None,
        interval_turns: int = 4,
        idle_seconds: int = 120,
        session_factory=None,
        flush_on_switch: bool = True,
        extraction_task_timeout: int = 300,
    ):
        self._coordinator = coordinator
        self._llm_client = llm_client
        self._redis = redis_service
        self._interval_turns = interval_turns
        self._idle_seconds = idle_seconds
        self._flush_on_switch = flush_on_switch
        self._extraction_task_timeout = extraction_task_timeout
        # Normalize: _session_factory 总是 async_sessionmaker 实例
        if session_factory is not None:
            from sqlalchemy.ext.asyncio import async_sessionmaker as _ASM
            if isinstance(session_factory, _ASM):
                self._session_factory = session_factory
            elif callable(session_factory):
                self._session_factory = session_factory()
            else:
                self._session_factory = session_factory
        else:
            self._session_factory = None
        # ── 并发控制状态 ──
        # 已运行的提取任务（同一 session 最多一个）
        self._running_tasks: dict[str, asyncio.Task] = {}
        # 正在启动中的 session（防止 _start_extraction 并发创建双任务）
        self._starting_sessions: set[str] = set()
        # 每个 starting 操作的完成信号，供 forget/shutdown 确定性等待
        self._starting_events: dict[str, asyncio.Event] = {}
        # 已被 forget 的 session（阻止后续 on_turn_completed / flush / _start_extraction）
        self._forgotten_sessions: set[str] = set()
        # per-session idle timer
        self._idle_timers: dict[str, asyncio.Task] = {}
        # 已知可能有 pending messages 的 session 集合（供 shutdown flush）
        self._pending_sessions: set[tuple[str, str]] = set()  # {(user_id, session_id)}
        # shutdown 状态
        self._shutting_down = False
        self._shutdown_event = asyncio.Event()
        self._shutdown_task: asyncio.Task | None = None

    def _get_session_factory(self):
        """获取 async_sessionmaker 工厂，优先使用注入的，否则使用全局的。"""
        if self._session_factory is not None:
            return self._session_factory
        return get_session_factory()

    # ── 公开接口 ──────────────────────────────────────────────────

    async def on_turn_completed(self, user_id: str, session_id: str) -> str:
        """每轮对话消息写入 DB 后调用。

        Returns:
            "scheduled" | "threshold_not_reached" | "already_running" | "shutdown" | "forgotten"
        """
        if self._shutting_down:
            return "shutdown"
        if session_id in self._forgotten_sessions:
            return "forgotten"

        # 读取游标 + 计算完整轮次
        complete_turn_count, last_msg_id = await self._count_complete_turns(
            user_id, session_id
        )

        # DB 查询期间可能发生 shutdown 或 delete。
        if self._shutting_down:
            return "shutdown"
        if session_id in self._forgotten_sessions:
            return "forgotten"

        if complete_turn_count < self._interval_turns:
            logger.debug(
                "Session %s: %d complete turns < threshold %d, skipping",
                session_id, complete_turn_count, self._interval_turns,
            )
            # 记录 pending 状态（供 shutdown flush）
            if complete_turn_count > 0:
                self._pending_sessions.add((user_id, session_id))
            # 重置 idle timer
            self._schedule_idle_flush(user_id, session_id)
            return "threshold_not_reached"

        # 达到阈值，触发提取
        status = await self._start_extraction(user_id, session_id)
        # 重置 idle timer
        self._cancel_idle_flush(session_id)
        self._schedule_idle_flush(user_id, session_id)
        return status

    async def flush_session(
        self, user_id: str, session_id: str, reason: str = "manual"
    ) -> str:
        """Flush 指定会话的待提取消息。

        Returns:
            "scheduled" | "already_running" | "no_pending" | "disabled" | "shutdown" | "forgotten"
        """
        if self._shutting_down and reason != "shutdown":
            return "shutdown"
        if session_id in self._forgotten_sessions:
            return "forgotten"

        # flush_on_switch 配置检查
        if reason == "switch" and not self._flush_on_switch:
            return "disabled"

        complete_turn_count, last_msg_id = await self._count_complete_turns(
            user_id, session_id
        )

        # DB 查询期间可能发生 shutdown 或 delete。
        if self._shutting_down and reason != "shutdown":
            return "shutdown"
        if session_id in self._forgotten_sessions:
            return "forgotten"

        if complete_turn_count == 0:
            return "no_pending"

        status = await self._start_extraction(user_id, session_id, reason=reason)
        self._cancel_idle_flush(session_id)
        # 不在这里 discard pending — 只有成功推进游标后才清理
        return status

    async def forget_session(
        self, user_id: str, session_id: str, timeout: float = 5.0
    ) -> bool:
        """会话删除时清理调度器状态。

        时序：
        1. 标记 forgotten（阻止新的 on_turn_completed / flush / _start_extraction）
        2. 取消 idle timer
        3. 如果有正在启动的任务（_starting_sessions），等待其完成或超时
        4. 取消正在运行的任务并等待真正结束
        5. 只移除已经停止的任务；超时任务保留跟踪以便重试

        Returns:
            True: 所有任务已停止；False: 有任务在 timeout 后仍未停止
        """
        # 1. 标记 forgotten（原子操作，在任何 await 之前）
        self._forgotten_sessions.add(session_id)
        self._cancel_idle_flush(session_id)
        self._pending_sessions = {
            (uid, sid) for uid, sid in self._pending_sessions if sid != session_id
        }

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        # 2. 等待正在启动的操作完成。forgotten 标记保证它不会创建新任务。
        starting_event = self._starting_events.get(session_id)
        if starting_event is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(starting_event.wait()),
                    timeout=max(0.0, deadline - loop.time()),
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Session %s: forget_session timed out waiting for task startup",
                    session_id,
                )

        # 3. 取消正在运行的任务
        task = self._running_tasks.get(session_id)
        if task and not task.done():
            # 重试 forget 时不要重复注入 CancelledError；继续等待同一个任务。
            if task.cancelling() == 0:
                task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=max(0.0, deadline - loop.time()),
                )
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass
            # 检查任务是否真正停止
            if not task.done():
                logger.warning(
                    "Session %s: forget_session timed out waiting for task cancellation",
                    session_id,
                )

        # 4. 只清理已经结束的任务。活任务必须保留，供后续重试继续等待。
        if task is not None and task.done():
            if self._running_tasks.get(session_id) is task:
                self._running_tasks.pop(session_id, None)
        self._pending_sessions = {
            (uid, sid) for uid, sid in self._pending_sessions if sid != session_id
        }
        self._cancel_idle_flush(session_id)

        tracked_task = self._running_tasks.get(session_id)
        return (
            session_id not in self._starting_sessions
            and (tracked_task is None or tracked_task.done())
        )

    def restore_session(self, session_id: str) -> None:
        """数据库删除失败时恢复调度资格；游标仍以 PostgreSQL 为准。"""
        self._forgotten_sessions.discard(session_id)

    async def shutdown(self, timeout: float = 5.0):
        """关闭时等待运行中任务；对仍有待提取消息的 session 做 best-effort flush。

        幂等：并发调用等待同一次 shutdown 完成。
        """
        # 创建 Task 之前没有 await；同一事件循环中只有一个调用者能成为 owner。
        if self._shutdown_task is None:
            self._shutting_down = True
            self._shutdown_task = asyncio.create_task(self._shutdown_impl(timeout))
        await asyncio.shield(self._shutdown_task)

    async def _shutdown_impl(self, timeout: float):
        """实际执行 shutdown 的内部方法。"""
        try:
            self._shutting_down = True

            # 1. 取消所有 idle timer
            for timer in list(self._idle_timers.values()):
                if not timer.done():
                    timer.cancel()
            self._idle_timers.clear()

            # 等待 shutdown 开始前已经进入 Redis/启动阶段的操作。它们会在
            # _start_extraction 的二次 shutdown 检查处退出。
            starting_events = list(self._starting_events.values())
            if starting_events:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(event.wait() for event in starting_events),
                        return_exceptions=True,
                    ),
                    timeout=timeout,
                )

            # 2. 对 pending sessions 调度 best-effort flush（复用 _start_extraction 路径）
            for user_id, session_id in list(self._pending_sessions):
                if session_id not in self._running_tasks:
                    await self._start_extraction(user_id, session_id, reason="shutdown")

            # 3. 等待所有运行中任务
            all_tasks = list(self._running_tasks.values())
            if all_tasks:
                done, pending = await asyncio.wait(all_tasks, timeout=timeout)
                for t in pending:
                    t.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                logger.info(
                    "Scheduler shutdown: %d completed, %d cancelled",
                    len(done), len(pending),
                )

            self._running_tasks.clear()
            self._pending_sessions.clear()
        finally:
            self._shutdown_event.set()

    # ── 内部方法 ──────────────────────────────────────────────────

    async def _count_complete_turns(
        self, user_id: str, session_id: str
    ) -> tuple[int, int | None]:
        """计算游标之后的完整轮次数量和最后一条消息 ID。"""
        from uuid import UUID

        user_uuid = UUID(user_id)
        async with self._get_session_factory()() as db:
            session_repo = SessionRepository(db, user_uuid)
            cursor, _ = await session_repo.get_memory_extraction_state(session_id)

            msg_repo = MessageRepository(db, user_uuid)
            messages = await msg_repo.get_messages_after(session_id, cursor)

        if not messages:
            return 0, None

        last_msg_id = messages[-1].id
        complete_turns = self._group_into_complete_turns(messages)
        return len(complete_turns), last_msg_id

    def _group_into_complete_turns(self, messages: list) -> list[list]:
        """将消息列表分组为完整的 user/assistant 轮次。"""
        turns: list[list] = []
        current_turn: list = []

        for msg in messages:
            role = msg.role
            if role == "user":
                if current_turn:
                    has_assistant = any(m.role == "assistant" for m in current_turn)
                    if has_assistant:
                        turns.append(current_turn)
                current_turn = [msg]
            elif role in ("assistant", "tool", "tool_call"):
                current_turn.append(msg)

        if current_turn:
            has_assistant = any(m.role == "assistant" for m in current_turn)
            if has_assistant:
                turns.append(current_turn)

        return turns

    async def _start_extraction(
        self, user_id: str, session_id: str, reason: str = "threshold"
    ) -> str:
        """启动后台提取任务。

        原子性保证：
        1. 在任何 await 之前检查 _running_tasks、_starting_sessions、_forgotten_sessions
        2. 通过检查后立即加入 _starting_sessions
        3. Redis 获取锁、创建 Task、写入 _running_tasks 完成后再移除 _starting_sessions
        4. 所有异常和提前返回路径在 finally 中清理 _starting_sessions

        Returns:
            "scheduled" | "already_running" | "forgotten" | "shutdown"
        """
        # ── 阶段 0：原子检查（无 await，单线程安全）──
        if self._shutting_down and reason != "shutdown":
            return "shutdown"
        if session_id in self._forgotten_sessions:
            return "forgotten"
        if session_id in self._running_tasks:
            task = self._running_tasks[session_id]
            if not task.done():
                return "already_running"
        if session_id in self._starting_sessions:
            return "already_running"

        # 标记 starting（无 await 点，原子操作）
        self._starting_sessions.add(session_id)
        starting_event = asyncio.Event()
        self._starting_events[session_id] = starting_event

        # 跨进程锁（可选）
        lock_key: str | None = None
        lock_token: str | None = None
        lock_acquired = False
        try:
            if self._redis and self._redis.is_available:
                lock_key = f"lock:mem_extract:{user_id}:{session_id}"
                lock_token = str(uuid.uuid4())
                ttl = max(120, self._extraction_task_timeout * 2 + 60)
                result = await self._redis.set_nx(lock_key, lock_token, ttl)
                if result is False:
                    logger.debug("Session %s extraction locked by another process", session_id)
                    return "already_running"
                lock_acquired = result is True

            # 检查是否在等待 Redis 期间 shutdown 或被 forget
            if self._shutting_down and reason != "shutdown":
                if lock_acquired and self._redis and self._redis.is_available and lock_key and lock_token:
                    try:
                        await self._redis.delete_if_value(lock_key, lock_token)
                    except Exception:
                        logger.warning("Failed to release Redis lock after shutdown during set_nx")
                return "shutdown"
            if session_id in self._forgotten_sessions:
                # 已获得锁，必须安全释放
                if lock_acquired and self._redis and self._redis.is_available and lock_key and lock_token:
                    try:
                        await self._redis.delete_if_value(lock_key, lock_token)
                    except Exception:
                        logger.warning("Failed to release Redis lock after forget during set_nx")
                return "forgotten"

            # 创建受管理的后台任务
            task = asyncio.create_task(
                self._run_extraction(
                    user_id, session_id, reason,
                    lock_key=lock_key, lock_token=lock_token, lock_acquired=lock_acquired,
                )
            )
            task.add_done_callback(
                lambda t, sid=session_id: self._on_task_done(sid, t)
            )
            task.add_done_callback(self._safe_task_callback)
            self._running_tasks[session_id] = task

            logger.info(
                "Session %s extraction scheduled (reason=%s)", session_id, reason
            )
            return "scheduled"

        except Exception:
            # 创建 task 失败时也必须释放已获得的锁
            if lock_acquired and self._redis and self._redis.is_available and lock_key and lock_token:
                try:
                    await self._redis.delete_if_value(lock_key, lock_token)
                except Exception:
                    logger.warning("Failed to release Redis lock after task creation failure")
            raise
        finally:
            # 无论成功还是失败，都移除 starting 标记
            self._starting_sessions.discard(session_id)
            if self._starting_events.get(session_id) is starting_event:
                self._starting_events.pop(session_id, None)
            starting_event.set()

    def _on_task_done(self, session_id: str, done_task: asyncio.Task):
        """任务完成回调：仅当映射中的 Task 就是当前完成 Task 时才删除。"""
        if self._running_tasks.get(session_id) is done_task:
            self._running_tasks.pop(session_id, None)

    async def _run_extraction(
        self, user_id: str, session_id: str, reason: str,
        lock_key: str | None = None,
        lock_token: str | None = None,
        lock_acquired: bool = False,
    ):
        """执行提取：读取消息 → 调用 Coordinator → CAS 推进游标。

        数据库连接在 LLM 调用期间不被持有。
        锁在 finally 中释放。
        forgotten session 不会被重新加入 pending。
        """
        from uuid import UUID

        try:
            user_uuid = UUID(user_id)

            # 检查是否已被 forget
            if session_id in self._forgotten_sessions:
                return

            # ── 阶段 1：读取游标和消息（短生命周期 DB session）──
            async with self._get_session_factory()() as db:
                session_repo = SessionRepository(db, user_uuid)
                cursor, _ = await session_repo.get_memory_extraction_state(
                    session_id
                )
                msg_repo = MessageRepository(db, user_uuid)
                messages = await msg_repo.get_messages_after(session_id, cursor)

            if not messages:
                logger.debug("Session %s: no messages after cursor", session_id)
                return

            # 分组为完整轮次
            complete_turns = self._group_into_complete_turns(messages)
            if not complete_turns:
                logger.debug("Session %s: no complete turns after cursor", session_id)
                return

            # 扁平化所有完整轮次的消息
            batch_messages = []
            for turn in complete_turns:
                batch_messages.extend(turn)

            # 记录批次中最后一条消息的 ID（用于 CAS 推进）
            batch_last_msg_id = batch_messages[-1].id

            # 转换为 dict 格式供 Coordinator 使用
            messages_for_llm = [
                {
                    "role": m.role,
                    "content": m.content or "",
                    **({"tool_calls": m.tool_calls} if m.tool_calls else {}),
                    **({"tool_call_id": m.tool_call_id} if m.tool_call_id else {}),
                }
                for m in batch_messages
            ]

            logger.info(
                "Session %s: extracting %d messages in %d turns (reason=%s)",
                session_id, len(messages_for_llm), len(complete_turns), reason,
            )

            # ── 阶段 2：调用 Coordinator（无 DB session 持有，带超时）──
            try:
                result: ExtractionResult = await asyncio.wait_for(
                    self._coordinator.extract_and_persist(
                        user_id=user_id,
                        session_id=session_id,
                        messages=messages_for_llm,
                        llm_client=self._llm_client,
                    ),
                    timeout=self._extraction_task_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Session %s: extraction timed out after %ds, cursor NOT advanced",
                    session_id, self._extraction_task_timeout,
                )
                # timeout 后保留 pending（除非已 forgotten）
                if session_id not in self._forgotten_sessions:
                    self._pending_sessions.add((user_id, session_id))
                return

            # ── 阶段 3：CAS 推进游标 ──
            if result.should_advance_cursor:
                async with self._get_session_factory()() as db:
                    session_repo = SessionRepository(db, user_uuid)
                    advanced = await session_repo.advance_memory_extraction_cursor(
                        session_id=session_id,
                        expected_cursor=cursor,
                        new_cursor=batch_last_msg_id,
                    )
                    if advanced:
                        await db.commit()
                        logger.info(
                            "Session %s: cursor advanced %s -> %d",
                            session_id, cursor, batch_last_msg_id,
                        )
                        # 推进成功后重新检查是否还有待提取的完整轮次
                        if session_id not in self._forgotten_sessions:
                            _, new_last_msg_id = await self._count_complete_turns(
                                user_id, session_id
                            )
                            if new_last_msg_id is not None:
                                self._pending_sessions.add((user_id, session_id))
                            else:
                                self._pending_sessions.discard((user_id, session_id))
                    else:
                        logger.warning(
                            "Session %s: CAS failed (expected=%s), "
                            "another task may have advanced the cursor",
                            session_id, cursor,
                        )
                        if session_id not in self._forgotten_sessions:
                            self._pending_sessions.add((user_id, session_id))
            else:
                logger.warning(
                    "Session %s: extraction status=%s, cursor NOT advanced",
                    session_id, result.status.value,
                )
                if session_id not in self._forgotten_sessions:
                    self._pending_sessions.add((user_id, session_id))

        except asyncio.CancelledError:
            logger.info("Session %s extraction cancelled", session_id)
            # 取消时保留 pending（除非已 forgotten）
            if session_id not in self._forgotten_sessions:
                self._pending_sessions.add((user_id, session_id))
            raise
        except Exception:
            logger.exception("Session %s extraction failed unexpectedly", session_id)
            # 异常时保留 pending（除非已 forgotten）
            if session_id not in self._forgotten_sessions:
                self._pending_sessions.add((user_id, session_id))
        finally:
            # 释放 Redis 锁
            if lock_acquired and self._redis and self._redis.is_available and lock_key and lock_token:
                try:
                    await self._redis.delete_if_value(lock_key, lock_token)
                except Exception:
                    logger.warning(
                        "Failed to release Redis lock key=%s (non-fatal)", lock_key
                    )

    def _schedule_idle_flush(self, user_id: str, session_id: str):
        """设置或重置空闲 flush 定时器。"""
        self._cancel_idle_flush(session_id)

        async def _idle_flush():
            try:
                await asyncio.sleep(self._idle_seconds)
                # 清理：从 _idle_timers 中移除自身
                current = self._idle_timers.get(session_id)
                if current is asyncio.current_task():
                    self._idle_timers.pop(session_id, None)
                # forgotten session 不触发
                if session_id in self._forgotten_sessions:
                    return
                logger.info(
                    "Session %s idle timeout (%ds), flushing",
                    session_id, self._idle_seconds,
                )
                await self.flush_session(user_id, session_id, reason="idle")
            except asyncio.CancelledError:
                pass

        timer = asyncio.create_task(_idle_flush())
        self._idle_timers[session_id] = timer

    def _cancel_idle_flush(self, session_id: str):
        """取消空闲 flush 定时器。"""
        timer = self._idle_timers.pop(session_id, None)
        if timer and not timer.done():
            timer.cancel()

    @staticmethod
    def _safe_task_callback(task: asyncio.Task):
        """安全的后台任务回调。"""
        try:
            exc = task.exception()
            if exc:
                logger.warning("Scheduler background task failed: %s", exc)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
