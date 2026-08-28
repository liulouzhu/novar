"""novare/subagents/registry.py — 子智能体生命周期管理

借鉴 claw-code 的 TaskRegistry/WorkerRegistry，但简化为 asyncio 版本。
管理子智能体的状态机：PENDING → RUNNING → COMPLETED / FAILED / CANCELLED。
"""

from __future__ import annotations

import asyncio
import logging
import time
from uuid import uuid4

from novare.subagents.types import (
    SubagentRecord,
    SubagentOutput,
    SubagentStatus,
    SubagentType,
)

logger = logging.getLogger("novare.subagents.registry")


def _make_subagent_id() -> str:
    return f"sa-{uuid4().hex[:12]}"


class SubagentRegistry:
    """中央子智能体注册表

    每个进程一个实例（CLI 模式或 Web 模式的 AgentService 中）。
    存储 SubagentRecord 的内存字典，管理生命周期状态转换。
    """

    def __init__(self, max_records: int = 1000, cleanup_age_seconds: float = 3600):
        self._records: dict[str, SubagentRecord] = {}
        self.max_records = max_records
        self.cleanup_age_seconds = cleanup_age_seconds

    # ── 创建 ────────────────────────────────────────────────────

    def create(self, subagent_type: SubagentType, task: str) -> SubagentRecord:
        """创建新子智能体记录（PENDING 状态）"""
        # 自动清理过期记录并强制容量上限
        self.cleanup_finished(self.cleanup_age_seconds)
        self._enforce_max_records()

        sid = _make_subagent_id()
        record = SubagentRecord(
            subagent_id=sid,
            type=subagent_type,
            task=task,
            status=SubagentStatus.PENDING,
        )
        self._records[sid] = record
        logger.info("Subagent created: %s (type=%s, task=%s)", sid, subagent_type.value, task[:60])
        return record

    # ── 启动 ────────────────────────────────────────────────────

    async def start(self, subagent_id: str, coro) -> None:
        """将子智能体状态转为 RUNNING，并启动 asyncio.Task

        Args:
            subagent_id: 子智能体 ID
            coro: 要执行的协程（通常是 run_subagent(...)）
        """
        record = self._records.get(subagent_id)
        if not record:
            raise KeyError(f"Subagent not found: {subagent_id}")

        record.status = SubagentStatus.RUNNING
        record.asyncio_task = asyncio.create_task(coro)
        logger.info("Subagent started: %s", subagent_id)

    # ── 完成 / 失败 ────────────────────────────────────────────

    def complete(self, subagent_id: str, result: str) -> None:
        """标记子智能体完成"""
        record = self._records.get(subagent_id)
        if not record:
            return

        record.status = SubagentStatus.COMPLETED
        record.result = result
        record.finished_at = time.monotonic()
        # 统计工具调用次数（从 session 消息中计算）
        logger.info(
            "Subagent completed: %s (%.1fs, %d tool calls)",
            subagent_id, record.elapsed, record.tool_calls_made,
        )

    def fail(self, subagent_id: str, error: str) -> None:
        """标记子智能体失败"""
        record = self._records.get(subagent_id)
        if not record:
            return

        record.status = SubagentStatus.FAILED
        record.error = error
        record.finished_at = time.monotonic()
        logger.warning("Subagent failed: %s — %s", subagent_id, error[:200])

    # ── 取消 ────────────────────────────────────────────────────

    async def cancel(self, subagent_id: str) -> bool:
        """取消运行中的子智能体"""
        record = self._records.get(subagent_id)
        if not record:
            return False

        if record.asyncio_task and not record.asyncio_task.done():
            record.asyncio_task.cancel()
            try:
                await record.asyncio_task
            except asyncio.CancelledError:
                pass

        record.status = SubagentStatus.CANCELLED
        record.finished_at = time.monotonic()
        logger.info("Subagent cancelled: %s", subagent_id)
        return True

    # ── 查询 ────────────────────────────────────────────────────

    def get(self, subagent_id: str) -> SubagentRecord | None:
        """查找子智能体记录"""
        return self._records.get(subagent_id)

    def get_output(self, subagent_id: str) -> SubagentOutput | None:
        """获取子智能体输出"""
        record = self._records.get(subagent_id)
        if not record:
            return None
        return record.to_output()

    def list_active(self) -> list[SubagentRecord]:
        """列出所有活跃（PENDING/RUNNING）的子智能体"""
        return [
            r for r in self._records.values()
            if r.status in (SubagentStatus.PENDING, SubagentStatus.RUNNING)
        ]

    def list_all(self) -> list[SubagentRecord]:
        """列出所有子智能体"""
        return list(self._records.values())

    # ── 清理 ────────────────────────────────────────────────────

    def cleanup_finished(self, max_age_seconds: float = 3600) -> int:
        """清理已完成/失败/取消的过期记录

        Args:
            max_age_seconds: 最大保留时间（秒）

        Returns:
            清理的记录数
        """
        now = time.monotonic()
        to_remove = [
            sid for sid, r in self._records.items()
            if r.status in (SubagentStatus.COMPLETED, SubagentStatus.FAILED, SubagentStatus.CANCELLED)
            and r.finished_at is not None
            and (now - r.finished_at) > max_age_seconds
        ]
        for sid in to_remove:
            del self._records[sid]
        if to_remove:
            logger.info("Cleaned up %d finished subagent records", len(to_remove))
        return len(to_remove)

    def _enforce_max_records(self) -> None:
        """当记录数超过 max_records 时，优先删除最老的已完成记录。"""
        if len(self._records) < self.max_records:
            return

        # 按完成时间（或创建时间）排序的 finished 记录
        _FINISHED = (SubagentStatus.COMPLETED, SubagentStatus.FAILED, SubagentStatus.CANCELLED)
        finished = [
            (sid, r) for sid, r in self._records.items()
            if r.status in _FINISHED
        ]
        if not finished:
            logger.warning(
                "SubagentRegistry at max_records=%d but no finished records to evict "
                "(all %d are active)", self.max_records, len(self._records),
            )
            return

        # 按 finished_at 或 created_at 升序，最老的排前面
        finished.sort(key=lambda item: item[1].finished_at or item[1].created_at)
        # 删除到 max_records 以下
        to_evict = len(self._records) - self.max_records + 1
        for sid, _ in finished[:to_evict]:
            del self._records[sid]
        logger.info("Evicted %d oldest finished subagent records (max_records=%d)", to_evict, self.max_records)

    async def cancel_all(self) -> int:
        """取消所有活跃的子智能体（用于进程退出时清理）"""
        active = self.list_active()
        for record in active:
            await self.cancel(record.subagent_id)
        return len(active)
