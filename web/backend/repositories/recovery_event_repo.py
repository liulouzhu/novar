"""recovery_event_repo.py — 恢复事件持久化仓库

支持：
- append: 追加事件（原子性，使用 unique 约束兜底）
- get_events_by_run: 按 run_id 查询事件
- get_latest_sequence: 获取最新 sequence
- cleanup_old: 清理过期事件
"""

from uuid import UUID

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from web.backend.db.models import RecoveryEventModel
from .base import BaseRepository


class RecoveryEventRepository(BaseRepository):
    def __init__(self, db: AsyncSession, user_id: UUID):
        super().__init__(db, user_id)

    async def append(
        self,
        session_id: str,
        run_id: str,
        event_type: str,
        payload: dict,
        event_key: str | None = None,
    ) -> RecoveryEventModel | None:
        """追加事件（原子性，使用 unique 约束兜底）。

        sequence 自动递增，并发安全（使用数据库序列或 select max + 1）。
        event_key 唯一约束阻止重复终态事件。
        """
        # 获取当前最大 sequence
        max_seq = await self.get_latest_sequence(session_id, run_id)
        sequence = max_seq + 1

        # 生成 event_key（如果未提供）
        if event_key is None:
            event_key = f"{run_id}:{sequence}:{event_type}"

        model = RecoveryEventModel(
            session_id=session_id,
            user_id=self.user_id,
            run_id=run_id,
            sequence=sequence,
            event_key=event_key,
            event_type=event_type,
            payload=payload,
        )
        self.db.add(model)
        try:
            await self.db.flush()
            return model
        except Exception:
            # unique 约束冲突 → 重复事件，忽略
            await self.db.rollback()
            return None

    async def get_events_by_run(
        self, session_id: str, run_id: str
    ) -> list[RecoveryEventModel]:
        """按 run_id 查询事件（按 sequence 排序）"""
        result = await self.db.execute(
            select(RecoveryEventModel).where(
                RecoveryEventModel.session_id == session_id,
                RecoveryEventModel.run_id == run_id,
                RecoveryEventModel.user_id == self.user_id,
            ).order_by(RecoveryEventModel.sequence)
        )
        return list(result.scalars().all())

    async def get_latest_sequence(
        self, session_id: str, run_id: str
    ) -> int:
        """获取最新 sequence"""
        result = await self.db.execute(
            select(func.max(RecoveryEventModel.sequence)).where(
                RecoveryEventModel.session_id == session_id,
                RecoveryEventModel.run_id == run_id,
                RecoveryEventModel.user_id == self.user_id,
            )
        )
        return result.scalar_one_or_none() or 0

    async def has_event(
        self, session_id: str, run_id: str, event_key: str
    ) -> bool:
        """检查事件是否已存在"""
        result = await self.db.execute(
            select(RecoveryEventModel.id).where(
                RecoveryEventModel.session_id == session_id,
                RecoveryEventModel.run_id == run_id,
                RecoveryEventModel.event_key == event_key,
                RecoveryEventModel.user_id == self.user_id,
            )
        )
        return result.scalar_one_or_none() is not None

    async def cleanup_old(self, max_age_hours: int = 24) -> int:
        """清理过期事件"""
        from datetime import datetime, timedelta, timezone

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        result = await self.db.execute(
            select(RecoveryEventModel.id).where(
                RecoveryEventModel.user_id == self.user_id,
                RecoveryEventModel.created_at < cutoff,
            )
        )
        ids = [row[0] for row in result.all()]
        if not ids:
            return 0

        await self.db.execute(
            delete(RecoveryEventModel).where(
                RecoveryEventModel.id.in_(ids)
            )
        )
        await self.db.flush()
        return len(ids)
