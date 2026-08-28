"""recovery_state_repo.py — RecoveryState 持久化仓库

支持：
- upsert: 创建或更新 RecoveryState（按 session_id + run_id 去重）
- get_by_run_id: 按 run_id 查询
- get_active_by_session: 查询 session 下活跃的 RecoveryState
- mark_status: 状态流转
- cleanup_old: 清理过期的 RecoveryState
"""

from uuid import UUID

from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from web.backend.db.models import RecoveryStateModel
from .base import BaseRepository


class RecoveryStateRepository(BaseRepository):
    def __init__(self, db: AsyncSession, user_id: UUID):
        super().__init__(db, user_id)

    async def upsert(
        self,
        session_id: str,
        run_id: str,
        turn_id: str,
        recovery_data: dict,
        run_status: str = "running",
        iteration: int = 0,
        retry_count: int = 0,
        schema_version: int = 2,
    ) -> RecoveryStateModel:
        """创建或更新 RecoveryState。

        按 session_id + run_id 去重：已存在时更新。
        使用 unique 约束兜底并发。
        """
        existing = await self._find_by_run_id(session_id, run_id)
        if existing:
            existing.turn_id = turn_id
            existing.run_status = run_status
            existing.iteration = iteration
            existing.retry_count = retry_count
            existing.recovery_data = recovery_data
            existing.schema_version = schema_version
            await self.db.flush()
            return existing

        model = RecoveryStateModel(
            session_id=session_id,
            user_id=self.user_id,
            run_id=run_id,
            turn_id=turn_id,
            run_status=run_status,
            iteration=iteration,
            retry_count=retry_count,
            recovery_data=recovery_data,
            schema_version=schema_version,
        )
        self.db.add(model)
        try:
            await self.db.flush()
            return model
        except Exception:
            # unique 约束冲突 → 并发 upsert，回退后重试
            await self.db.rollback()
            existing = await self._find_by_run_id(session_id, run_id)
            if existing:
                existing.turn_id = turn_id
                existing.run_status = run_status
                existing.iteration = iteration
                existing.retry_count = retry_count
                existing.recovery_data = recovery_data
                existing.schema_version = schema_version
                await self.db.flush()
                return existing
            raise

    async def get_by_run_id(
        self, session_id: str, run_id: str
    ) -> RecoveryStateModel | None:
        """按 session_id + run_id 查询"""
        result = await self.db.execute(
            select(RecoveryStateModel).where(
                RecoveryStateModel.session_id == session_id,
                RecoveryStateModel.run_id == run_id,
                RecoveryStateModel.user_id == self.user_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_active_by_session(
        self, session_id: str
    ) -> list[RecoveryStateModel]:
        """查询 session 下活跃的 RecoveryState"""
        result = await self.db.execute(
            select(RecoveryStateModel).where(
                RecoveryStateModel.session_id == session_id,
                RecoveryStateModel.user_id == self.user_id,
                RecoveryStateModel.run_status == "running",
            ).order_by(RecoveryStateModel.id.desc())
        )
        return list(result.scalars().all())

    async def mark_status(
        self, session_id: str, run_id: str, status: str, error: str | None = None
    ) -> bool:
        """更新 run_status"""
        model = await self.get_by_run_id(session_id, run_id)
        if not model:
            return False
        model.run_status = status
        if error:
            from novare.recovery.classifier import sanitize_error
            data = dict(model.recovery_data)
            data["last_error"] = sanitize_error(error)
            model.recovery_data = data
        await self.db.flush()
        return True

    async def mark_completed(self, session_id: str, run_id: str) -> bool:
        """标记为已完成"""
        return await self.mark_status(session_id, run_id, "completed")

    async def mark_failed(
        self, session_id: str, run_id: str, error: str | None = None
    ) -> bool:
        """标记为失败"""
        return await self.mark_status(session_id, run_id, "failed", error)

    async def mark_interrupted(self, session_id: str, run_id: str) -> bool:
        """标记为中断"""
        return await self.mark_status(session_id, run_id, "interrupted")

    async def mark_cancelled(self, session_id: str, run_id: str) -> bool:
        """标记为取消"""
        return await self.mark_status(session_id, run_id, "cancelled")

    async def mark_timed_out(self, session_id: str, run_id: str) -> bool:
        """标记为超时"""
        return await self.mark_status(session_id, run_id, "timed_out")

    async def cleanup_old(self, max_age_hours: int = 24) -> int:
        """清理过期的 RecoveryState（非 running 状态超过 max_age_hours 小时）。

        真正删除记录，而不是软删除。
        """
        from datetime import datetime, timedelta, timezone
        from sqlalchemy import delete

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        result = await self.db.execute(
            select(RecoveryStateModel.id).where(
                RecoveryStateModel.user_id == self.user_id,
                RecoveryStateModel.run_status != "running",
                RecoveryStateModel.updated_at < cutoff,
            )
        )
        ids = [row[0] for row in result.all()]
        if not ids:
            return 0

        await self.db.execute(
            delete(RecoveryStateModel).where(
                RecoveryStateModel.id.in_(ids)
            )
        )
        await self.db.flush()
        return len(ids)

    async def _find_by_run_id(
        self, session_id: str, run_id: str
    ) -> RecoveryStateModel | None:
        """内部方法：按 session_id + run_id 查找"""
        result = await self.db.execute(
            select(RecoveryStateModel).where(
                RecoveryStateModel.session_id == session_id,
                RecoveryStateModel.run_id == run_id,
                RecoveryStateModel.user_id == self.user_id,
            )
        )
        return result.scalar_one_or_none()
