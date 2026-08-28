"""用户长期记忆 Repository — CRUD 操作"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from web.backend.db.models import UserMemory
from .base import BaseRepository


class MemoryRepository(BaseRepository):
    def __init__(self, db: AsyncSession, user_id: UUID):
        super().__init__(db, user_id)

    async def get_all(self) -> list[UserMemory]:
        """获取该用户的所有记忆条目"""
        result = await self.db.execute(
            select(UserMemory)
            .where(UserMemory.user_id == self.user_id)
            .order_by(UserMemory.category, UserMemory.key)
        )
        return list(result.scalars().all())

    async def get_by_category(self, category: str) -> list[UserMemory]:
        """按类别查询记忆"""
        result = await self.db.execute(
            select(UserMemory)
            .where(UserMemory.user_id == self.user_id, UserMemory.category == category)
            .order_by(UserMemory.key)
        )
        return list(result.scalars().all())

    async def get_by_key(self, category: str, key: str) -> UserMemory | None:
        """查询单条记忆"""
        result = await self.db.execute(
            select(UserMemory)
            .where(
                UserMemory.user_id == self.user_id,
                UserMemory.category == category,
                UserMemory.key == key,
            )
        )
        return result.scalar_one_or_none()

    async def upsert(
        self,
        category: str,
        key: str,
        value: str,
        confidence: float = 1.0,
        tags: list[str] | None = None,
        source: str = "auto",
        pinned: bool = False,
    ) -> UserMemory:
        """插入或更新记忆条目"""
        existing = await self.get_by_key(category, key)
        if existing:
            existing.value = value
            existing.confidence = confidence
            if tags is not None:
                existing.tags = tags
            existing.source = source
            if pinned:
                existing.pinned = True
            await self.db.flush()
            return existing

        memory = UserMemory(
            user_id=self.user_id,
            category=category,
            key=key,
            value=value,
            confidence=confidence,
            pinned=pinned,
            tags=tags or [],
            source=source,
        )
        self.db.add(memory)
        await self.db.flush()
        return memory

    async def delete(self, memory_id: int) -> bool:
        """删除单条记忆"""
        result = await self.db.execute(
            select(UserMemory)
            .where(UserMemory.id == memory_id, UserMemory.user_id == self.user_id)
        )
        memory = result.scalar_one_or_none()
        if memory:
            await self.db.delete(memory)
            await self.db.flush()
            return True
        return False

    async def delete_all(self) -> int:
        """删除该用户的所有记忆，返回删除数量"""
        from sqlalchemy import delete as sa_delete
        result = await self.db.execute(
            sa_delete(UserMemory).where(UserMemory.user_id == self.user_id)
        )
        await self.db.flush()
        return result.rowcount

    async def count(self) -> int:
        """统计该用户的记忆条目数"""
        result = await self.db.execute(
            select(func.count()).select_from(UserMemory).where(UserMemory.user_id == self.user_id)
        )
        return result.scalar_one()

    async def evict_excess(self, max_count: int) -> int:
        """淘汰超出上限的记忆条目，返回删除数量"""
        current = await self.count()
        if current <= max_count:
            return 0

        to_remove = current - max_count

        # 只从非 pinned 条目中淘汰
        result = await self.db.execute(
            select(UserMemory)
            .where(
                UserMemory.user_id == self.user_id,
                UserMemory.pinned == False,  # noqa: E712
            )
            .order_by(UserMemory.confidence.asc(), UserMemory.updated_at.asc())
            .limit(to_remove)
        )
        candidates = list(result.scalars().all())

        for m in candidates:
            await self.db.delete(m)

        await self.db.flush()
        return len(candidates)
