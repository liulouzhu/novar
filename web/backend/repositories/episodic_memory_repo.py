"""情景记忆 PostgreSQL Repository — CRUD 操作。"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from web.backend.db.models import EpisodicMemory, utcnow
from .base import BaseRepository


class EpisodicMemoryRepository(BaseRepository):
    """情景记忆 Repository，所有操作按 user_id 隔离。"""

    def __init__(self, db: AsyncSession, user_id: UUID):
        super().__init__(db, user_id)

    async def create(
        self,
        *,
        memory_type: str,
        summary: str,
        context: str = "",
        action: str = "",
        outcome: str = "",
        topics: list[str] | None = None,
        source_message_ids: list[str] | None = None,
        importance: float = 0.5,
        confidence: float = 0.5,
        content_hash: str,
        session_id: str | None = None,
        occurred_at=None,
    ) -> EpisodicMemory:
        """创建情景记忆，index_status=pending。"""
        memory = EpisodicMemory(
            user_id=self.user_id,
            session_id=session_id,
            memory_type=memory_type,
            summary=summary[:500],
            context=(context or "")[:1000],
            action=(action or "")[:1000],
            outcome=(outcome or "")[:1000],
            topics=topics or [],
            source_message_ids=source_message_ids or [],
            importance=max(0.0, min(1.0, importance)),
            confidence=max(0.0, min(1.0, confidence)),
            content_hash=content_hash,
            index_status="pending",
            status="active",
            occurred_at=occurred_at,
        )
        self.db.add(memory)
        await self.db.flush()
        return memory

    async def get_by_id(self, memory_id: UUID) -> EpisodicMemory | None:
        """按 ID 获取，验证 user_id 隔离。"""
        result = await self.db.execute(
            select(EpisodicMemory)
            .where(
                EpisodicMemory.id == memory_id,
                EpisodicMemory.user_id == self.user_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_by_hash(self, content_hash: str) -> EpisodicMemory | None:
        """按 content_hash 精确去重查询。"""
        result = await self.db.execute(
            select(EpisodicMemory)
            .where(
                EpisodicMemory.user_id == self.user_id,
                EpisodicMemory.content_hash == content_hash,
            )
        )
        return result.scalar_one_or_none()

    async def list_active(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[EpisodicMemory]:
        """列出用户所有 active 状态的情景记忆。"""
        result = await self.db.execute(
            select(EpisodicMemory)
            .where(
                EpisodicMemory.user_id == self.user_id,
                EpisodicMemory.status == "active",
            )
            .order_by(EpisodicMemory.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def list_by_session(self, session_id: str) -> list[EpisodicMemory]:
        """列出某个 session 的情景记忆。"""
        result = await self.db.execute(
            select(EpisodicMemory)
            .where(
                EpisodicMemory.user_id == self.user_id,
                EpisodicMemory.session_id == session_id,
            )
            .order_by(EpisodicMemory.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_active_by_ids(self, memory_ids: list[UUID]) -> list[EpisodicMemory]:
        """按 ID 批量获取 active 状态的记忆（用于 Milvus 返回后的二次验证）。"""
        if not memory_ids:
            return []
        result = await self.db.execute(
            select(EpisodicMemory)
            .where(
                EpisodicMemory.id.in_(memory_ids),
                EpisodicMemory.user_id == self.user_id,
                EpisodicMemory.status == "active",
                EpisodicMemory.index_status == "indexed",
            )
        )
        return list(result.scalars().all())

    async def mark_indexed(self, memory_id: UUID, vector_id: str, embedding_model: str) -> None:
        """标记 Milvus 索引成功。"""
        await self.db.execute(
            update(EpisodicMemory)
            .where(EpisodicMemory.id == memory_id, EpisodicMemory.user_id == self.user_id)
            .values(
                index_status="indexed",
                vector_id=vector_id,
                embedding_model=embedding_model,
            )
        )
        await self.db.flush()

    async def mark_index_failed(self, memory_id: UUID) -> None:
        """标记 Milvus 索引失败。"""
        await self.db.execute(
            update(EpisodicMemory)
            .where(EpisodicMemory.id == memory_id, EpisodicMemory.user_id == self.user_id)
            .values(index_status="failed")
        )
        await self.db.flush()

    async def increment_retrieval_count(self, memory_id: UUID) -> None:
        """增加检索计数并更新最后检索时间。"""
        await self.db.execute(
            update(EpisodicMemory)
            .where(EpisodicMemory.id == memory_id, EpisodicMemory.user_id == self.user_id)
            .values(
                retrieval_count=EpisodicMemory.retrieval_count + 1,
                last_retrieved_at=utcnow(),
            )
        )
        await self.db.flush()

    async def archive(self, memory_id: UUID) -> bool:
        """归档情景记忆（软删除到 archived 状态）。"""
        memory = await self.get_by_id(memory_id)
        if not memory:
            return False
        memory.status = "archived"
        await self.db.flush()
        return True

    async def delete(self, memory_id: UUID) -> bool:
        """删除情景记忆（软删除到 deleted 状态）。

        其他用户无法得知记录是否存在。
        """
        memory = await self.get_by_id(memory_id)
        if not memory:
            return False
        memory.status = "deleted"
        await self.db.flush()
        return True

    async def count_active(self) -> int:
        """统计 active 状态的记忆条数。"""
        from sqlalchemy import func
        result = await self.db.execute(
            select(func.count())
            .select_from(EpisodicMemory)
            .where(
                EpisodicMemory.user_id == self.user_id,
                EpisodicMemory.status == "active",
            )
        )
        return result.scalar_one()
