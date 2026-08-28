from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from web.backend.db.models import Paper, UserPaper, utcnow
from .base import BaseRepository


class UserPaperRepository(BaseRepository):
    def __init__(self, db: AsyncSession, user_id: UUID):
        super().__init__(db, user_id)

    async def associate(
        self,
        paper_id: str,
        relation_type: str = "searched",
        has_fulltext_access: bool = False,
        source: str | None = None,
    ) -> UserPaper:
        # Serialize association creation against a last-reference cleanup worker.
        paper_result = await self.db.execute(
            select(Paper).where(Paper.id == paper_id).with_for_update()
        )
        paper = paper_result.scalar_one_or_none()
        if paper is None or paper.deleted_at is not None:
            raise ValueError(f"Paper {paper_id} is unavailable or being cleaned up")
        result = await self.db.execute(
            select(UserPaper).where(
                UserPaper.user_id == self.user_id,
                UserPaper.paper_id == paper_id,
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            existing.deleted_at = None
            if existing.relation_type == "searched" and relation_type != "searched":
                existing.relation_type = relation_type
            if has_fulltext_access and not existing.has_fulltext_access:
                existing.has_fulltext_access = True
            if source:
                existing.source = source
            return existing
        up = UserPaper(
            user_id=self.user_id,
            paper_id=paper_id,
            relation_type=relation_type,
            has_fulltext_access=has_fulltext_access,
            source=source,
        )
        self.db.add(up)
        await self.db.flush()
        return up

    async def get_user_papers(self) -> list[str]:
        """返回用户关联的所有 paper_id（不限 relation_type）。"""
        result = await self.db.execute(
            select(UserPaper.paper_id).where(
                UserPaper.user_id == self.user_id,
                UserPaper.deleted_at.is_(None),
            )
        )
        return [r[0] for r in result.all()]

    async def get_fulltext_paper_ids(self) -> set[str]:
        """返回用户有全文访问权限的 paper_id 集合。"""
        result = await self.db.execute(
            select(UserPaper.paper_id).where(
                UserPaper.user_id == self.user_id,
                UserPaper.has_fulltext_access.is_(True),
                UserPaper.deleted_at.is_(None),
            )
        )
        return {r[0] for r in result.all()}

    async def has_parsed(self, paper_id: str) -> bool:
        """向后兼容：是否有关联记录（任意类型）。"""
        result = await self.db.execute(
            select(UserPaper).where(
                UserPaper.user_id == self.user_id,
                UserPaper.paper_id == paper_id,
                UserPaper.deleted_at.is_(None),
            )
        )
        return result.scalar_one_or_none() is not None

    async def has_fulltext_access(self, paper_id: str) -> bool:
        """是否有全文访问权限（parsed / uploaded / shared）。"""
        result = await self.db.execute(
            select(UserPaper).where(
                UserPaper.user_id == self.user_id,
                UserPaper.paper_id == paper_id,
                UserPaper.deleted_at.is_(None),
            )
        )
        row = result.scalar_one_or_none()
        return row is not None and row.has_fulltext_access

    async def dissociate(self, paper_id: str) -> bool:
        """Logically remove the user's paper association."""
        result = await self.db.execute(
            select(UserPaper).where(
                UserPaper.user_id == self.user_id,
                UserPaper.paper_id == paper_id,
                UserPaper.deleted_at.is_(None),
            )
        )
        row = result.scalar_one_or_none()
        if row is None:
            return False
        row.deleted_at = utcnow()
        await self.db.flush()
        return True

    async def count_associations(self, paper_id: str) -> int:
        """统计有多少用户关联了该论文。"""
        result = await self.db.execute(
            select(func.count()).select_from(UserPaper).where(
                UserPaper.paper_id == paper_id,
                UserPaper.deleted_at.is_(None),
            )
        )
        return result.scalar_one()
