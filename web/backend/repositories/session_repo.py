from uuid import UUID
from datetime import datetime, timezone
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession
from web.backend.db.models import SessionModel
from .base import BaseRepository


class SessionRepository(BaseRepository):
    def __init__(self, db: AsyncSession, user_id: UUID):
        super().__init__(db, user_id)

    async def create(self, session_id: str, title: str = "New Chat") -> SessionModel:
        s = SessionModel(id=session_id, user_id=self.user_id, title=title)
        self.db.add(s)
        await self.db.flush()
        return s

    async def get_by_id(self, session_id: str) -> SessionModel | None:
        result = await self.db.execute(
            select(SessionModel).where(
                SessionModel.id == session_id,
                SessionModel.user_id == self.user_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_all(self) -> list[SessionModel]:
        result = await self.db.execute(
            select(SessionModel)
            .where(SessionModel.user_id == self.user_id)
            .order_by(SessionModel.updated_at.desc())
        )
        return list(result.scalars().all())

    async def delete(self, session_id: str) -> bool:
        s = await self.get_by_id(session_id)
        if s:
            await self.db.delete(s)
            await self.db.flush()
            return True
        return False

    async def update_title(self, session_id: str, title: str) -> bool:
        s = await self.get_by_id(session_id)
        if s:
            s.title = title
            await self.db.flush()
            return True
        return False

    # ── 批量记忆提取游标 ──────────────────────────────────────────

    async def get_memory_extraction_state(
        self, session_id: str
    ) -> tuple[int | None, datetime | None]:
        """返回 (last_extracted_message_id, last_memory_extracted_at)。

        Session 不属于当前用户时返回 (None, None)。
        """
        result = await self.db.execute(
            select(
                SessionModel.last_extracted_message_id,
                SessionModel.last_memory_extracted_at,
            ).where(
                SessionModel.id == session_id,
                SessionModel.user_id == self.user_id,
            )
        )
        row = result.one_or_none()
        if row is None:
            return None, None
        return row[0], row[1]

    async def advance_memory_extraction_cursor(
        self,
        session_id: str,
        expected_cursor: int | None,
        new_cursor: int,
    ) -> bool:
        """CAS 推进记忆提取游标。

        expected_cursor 为 None 时使用 IS NULL 条件（首次提取）。
        同时更新 last_memory_extracted_at 和 updated_at。
        返回 True 表示 CAS 成功，False 表示游标已被其他进程更新。
        """
        stmt = (
            update(SessionModel)
            .where(
                SessionModel.id == session_id,
                SessionModel.user_id == self.user_id,
            )
        )
        if expected_cursor is None:
            stmt = stmt.where(SessionModel.last_extracted_message_id.is_(None))
        else:
            stmt = stmt.where(SessionModel.last_extracted_message_id == expected_cursor)

        now = datetime.now(timezone.utc)
        stmt = stmt.values(
            last_extracted_message_id=new_cursor,
            last_memory_extracted_at=now,
            updated_at=now,
        )
        result = await self.db.execute(stmt)
        return result.rowcount > 0
