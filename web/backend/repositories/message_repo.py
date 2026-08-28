from uuid import UUID
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from web.backend.db.models import MessageModel, SessionModel
from .base import BaseRepository


class MessageRepository(BaseRepository):
    def __init__(self, db: AsyncSession, user_id: UUID):
        super().__init__(db, user_id)

    async def add_message(self, session_id: str, role: str, content: str | None = None,
                    tool_calls: list | None = None, tool_call_id: str | None = None,
                    name: str | None = None) -> MessageModel:
        msg = MessageModel(
            session_id=session_id, role=role, content=content,
            tool_calls=tool_calls, tool_call_id=tool_call_id, name=name,
        )
        self.db.add(msg)
        await self.db.flush()
        return msg

    async def get_messages_after(
        self, session_id: str, last_message_id: int | None
    ) -> list[MessageModel]:
        """返回 last_message_id 之后的所有消息（按 id 排序）。

        last_message_id 为 None 时返回会话所有消息。
        仅返回属于当前用户的 session 的消息。
        """
        # 验证 session 归属
        result = await self.db.execute(
            select(SessionModel.id).where(
                SessionModel.id == session_id,
                SessionModel.user_id == self.user_id,
            )
        )
        if result.scalar_one_or_none() is None:
            return []

        stmt = (
            select(MessageModel)
            .where(MessageModel.session_id == session_id)
            .order_by(MessageModel.id)
        )
        if last_message_id is not None:
            stmt = stmt.where(MessageModel.id > last_message_id)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_messages(self, session_id: str) -> list[MessageModel]:
        # 先验证 session 归属
        result = await self.db.execute(
            select(SessionModel.id).where(
                SessionModel.id == session_id,
                SessionModel.user_id == self.user_id,
            )
        )
        if result.scalar_one_or_none() is None:
            return []

        result = await self.db.execute(
            select(MessageModel)
            .where(MessageModel.session_id == session_id)
            .order_by(MessageModel.id)
        )
        return list(result.scalars().all())

    async def get_latest_message_id(self, session_id: str) -> int | None:
        """Return the latest immutable raw-message id for an owned session."""
        if not await self._verify_session_ownership(session_id):
            return None
        result = await self.db.execute(
            select(func.max(MessageModel.id)).where(
                MessageModel.session_id == session_id
            )
        )
        return result.scalar_one_or_none()

    async def _verify_session_ownership(self, session_id: str) -> bool:
        """校验 session 是否属于当前用户，供所有写操作内部使用。"""
        result = await self.db.execute(
            select(SessionModel.id).where(
                SessionModel.id == session_id,
                SessionModel.user_id == self.user_id,
            )
        )
        return result.scalar_one_or_none() is not None

    async def delete_by_session(self, session_id: str) -> bool:
        """删除 session 下所有消息。返回 False 表示 session 不属于当前用户（拒绝操作）。"""
        if not await self._verify_session_ownership(session_id):
            return False
        await self.db.execute(
            delete(MessageModel).where(MessageModel.session_id == session_id)
        )
        await self.db.flush()
        return True
