from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web.backend.db.models import ContextSnapshot, SessionModel, utcnow

from .base import BaseRepository


class ContextSnapshotRepository(BaseRepository):
    """User-scoped access to the active compacted context for a session."""

    def __init__(self, db: AsyncSession, user_id: UUID):
        super().__init__(db, user_id)

    async def get_by_session(self, session_id: str) -> ContextSnapshot | None:
        result = await self.db.execute(
            select(ContextSnapshot)
            .join(SessionModel, SessionModel.id == ContextSnapshot.session_id)
            .where(
                ContextSnapshot.session_id == session_id,
                ContextSnapshot.user_id == self.user_id,
                SessionModel.user_id == self.user_id,
            )
        )
        return result.scalar_one_or_none()

    async def upsert(
        self,
        session_id: str,
        snapshot_data: list[dict],
        compacted_through_message_id: int,
        estimated_tokens: int | None = None,
        schema_version: int = 1,
    ) -> ContextSnapshot | None:
        owner = await self.db.execute(
            select(SessionModel.id).where(
                SessionModel.id == session_id,
                SessionModel.user_id == self.user_id,
            )
        )
        if owner.scalar_one_or_none() is None:
            return None

        snapshot = await self.get_by_session(session_id)
        if snapshot is None:
            snapshot = ContextSnapshot(
                session_id=session_id,
                user_id=self.user_id,
                snapshot_data=snapshot_data,
                compacted_through_message_id=compacted_through_message_id,
                estimated_tokens=estimated_tokens,
                schema_version=schema_version,
            )
            self.db.add(snapshot)
        else:
            snapshot.snapshot_data = snapshot_data
            snapshot.compacted_through_message_id = compacted_through_message_id
            snapshot.estimated_tokens = estimated_tokens
            snapshot.schema_version = schema_version
            snapshot.updated_at = utcnow()

        await self.db.flush()
        return snapshot
