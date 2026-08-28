from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession


class BaseRepository:
    """Base with user_id for scoped repositories."""

    def __init__(self, db: AsyncSession, user_id: UUID):
        self.db = db
        self.user_id = user_id


class SharedRepository:
    """Base for shared (non-user-scoped) repositories."""

    def __init__(self, db: AsyncSession):
        self.db = db
