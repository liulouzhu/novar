"""记忆 CRUD 端点"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from web.backend.db.base import get_db
from web.backend.db.models import User
from web.backend.auth.dependencies import get_current_user
from web.backend.repositories.memory_repo import MemoryRepository
from web.backend.models import MemoryOut, MemoryUpdate

logger = logging.getLogger("novare.web.memories")
router = APIRouter(prefix="/api/memories", tags=["memories"])


def _to_out(m) -> MemoryOut:
    return MemoryOut(
        id=m.id,
        category=m.category,
        key=m.key,
        value=m.value,
        confidence=m.confidence,
        pinned=m.pinned,
        tags=m.tags or [],
        source=m.source or "auto",
        created_at=m.created_at.isoformat() if m.created_at else None,
        updated_at=m.updated_at.isoformat() if m.updated_at else None,
    )


@router.get("", response_model=list[MemoryOut])
async def list_memories(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    repo = MemoryRepository(db, user.id)
    return [_to_out(m) for m in await repo.get_all()]


@router.patch("/{memory_id}", response_model=MemoryOut)
async def update_memory(
    memory_id: int,
    body: MemoryUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    repo = MemoryRepository(db, user.id)
    memories = await repo.get_all()
    mem = next((m for m in memories if m.id == memory_id), None)
    if not mem:
        raise HTTPException(404, "Memory not found")

    if body.value is not None:
        mem.value = body.value
    if body.tags is not None:
        mem.tags = body.tags
    if body.confidence is not None:
        mem.confidence = max(0.0, min(1.0, body.confidence))

    await db.commit()
    await db.refresh(mem)
    return _to_out(mem)


@router.patch("/{memory_id}/pin", response_model=MemoryOut)
async def toggle_pin(
    memory_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    repo = MemoryRepository(db, user.id)
    memories = await repo.get_all()
    mem = next((m for m in memories if m.id == memory_id), None)
    if not mem:
        raise HTTPException(404, "Memory not found")

    mem.pinned = not mem.pinned
    await db.commit()
    await db.refresh(mem)
    return _to_out(mem)


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    repo = MemoryRepository(db, user.id)
    deleted = await repo.delete(memory_id)
    if not deleted:
        raise HTTPException(404, "Memory not found")
    await db.commit()
    return {"ok": True}


@router.delete("")
async def clear_memories(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    repo = MemoryRepository(db, user.id)
    count = await repo.delete_all()
    await db.commit()
    return {"deleted": count}
