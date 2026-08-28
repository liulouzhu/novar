"""情景记忆 CRUD 端点 — 用户级 API。"""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from web.backend.db.base import get_db
from web.backend.db.models import User
from web.backend.auth.dependencies import get_current_user
from web.backend.repositories.episodic_memory_repo import EpisodicMemoryRepository
from web.backend.episodic_memory.vector_store import EpisodicMemoryVectorStore
from web.backend.episodic_memory.schemas import EpisodicMemoryOut

logger = logging.getLogger("novare.web.episodic_memories")

router = APIRouter(prefix="/api/memories/episodes", tags=["episodic_memories"])

# 模块级 VectorStore 实例（惰性初始化）
_vector_store = EpisodicMemoryVectorStore()


def _to_out(m) -> EpisodicMemoryOut:
    return EpisodicMemoryOut(
        id=str(m.id),
        memory_type=m.memory_type,
        summary=m.summary or "",
        context=m.context or "",
        action=m.action or "",
        outcome=m.outcome or "",
        topics=m.topics or [],
        importance=m.importance or 0.5,
        confidence=m.confidence or 0.5,
        status=m.status or "active",
        pinned=m.pinned or False,
        session_id=m.session_id,
        occurred_at=m.occurred_at.isoformat() if m.occurred_at else None,
        created_at=m.created_at.isoformat() if m.created_at else None,
        updated_at=m.updated_at.isoformat() if m.updated_at else None,
        last_retrieved_at=m.last_retrieved_at.isoformat() if m.last_retrieved_at else None,
        retrieval_count=m.retrieval_count or 0,
        index_status=m.index_status or "pending",
    )


@router.get("", response_model=list[EpisodicMemoryOut])
async def list_episodic_memories(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """列出当前用户的所有 active 情景记忆。"""
    repo = EpisodicMemoryRepository(db, user.id)
    memories = await repo.list_active()
    return [_to_out(m) for m in memories]


@router.get("/{memory_id}", response_model=EpisodicMemoryOut)
async def get_episodic_memory(
    memory_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取单条情景记忆。"""
    repo = EpisodicMemoryRepository(db, user.id)
    try:
        mid = UUID(memory_id)
    except ValueError:
        raise HTTPException(404, "Memory not found")
    memory = await repo.get_by_id(mid)
    if not memory:
        raise HTTPException(404, "Memory not found")
    return _to_out(memory)


@router.delete("/{memory_id}")
async def delete_episodic_memory(
    memory_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """删除情景记忆（软删除 + Milvus 向量清理）。"""
    repo = EpisodicMemoryRepository(db, user.id)
    try:
        mid = UUID(memory_id)
    except ValueError:
        raise HTTPException(404, "Memory not found")

    memory = await repo.get_by_id(mid)
    if not memory:
        raise HTTPException(404, "Memory not found")

    # PostgreSQL 软删除
    deleted = await repo.delete(mid)
    if not deleted:
        raise HTTPException(404, "Memory not found")
    await db.commit()

    # 尝试删除 Milvus 向量
    vector_deleted = False
    warnings: list[str] = []
    try:
        vector_deleted = await _vector_store.delete_memory(str(mid))
    except Exception:
        logger.warning("Milvus vector delete failed for %s (non-fatal)", memory_id)

    result: dict = {"ok": True, "vector_deleted": vector_deleted}
    if not vector_deleted:
        result["warnings"] = ["数据库记忆已删除，但向量索引清理失败。"]
    return result


@router.post("/{memory_id}/pin")
async def toggle_pin_episodic_memory(
    memory_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """切换情景记忆的 pinned 状态。"""
    repo = EpisodicMemoryRepository(db, user.id)
    try:
        mid = UUID(memory_id)
    except ValueError:
        raise HTTPException(404, "Memory not found")

    memory = await repo.get_by_id(mid)
    if not memory:
        raise HTTPException(404, "Memory not found")

    memory.pinned = not memory.pinned
    await db.commit()
    await db.refresh(memory)
    return _to_out(memory)


@router.post("/{memory_id}/archive")
async def archive_episodic_memory(
    memory_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """归档情景记忆。"""
    repo = EpisodicMemoryRepository(db, user.id)
    try:
        mid = UUID(memory_id)
    except ValueError:
        raise HTTPException(404, "Memory not found")

    archived = await repo.archive(mid)
    if not archived:
        raise HTTPException(404, "Memory not found")
    await db.commit()

    # 尝试删除 Milvus 向量
    vector_deleted = False
    warnings: list[str] = []
    try:
        vector_deleted = await _vector_store.delete_memory(str(mid))
    except Exception:
        logger.warning("Milvus vector delete failed for %s (non-fatal)", memory_id)

    result: dict = {"ok": True, "vector_deleted": vector_deleted}
    if not vector_deleted:
        result["warnings"] = ["数据库记忆已归档，但向量索引清理失败。"]
    return result
