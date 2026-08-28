"""会话 CRUD 端点"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from web.backend.db.base import get_db
from web.backend.db.models import User, MessageModel
from web.backend.auth.dependencies import get_current_user
from web.backend.repositories import SessionRepository, MessageRepository
from web.backend.models import SessionDetail, SessionMeta

logger = logging.getLogger("novare.web.sessions")
router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionMeta])
async def list_sessions(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """列出当前用户的所有会话"""
    repo = SessionRepository(db, user.id)
    sessions = await repo.list_all()

    # 批量查询每个会话的消息数
    session_ids = [s.id for s in sessions]
    if session_ids:
        result = await db.execute(
            select(MessageModel.session_id, func.count(MessageModel.id))
            .where(MessageModel.session_id.in_(session_ids))
            .group_by(MessageModel.session_id)
        )
        counts = dict(result.all())
    else:
        counts = {}

    return [
        SessionMeta(
            session_id=s.id,
            title=s.title or "",
            message_count=counts.get(s.id, 0),
            updated_at=s.updated_at.isoformat() if s.updated_at else "",
        )
        for s in sessions
    ]


@router.post("", response_model=SessionMeta)
async def create_session(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """创建新会话，ID 格式: {timestamp}-{uuid_hex[:8]}"""
    session_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    repo = SessionRepository(db, user.id)
    s = await repo.create(session_id=session_id, title="新会话")
    await db.commit()
    return SessionMeta(
        session_id=s.id,
        title=s.title or "新会话",
        message_count=0,
        updated_at=s.updated_at.isoformat() if s.updated_at else "",
    )


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取会话详情（含消息列表）"""
    session_repo = SessionRepository(db, user.id)
    s = await session_repo.get_by_id(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    msg_repo = MessageRepository(db, user.id)
    messages = await msg_repo.get_messages(session_id)

    result_messages = []
    for msg in messages:
        content = msg.content or ""
        if msg.role == "tool":
            content = content[:500] + ("..." if len(content) > 500 else "")

        item = {
            "role": msg.role,
            "content": content,
        }
        if msg.tool_calls:
            item["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            item["tool_call_id"] = msg.tool_call_id
        result_messages.append(item)

    return SessionDetail(
        session_id=s.id,
        messages=result_messages,
        title=s.title or "",
    )


@router.patch("/{session_id}")
async def update_session(
    session_id: str,
    body: dict,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """更新会话标题"""
    repo = SessionRepository(db, user.id)
    title = body.get("title")
    if title:
        updated = await repo.update_title(session_id, title)
        if not updated:
            raise HTTPException(status_code=404, detail="Session not found")
        await db.commit()
    return {"ok": True}


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """删除会话及关联消息。

    时序：
    1. 验证 session 存在且属于当前用户
    2. 通知 Scheduler forget_session（等待运行中任务停止）
    3. forget 成功后才执行数据库删除并 commit
    4. forget 失败时返回 503，不删除 session
    """
    # 1. 验证 session 存在
    session_repo = SessionRepository(db, user.id)
    session_model = await session_repo.get_by_id(session_id)
    if not session_model:
        raise HTTPException(status_code=404, detail="Session not found")

    # 所有权查询已经完成；等待后台任务期间不占用数据库连接。
    await db.rollback()

    # 2. 通知 Scheduler 停止运行中的任务
    scheduler = None
    try:
        from web.backend.app import agent_service
        scheduler = agent_service.memory_scheduler
        if scheduler:
            stopped = await scheduler.forget_session(
                str(user.id), session_id
            )
            if not stopped:
                logger.warning(
                    "Session %s: forget_session could not stop all tasks within timeout",
                    session_id,
                )
                raise HTTPException(
                    status_code=503,
                    detail="Memory extraction task could not be stopped. Please retry.",
                )
    except HTTPException:
        raise
    except Exception:
        logger.exception("forget_session failed for session %s", session_id)
        raise HTTPException(
            status_code=503,
            detail="Failed to stop memory extraction. Please retry.",
        )

    # 3. forget 成功后才删除数据库记录
    try:
        msg_repo = MessageRepository(db, user.id)
        await msg_repo.delete_by_session(session_id)
        deleted = await session_repo.delete(session_id)
        if not deleted:
            await db.rollback()
            if scheduler:
                scheduler.restore_session(session_id)
            raise HTTPException(status_code=404, detail="Session not found")
        await db.commit()
    except HTTPException:
        raise
    except Exception:
        await db.rollback()
        if scheduler:
            scheduler.restore_session(session_id)
        logger.exception("Failed to delete session %s after scheduler cleanup", session_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to delete session. Please retry.",
        )

    return {"ok": True}


@router.post("/{session_id}/memory/flush")
async def flush_memory_extraction(
    session_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Flush 指定会话的待提取消息（用于会话切换）。

    验证 session 属于当前用户（防 IDOR）。
    不等待 LLM 执行完成，立即返回调度状态。
    """
    # 验证 session 归属（防 IDOR）
    session_repo = SessionRepository(db, user.id)
    session_model = await session_repo.get_by_id(session_id)
    if not session_model:
        raise HTTPException(status_code=404, detail="Session not found")

    from web.backend.app import agent_service
    if not agent_service.memory_scheduler:
        return {"status": "no_pending"}

    status = await agent_service.memory_scheduler.flush_session(
        user_id=str(user.id),
        session_id=session_id,
        reason="switch",
    )
    return {"status": status}
