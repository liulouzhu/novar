"""WebSocket 聊天端点 + 任务取消/状态 HTTP 接口"""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web.backend.auth.dependencies import get_current_user
from web.backend.auth.service import decode_access_token
from web.backend.db.base import get_session_factory
from web.backend.db.models import User
from web.backend.redis_service import redis_service
from web.backend.repositories import SessionRepository

logger = logging.getLogger("novare.web.chat")
router = APIRouter()


@router.websocket("/ws/chat/{session_id}")
async def ws_chat(websocket: WebSocket, session_id: str, token: str = Query(...)):
    """WebSocket 聊天端点

    认证：通过 query param ?token=xxx 传递 JWT。
    WebSocket 无法使用 Authorization header，因此 token 通过 URL 参数传递。

    协议：
      客户端 → 服务端:
        {"type": "send", "content": "用户消息"}
        {"type": "send_with_refs", "content": "...", "references": [...]}
        {"type": "stop"}

      服务端 → 客户端:
        {"type": "text_delta", "content": "..."}
        {"type": "tool_start", "tool": "...", "params": {...}}
        {"type": "tool_end", "tool": "...", "result": "...", "duration": 2.3}
        {"type": "tool_error", "tool": "...", "error": "..."}
        {"type": "task_state", "goal": "...", "completed": [...], "pending": [...], ...}
        {"type": "verification", "status": "revised", "risk_level": "high", ...}
        {"type": "done"}
        {"type": "error", "message": "..."}
    """
    # ── 认证 ──
    user_id_str = decode_access_token(token)
    if not user_id_str:
        await websocket.accept()
        await websocket.close(code=4001, reason="Invalid or expired token")
        return

    # 验证会话属于当前用户
    async with get_session_factory()() as db:
        try:
            repo = SessionRepository(db, UUID(user_id_str))
            session_model = await repo.get_by_id(session_id)
            if not session_model:
                await websocket.accept()
                await websocket.close(code=4004, reason="Session not found")
                return
        finally:
            await db.close()

    # ── 接受连接 ──
    await websocket.accept()
    logger.info("WebSocket connected: session=%s user=%s", session_id, user_id_str)

    from web.backend.app import agent_service
    session = await agent_service.load_session(session_id, user_id=user_id_str)
    queue: asyncio.Queue = asyncio.Queue()
    current_task: asyncio.Task | None = None
    stopped = False

    # 任务状态 TTL（与 AgentService 保持一致）
    task_ttl = max(3600, (agent_service.config.turn_timeout if agent_service.config else 300) + 300)

    async def recv_messages():
        """监听 WebSocket 收到的消息，转发到 queue 供主循环处理"""
        nonlocal stopped, current_task
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = data.get("type", "send")

                if msg_type == "stop":
                    if redis_service.is_available:
                        cancel_key = f"cancel:user:{user_id_str}:session:{session_id}"
                        await redis_service.set(cancel_key, "1", ttl=task_ttl)
                    else:
                        stopped = True
                        if current_task and not current_task.done():
                            current_task.cancel()
                            logger.info("Agent task force-cancelled (no Redis): session=%s", session_id)
                    continue

                await queue.put(data)
        except WebSocketDisconnect:
            pass

    try:
        recv_task = asyncio.create_task(recv_messages())

        while True:
            data = await queue.get()

            msg_type = data.get("type", "send")
            content = data.get("content", "")

            if not content.strip():
                continue

            stopped = False

            user_input = content
            if msg_type == "send_with_refs" and data.get("references"):
                refs_text = "\n\n参考文献：\n"
                for ref in data["references"]:
                    refs_text += f"- {ref.get('title', ref.get('id', ''))}\n"
                user_input = content + refs_text

            event_queue: asyncio.Queue = asyncio.Queue()

            # PR 3：显式恢复入口（可选字段 recovery_run_id：非空、限长字符串）
            recovery_run_id = data.get("recovery_run_id")
            if recovery_run_id is not None:
                if (
                    not isinstance(recovery_run_id, str)
                    or not recovery_run_id.strip()
                    or len(recovery_run_id) > 128
                    or not user_id_str
                ):
                    # 无效恢复请求 → fail closed，不执行新 turn
                    await websocket.send_json({
                        "type": "error",
                        "code": "RECOVERY_RESUME_FAILED",
                        "message": "无法恢复指定任务，请重新开始或选择有效的运行记录。",
                    })
                    continue
                recovery_run_id = recovery_run_id.strip()

            current_task = asyncio.create_task(
                agent_service.run_turn(
                    session, user_input, event_queue,
                    user_id=user_id_str,
                    recovery_run_id=recovery_run_id,
                )
            )

            try:
                while True:
                    event = await asyncio.wait_for(event_queue.get(), timeout=300)
                    if stopped:
                        break
                    await websocket.send_json(event)
                    if event.get("type") in ("done", "error"):
                        break
            except asyncio.TimeoutError:
                if not stopped:
                    await websocket.send_json({"type": "error", "message": "响应超时"})
                current_task.cancel()
            except asyncio.CancelledError:
                logger.info("Agent task cancelled: session=%s", session_id)

            if stopped:
                await websocket.send_json({"type": "done"})
                stopped = False

            current_task = None

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected: session=%s", session_id)
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if current_task and not current_task.done():
            current_task.cancel()
        if not recv_task.done():
            recv_task.cancel()


# ── HTTP 接口：取消任务 / 查询任务状态 ──────────────────────────────────────


@router.post("/api/chat/{session_id}/cancel")
async def cancel_task(session_id: str, user: User = Depends(get_current_user)):
    """协作式取消：写入 Redis cancel key，AgentLoop 在下个检查点优雅停止。"""
    if not redis_service.is_available:
        return {"ok": False, "reason": "redis_unavailable"}
    from web.backend.app import agent_service
    cancel_key = f"cancel:user:{user.id}:session:{session_id}"
    task_ttl = max(3600, (agent_service.config.turn_timeout if agent_service.config else 300) + 300)
    await redis_service.set(cancel_key, "1", ttl=task_ttl)
    return {"ok": True}


@router.get("/api/chat/{session_id}/task")
async def get_task_status(session_id: str, user: User = Depends(get_current_user)):
    """查询当前任务状态。无任务或 Redis 不可用时返回 idle。"""
    if not redis_service.is_available:
        return {"status": "idle"}
    task_key = f"task:user:{user.id}:session:{session_id}"
    state = await redis_service.get_json(task_key)
    if state is None:
        return {"status": "idle"}
    return state
