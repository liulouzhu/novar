"""novare/channels/adapter.py — MessageBus ↔ AgentLoop 桥接层

从 MessageBus 消费 InboundMessage，调用 AgentLoop.run_turn()，
将响应通过 OutboundMessage 推回总线。
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import logging
from typing import Any

from novare.channels.bus import MessageBus
from novare.channels.events import InboundMessage, OutboundMessage
from novare.config import get_user_workspace
from web.backend.redis_service import redis_service

logger = logging.getLogger("novare.channels.adapter")


class AgentAdapter:
    """桥接 MessageBus 和 AgentLoop。"""

    def __init__(
        self,
        bus: MessageBus,
        agent_service: Any,
        db_session_factory: Any = None,
        default_user_id: str | None = None,
    ):
        self.bus = bus
        self.agent_service = agent_service
        self.db_session_factory = db_session_factory
        self.default_user_id = default_user_id
        self._running = False

    async def run(self) -> None:
        """主循环：持续消费入站消息并处理。"""
        self._running = True
        logger.info("AgentAdapter started, waiting for inbound messages...")

        while self._running:
            try:
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=5.0,
                )
                asyncio.create_task(self._handle_one(msg))
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("AgentAdapter main loop error")

    async def stop(self) -> None:
        self._running = False

    async def _handle_one(self, msg: InboundMessage) -> None:
        """处理单条入站消息。"""
        try:
            if redis_service.is_available:
                dedupe_key = self._build_dedupe_key(msg)
                if dedupe_key:
                    is_new = await redis_service.set_nx(dedupe_key, "1", ttl=3600)
                    if is_new is False:
                        logger.debug("Duplicate channel message skipped: %s", dedupe_key)
                        return

            user_id = await self._resolve_user(msg.sender_id, msg.channel)
            session = await self.agent_service.load_session(
                session_id=msg.session_key,
                user_id=user_id,
            )

            channel_supports_stream = msg.metadata.get("_wants_stream", False)

            if channel_supports_stream:
                await self._run_with_streaming(msg, session, user_id)
            else:
                await self._run_with_buffering(msg, session, user_id)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to handle message from %s:%s", msg.channel, msg.sender_id)
            try:
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="⚠️ 处理消息时出错，请稍后重试。",
                ))
            except Exception:
                pass

    def _build_tool_context(self, user_id: str | None) -> dict | None:
        if not user_id:
            return None
        ws_path = None
        try:
            ws_path = self.agent_service._workspace_for(user_id)
        except AttributeError:
            ws_path = None
        if ws_path is None:
            ws_path = get_user_workspace(user_id)
        return {"user_id": user_id, "workspace": str(ws_path)}

    @staticmethod
    def _build_dedupe_key(msg: InboundMessage) -> str | None:
        meta = msg.metadata
        for field in ("message_id", "msg_id", "id"):
            mid = meta.get(field)
            if mid:
                return f"dedupe:channel:{msg.channel}:{mid}"

        content_hash = hashlib.md5(msg.content.encode("utf-8")).hexdigest()[:12]
        return f"dedupe:channel:{msg.channel}:{msg.sender_id}:{msg.session_key}:{content_hash}"

    async def _run_with_streaming(
        self,
        msg: InboundMessage,
        session: Any,
        user_id: str | None,
    ) -> None:
        """流式模式：逐 chunk 发送 _stream_delta。"""
        raw_messages: list[dict] = []
        compacted = False

        def on_message(message: dict):
            raw_messages.append(deepcopy(message))

        def on_compact(_session):
            nonlocal compacted
            compacted = True

        def on_text(delta: str):
            self.bus.outbound.put_nowait(OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=delta,
                metadata={"_stream_delta": True},
            ))

        def on_tool(event: str, name: str, args: dict, result: str | None, duration: float | None):
            if event == "start":
                self.bus.outbound.put_nowait(OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"🔧 {name}...",
                    metadata={"_progress": True, "_tool_hint": True},
                ))

        ctx = self._build_tool_context(user_id)
        await self.agent_service.agent.run_turn(
            session,
            msg.content,
            on_text=on_text,
            on_tool=on_tool,
            tool_context=ctx,
            system_prompt=self.agent_service.config.system_prompt,
            autosave=False,
            on_compact=on_compact,
            on_message=on_message,
        )

        self.bus.outbound.put_nowait(OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="",
            metadata={"_stream_delta": True, "_stream_end": True, "_streamed": True},
        ))

        await self._persist_session(session, user_id, raw_messages, compacted)

    async def _run_with_buffering(
        self,
        msg: InboundMessage,
        session: Any,
        user_id: str | None,
    ) -> None:
        """非流式模式（微信等）：缓存全部输出后一次性发送。"""
        final_parts: list[str] = []
        raw_messages: list[dict] = []
        compacted = False

        def on_message(message: dict):
            raw_messages.append(deepcopy(message))

        def on_compact(_session):
            nonlocal compacted
            compacted = True

        def on_text(delta: str):
            final_parts.append(delta)

        def on_tool(event: str, name: str, args: dict, result: str | None, duration: float | None):
            pass

        ctx = self._build_tool_context(user_id)
        await self.agent_service.agent.run_turn(
            session,
            msg.content,
            on_text=on_text,
            on_tool=on_tool,
            tool_context=ctx,
            system_prompt=self.agent_service.config.system_prompt,
            autosave=False,
            on_compact=on_compact,
            on_message=on_message,
        )

        final_text = "".join(final_parts).strip()
        if final_text:
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=final_text,
            ))

        await self._persist_session(session, user_id, raw_messages, compacted)

    async def _resolve_user(self, sender_id: str, channel: str) -> str | None:
        """将渠道 sender_id 映射到 Novare user_id。"""
        if self.default_user_id:
            return self.default_user_id

        if not self.db_session_factory:
            return None

        try:
            from uuid import UUID
            from web.backend.db.models import ChannelUser, User

            async with self.db_session_factory() as db:
                try:
                    from sqlalchemy import select
                    result = await db.execute(
                        select(ChannelUser).where(
                            ChannelUser.channel == channel,
                            ChannelUser.platform_user_id == sender_id,
                        )
                    )
                    mapping = result.scalar_one_or_none()

                    if mapping:
                        return str(mapping.novare_user_id)

                    user_id = await self._auto_register_user(db, sender_id, channel)
                    await db.commit()
                    return user_id
                except Exception:
                    await db.rollback()
                    logger.exception("Failed to resolve channel user")
                    return None
        except Exception:
            logger.exception("DB connection failed in _resolve_user")
            return None

    @staticmethod
    async def _auto_register_user(db: Any, sender_id: str, channel: str) -> str:
        """为渠道用户自动创建 Novare 用户和映射记录。"""
        import hashlib
        import secrets
        from web.backend.db.models import ChannelUser, User

        sender_hash = hashlib.md5(f"{channel}:{sender_id}".encode()).hexdigest()[:10]
        username = f"{channel}_{sender_hash}"
        email = f"{channel}_{sender_hash}@channel.novare.local"
        password_hash = secrets.token_hex(32)

        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
        )
        db.add(user)
        await db.flush()

        mapping = ChannelUser(
            novare_user_id=user.id,
            channel=channel,
            platform_user_id=sender_id,
        )
        db.add(mapping)
        await db.flush()

        logger.info("Auto-registered channel user: %s:%s -> %s (%s)", channel, sender_id, user.id, username)
        return str(user.id)

    async def _persist_session(
        self,
        session: Any,
        user_id: str | None,
        raw_messages: list[dict],
        compacted: bool,
    ) -> None:
        """Append raw messages and atomically update a snapshot after compaction."""
        if not user_id or not self.db_session_factory:
            return
        try:
            async with self.db_session_factory() as db:
                try:
                    from novare.context_manager import estimate_messages_tokens
                    from web.backend.repositories import (
                        ContextSnapshotRepository,
                        MessageRepository,
                        SessionRepository,
                    )
                    from uuid import UUID

                    user_uuid = UUID(user_id)
                    session_repo = SessionRepository(db, user_uuid)
                    session_model = await session_repo.get_by_id(session.session_id)
                    if not session_model:
                        await session_repo.create(session.session_id, title="微信会话")

                    msg_repo = MessageRepository(db, user_uuid)
                    last_raw_message_id = None
                    for message in raw_messages:
                        saved = await msg_repo.add_message(
                            session_id=session.session_id,
                            role=message["role"],
                            content=message.get("content"),
                            tool_calls=message.get("tool_calls"),
                            tool_call_id=message.get("tool_call_id"),
                            name=message.get("name"),
                        )
                        last_raw_message_id = saved.id

                    if compacted:
                        if last_raw_message_id is None:
                            last_raw_message_id = await msg_repo.get_latest_message_id(
                                session.session_id
                            )
                        if last_raw_message_id is None:
                            raise RuntimeError(
                                "Cannot persist context snapshot without raw messages"
                            )
                        snapshot_repo = ContextSnapshotRepository(db, user_uuid)
                        snapshot = await snapshot_repo.upsert(
                            session_id=session.session_id,
                            snapshot_data=deepcopy(session.messages),
                            compacted_through_message_id=last_raw_message_id,
                            estimated_tokens=estimate_messages_tokens(session.messages),
                            schema_version=_context_snapshot_schema_version(session.messages),
                        )
                        if snapshot is None:
                            raise PermissionError(
                                f"Session {session.session_id} is not owned by user {user_id}"
                            )
                    await db.commit()
                except Exception:
                    await db.rollback()
                    logger.exception("Failed to persist channel session to DB")
        except Exception:
            logger.exception("DB session creation failed")


def _context_snapshot_schema_version(messages: list[dict]) -> int:
    versions = [
        message.get("_compaction_meta", {}).get("schema_version", 1)
        for message in messages
        if message.get("_compaction_meta")
    ]
    return max((int(version) for version in versions), default=1)
