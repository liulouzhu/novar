"""novare/channels/base.py — 渠道抽象基类

从 researchbot 移植，去掉 Groq 转写依赖，改用 logging 替代 loguru。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from novare.channels.events import InboundMessage, OutboundMessage
from novare.channels.bus import MessageBus

logger = logging.getLogger("novare.channels")


class BaseChannel(ABC):
    """聊天渠道抽象基类。

    每个渠道（Telegram、微信等）实现此接口，通过 MessageBus 与 Agent 通信。
    """

    name: str = "base"
    display_name: str = "Base"

    def __init__(self, config: Any, bus: MessageBus):
        self.config = config
        self.bus = bus
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """启动渠道，开始监听消息（长期运行的 async task）。"""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """停止渠道，清理资源。"""
        ...

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> None:
        """通过此渠道发送消息。失败时应抛出异常，由上层重试策略处理。"""
        ...

    async def send_delta(self, chat_id: str, delta: str, metadata: dict[str, Any] | None = None) -> None:
        """发送流式文本块。子类覆盖以支持流式输出（如 Telegram 的 edit-in-place）。"""
        pass

    @property
    def supports_streaming(self) -> bool:
        """config 启用流式 且 子类覆盖了 send_delta 时返回 True。"""
        cfg = self.config
        streaming = cfg.get("streaming", False) if isinstance(cfg, dict) else getattr(cfg, "streaming", False)
        return bool(streaming) and type(self).send_delta is not BaseChannel.send_delta

    def is_allowed(self, sender_id: str) -> bool:
        """检查 sender_id 是否被允许。空列表→拒绝所有；"*"→允许所有。"""
        allow_list = getattr(self.config, "allow_from", [])
        if not allow_list:
            logger.warning("%s: allow_from is empty — all access denied", self.name)
            return False
        if "*" in allow_list:
            return True
        return str(sender_id) in allow_list

    async def _handle_message(
        self,
        sender_id: str,
        chat_id: str,
        content: str,
        media: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        session_key: str | None = None,
    ) -> None:
        """处理来自平台的入站消息：权限检查 → 构造 InboundMessage → 发布到总线。"""
        if not self.is_allowed(sender_id):
            logger.warning(
                "Access denied for sender %s on channel %s. "
                "Add them to allowFrom list in config to grant access.",
                sender_id, self.name,
            )
            return

        meta = metadata or {}
        if self.supports_streaming:
            meta = {**meta, "_wants_stream": True}

        msg = InboundMessage(
            channel=self.name,
            sender_id=str(sender_id),
            chat_id=str(chat_id),
            content=content,
            media=media or [],
            metadata=meta,
            session_key_override=session_key,
        )

        await self.bus.publish_inbound(msg)

    async def login(self, force: bool = False) -> bool:
        """执行渠道特有的交互式登录（如扫码）。默认返回 True。"""
        return True

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        """返回默认配置。子类可覆盖。"""
        return {"enabled": False}

    @property
    def is_running(self) -> bool:
        return self._running
