"""novare/channels/events.py — 消息总线事件类型

从 researchbot 移植，适配 Novare 架构。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class InboundMessage:
    """从聊天渠道收到的消息。"""

    channel: str  # 渠道标识：telegram, weixin, discord, ...
    sender_id: str  # 平台用户 ID
    chat_id: str  # 聊天/频道 ID
    content: str  # 消息文本
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)  # 本地媒体文件路径
    metadata: dict[str, Any] = field(default_factory=dict)  # 渠道特有数据
    session_key_override: str | None = None  # 可选的会话 key 覆盖（如话题线程）

    @property
    def session_key(self) -> str:
        """用于会话标识的唯一 key。"""
        return self.session_key_override or f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """发往聊天渠道的消息。"""

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
