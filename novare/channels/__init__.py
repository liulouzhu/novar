"""novare/channels — 多渠道接入系统

支持通过微信、Telegram 等平台与 Novare Agent 交互。
"""

from novare.channels.base import BaseChannel
from novare.channels.bus import MessageBus
from novare.channels.events import InboundMessage, OutboundMessage
from novare.channels.manager import ChannelManager
from novare.channels.adapter import AgentAdapter

__all__ = [
    "BaseChannel",
    "MessageBus",
    "InboundMessage",
    "OutboundMessage",
    "ChannelManager",
    "AgentAdapter",
]
