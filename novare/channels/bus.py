"""novare/channels/bus.py — 异步消息总线

渠道和 Agent 核心通过 asyncio.Queue 解耦通信。
"""

import asyncio

from novare.channels.events import InboundMessage, OutboundMessage


class MessageBus:
    """异步消息总线：渠道推入入站队列，Agent 处理后推入出站队列。"""

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """发布一条来自渠道的入站消息。"""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """消费下一条入站消息（阻塞直到有消息）。"""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """发布一条发往渠道的出站消息。"""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """消费下一条出站消息（阻塞直到有消息）。"""
        return await self.outbound.get()

    @property
    def inbound_size(self) -> int:
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        return self.outbound.qsize()
