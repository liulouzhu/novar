"""novare/channels/manager.py — 渠道生命周期管理

负责初始化、启动、停止所有渠道，以及出站消息的路由分发。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from novare.channels.bus import MessageBus
from novare.channels.base import BaseChannel
from novare.channels.events import OutboundMessage

logger = logging.getLogger("novare.channels")

# 指数退避重试延迟
_SEND_RETRY_DELAYS = (1, 2, 4)


class ChannelManager:
    """管理所有聊天渠道的生命周期和消息路由。"""

    def __init__(self, channels_config: dict[str, Any], bus: MessageBus):
        """
        Args:
            channels_config: 渠道配置字典，key 为渠道名，value 为渠道配置 dict。
                             例: {"weixin": {"enabled": True, "allow_from": ["*"], "token": "..."}}
            bus: 共享的消息总线。
        """
        self.channels_config = channels_config
        self.bus = bus
        self.channels: dict[str, BaseChannel] = {}
        self._dispatch_task: asyncio.Task | None = None
        self._send_max_retries: int = 3
        self._send_tool_hints: bool = False
        self._send_progress: bool = False

        self._init_channels()

    def _init_channels(self) -> None:
        """自动发现并实例化已启用的渠道。

        支持两种配置格式：
        - 单实例: {"weixin": {"enabled": true, ...}}
        - 多实例: {"weixin": [{"enabled": true, "instance_id": "personal", ...},
                              {"enabled": true, "instance_id": "work", ...}]}
        """
        from novare.channels.registry import discover_all

        all_classes = discover_all()

        for key, section in self.channels_config.items():
            # 支持 list 配置（同一渠道多实例）
            sections = section if isinstance(section, list) else [section]

            for idx, cfg in enumerate(sections):
                if not isinstance(cfg, dict):
                    continue
                if not cfg.get("enabled", False):
                    continue

                # 确定渠道类：优先 channel_type 字段，否则用 config key 名匹配模块名
                channel_type = cfg.get("channel_type", key)
                cls = all_classes.get(channel_type)
                if cls is None:
                    logger.warning("Unknown channel type: %s (from key '%s')", channel_type, key)
                    continue

                # 生成唯一实例名
                instance_id = cfg.get("instance_id", "")
                if len(sections) > 1:
                    # 多实例：必须有 instance_id 或自动生成
                    if not instance_id:
                        instance_id = str(idx)
                    instance_name = f"{channel_type}:{instance_id}"
                else:
                    # 单实例：用 config key 作为 name（允许 key 与模块名不同）
                    instance_name = key

                # 将 instance_id 注入 config，供渠道区分状态目录
                # 单实例且无显式 instance_id 时不注入（避免 weixin_weixin 这样的冗余路径）
                if instance_id:
                    cfg["instance_id"] = instance_id

                try:
                    channel = cls(cfg, self.bus)
                    # 设置渠道内部 name 为唯一实例名（用于出站路由）
                    channel.name = instance_name
                    self.channels[instance_name] = channel
                    logger.info("%s channel enabled (name=%s)", cls.display_name, instance_name)
                except Exception as e:
                    logger.warning("%s channel not available: %s", instance_name, e)

        self._validate_allow_from()

    def _validate_allow_from(self) -> None:
        for name, ch in self.channels.items():
            allow = getattr(ch.config, "allow_from", None)
            if allow is not None and allow == []:
                raise SystemExit(
                    f'Error: "{name}" has empty allow_from (denies all). '
                    f'Set ["*"] to allow everyone, or add specific user IDs.'
                )

    async def _start_channel(self, name: str, channel: BaseChannel) -> None:
        try:
            await channel.start()
        except Exception as e:
            logger.error("Failed to start channel %s: %s", name, e)

    async def start_all(self) -> None:
        """启动所有渠道和出站消息分发器。"""
        if not self.channels:
            logger.warning("No channels enabled")
            return

        # 启动出站分发器
        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())

        # 启动各渠道（每个渠道一个 task，它们会自行长期运行）
        tasks = []
        for name, channel in self.channels.items():
            logger.info("Starting %s channel...", name)
            tasks.append(asyncio.create_task(self._start_channel(name, channel)))

        # 等待所有渠道结束（正常情况下不会结束）
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_all(self) -> None:
        """停止所有渠道和分发器。"""
        logger.info("Stopping all channels...")

        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

        for name, channel in self.channels.items():
            try:
                await channel.stop()
                logger.info("Stopped %s channel", name)
            except Exception as e:
                logger.error("Error stopping %s: %s", name, e)

    async def _dispatch_outbound(self) -> None:
        """分发出站消息到对应的渠道。"""
        logger.info("Outbound dispatcher started")
        pending: list[OutboundMessage] = []

        while True:
            try:
                if pending:
                    msg = pending.pop(0)
                else:
                    msg = await asyncio.wait_for(
                        self.bus.consume_outbound(),
                        timeout=1.0,
                    )

                # 过滤 progress / tool_hint 消息
                if msg.metadata.get("_progress"):
                    if msg.metadata.get("_tool_hint") and not self._send_tool_hints:
                        continue
                    if not msg.metadata.get("_tool_hint") and not self._send_progress:
                        continue

                # 合并连续的流式 delta
                if msg.metadata.get("_stream_delta") and not msg.metadata.get("_stream_end"):
                    msg, extra_pending = self._coalesce_stream_deltas(msg)
                    pending.extend(extra_pending)

                channel = self.channels.get(msg.channel)
                if channel:
                    await self._send_with_retry(channel, msg)
                else:
                    logger.warning("Unknown channel: %s", msg.channel)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    @staticmethod
    async def _send_once(channel: BaseChannel, msg: OutboundMessage) -> None:
        if msg.metadata.get("_stream_delta") or msg.metadata.get("_stream_end"):
            await channel.send_delta(msg.chat_id, msg.content, msg.metadata)
        elif not msg.metadata.get("_streamed"):
            await channel.send(msg)

    def _coalesce_stream_deltas(
        self, first_msg: OutboundMessage,
    ) -> tuple[OutboundMessage, list[OutboundMessage]]:
        """合并队列中连续的流式 delta 消息。"""
        target_key = (first_msg.channel, first_msg.chat_id)
        combined_content = first_msg.content
        final_metadata = dict(first_msg.metadata or {})
        non_matching: list[OutboundMessage] = []

        while True:
            try:
                next_msg = self.bus.outbound.get_nowait()
            except asyncio.QueueEmpty:
                break

            same_target = (next_msg.channel, next_msg.chat_id) == target_key
            is_delta = next_msg.metadata and next_msg.metadata.get("_stream_delta")
            is_end = next_msg.metadata and next_msg.metadata.get("_stream_end")

            if same_target and is_delta and not final_metadata.get("_stream_end"):
                combined_content += next_msg.content
                if is_end:
                    final_metadata["_stream_end"] = True
                    break
            else:
                non_matching.append(next_msg)
                break

        merged = OutboundMessage(
            channel=first_msg.channel,
            chat_id=first_msg.chat_id,
            content=combined_content,
            metadata=final_metadata,
        )
        return merged, non_matching

    async def _send_with_retry(self, channel: BaseChannel, msg: OutboundMessage) -> None:
        """带指数退避重试的发送。"""
        max_attempts = max(self._send_max_retries, 1)

        for attempt in range(max_attempts):
            try:
                await self._send_once(channel, msg)
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if attempt == max_attempts - 1:
                    logger.error(
                        "Failed to send to %s after %d attempts: %s - %s",
                        msg.channel, max_attempts, type(e).__name__, e,
                    )
                    return
                delay = _SEND_RETRY_DELAYS[min(attempt, len(_SEND_RETRY_DELAYS) - 1)]
                logger.warning(
                    "Send to %s failed (attempt %d/%d): %s, retrying in %ds",
                    msg.channel, attempt + 1, max_attempts, type(e).__name__, delay,
                )
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    raise

    def get_channel(self, name: str) -> BaseChannel | None:
        return self.channels.get(name)

    def get_status(self) -> dict[str, Any]:
        return {
            name: {"enabled": True, "running": channel.is_running}
            for name, channel in self.channels.items()
        }

    @property
    def enabled_channels(self) -> list[str]:
        return list(self.channels.keys())
