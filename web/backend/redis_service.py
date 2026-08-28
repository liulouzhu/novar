"""web/backend/redis_service.py — 统一 Redis 访问封装

Redis disabled 或连接失败时安全降级，不抛业务异常。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("novare.redis")


class RedisService:
    """统一的异步 Redis 访问层。

    - disabled 时所有写方法静默返回降级值，读方法返回 None / False
    - 连接失败后标记为 unavailable，后续调用跳过连接尝试
    """

    def __init__(self) -> None:
        self._enabled: bool = False
        self._available: bool = False
        self._client: Any = None  # redis.asyncio.Redis | None

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    async def initialize(self, enabled: bool, url: str) -> None:
        """应用启动时调用。enabled=False 时跳过连接。"""
        self._enabled = enabled
        if not enabled:
            logger.info("Redis disabled by config, running without Redis")
            return

        try:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(url, decode_responses=True)
            await self._client.ping()
            self._available = True
            logger.info("Redis connected: %s", url)
        except Exception:
            logger.warning("Redis connection failed (%s), falling back to no-Redis mode", url, exc_info=True)
            self._available = False
            self._client = None

    async def close(self) -> None:
        """应用关闭时调用。"""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None
        self._available = False

    # ── 状态查询 ──────────────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Redis 是否已启用且连接正常。"""
        return self._enabled and self._available

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    # ── 基础操作 ──────────────────────────────────────────────────────────────

    async def set_nx(self, key: str, value: str, ttl: int) -> bool | None:
        """SET key NX EX ttl — 用于锁 / 去重。

        Returns True  — key 不存在且设置成功（获得锁 / 首次消息）；
                False — Redis 正常响应但 key 已存在（锁冲突 / 重复消息）；
                None  — Redis disabled / unavailable / 操作异常（降级）。
        """
        if not self._enabled or self._client is None:
            return None
        try:
            result = await self._client.set(key, value, nx=True, ex=ttl)
            return bool(result)
        except Exception:
            logger.warning("Redis set_nx failed for key=%s", key, exc_info=True)
            self._available = False
            return None

    async def get(self, key: str) -> str | None:
        """GET key — Redis 不可用时返回 None。"""
        if not self.is_available or self._client is None:
            return None
        try:
            return await self._client.get(key)
        except Exception:
            logger.warning("Redis get failed for key=%s", key, exc_info=True)
            self._available = False
            return None

    async def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        """SET key value — 通用写入，可选 TTL。

        Returns True 设置成功，False Redis 不可用或操作异常。
        """
        if not self.is_available or self._client is None:
            return False
        try:
            if ttl is not None:
                await self._client.set(key, value, ex=ttl)
            else:
                await self._client.set(key, value)
            return True
        except Exception:
            logger.warning("Redis set failed for key=%s", key, exc_info=True)
            self._available = False
            return False

    async def delete(self, key: str) -> None:
        """DEL key — Redis 不可用时静默跳过。"""
        if not self.is_available or self._client is None:
            return
        try:
            await self._client.delete(key)
        except Exception:
            logger.warning("Redis delete failed for key=%s", key, exc_info=True)
            self._available = False

    async def delete_prefix(self, prefix: str) -> bool:
        """Delete keys under a cache prefix without using blocking KEYS."""
        if not self._enabled:
            return True
        if not self.is_available or self._client is None:
            return False
        try:
            batch: list[str] = []
            async for key in self._client.scan_iter(match=f"{prefix}*", count=200):
                batch.append(key)
                if len(batch) >= 200:
                    await self._client.delete(*batch)
                    batch.clear()
            if batch:
                await self._client.delete(*batch)
            return True
        except Exception:
            logger.warning("Redis delete_prefix failed for prefix=%s", prefix, exc_info=True)
            self._available = False
            return False

    async def delete_if_value(self, key: str, expected_value: str) -> bool:
        """原子 compare-and-delete（Lua 脚本）。

        Returns True 表示删除成功（value 匹配）；
                False 表示 value 不匹配或 Redis 不可用。
        后续可作为 lock release 的安全版本。
        """
        if not self.is_available or self._client is None:
            return False
        try:
            lua = """
            if redis.call('get', KEYS[1]) == ARGV[1] then
                return redis.call('del', KEYS[1])
            else
                return 0
            end
            """
            result = await self._client.eval(lua, 1, key, expected_value)
            return bool(result)
        except Exception:
            logger.warning("Redis delete_if_value failed for key=%s", key, exc_info=True)
            self._available = False
            return False

    async def ping(self) -> bool:
        """PING — 用于健康检查。"""
        if not self.is_available or self._client is None:
            return False
        try:
            return await self._client.ping()
        except Exception:
            self._available = False
            return False

    async def set_json(self, key: str, value: dict, ttl: int | None = None) -> bool:
        """SET key JSON — 存储小状态对象。

        Returns True 设置成功，False Redis 不可用或序列化失败。
        """
        if not self.is_available or self._client is None:
            return False
        try:
            import json
            data = json.dumps(value, ensure_ascii=False)
            if ttl is not None:
                await self._client.set(key, data, ex=ttl)
            else:
                await self._client.set(key, data)
            return True
        except Exception:
            logger.warning("Redis set_json failed for key=%s", key, exc_info=True)
            self._available = False
            return False

    async def get_json(self, key: str) -> dict | None:
        """GET key → JSON 解码 — key 不存在或 decode 失败返回 None。"""
        if not self.is_available or self._client is None:
            return None
        try:
            raw = await self._client.get(key)
            if raw is None:
                return None
            import json
            return json.loads(raw)
        except Exception:
            logger.warning("Redis get_json failed for key=%s", key, exc_info=True)
            self._available = False
            return None

    async def expire(self, key: str, ttl: int) -> bool:
        """EXPIRE key ttl — 续期。Redis 不可用返回 False。"""
        if not self.is_available or self._client is None:
            return False
        try:
            return bool(await self._client.expire(key, ttl))
        except Exception:
            logger.warning("Redis expire failed for key=%s", key, exc_info=True)
            self._available = False
            return False


# 全局单例
redis_service = RedisService()
