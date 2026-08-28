"""novare/cache.py — 轻量缓存 key 构造 + 大小检查

不依赖 Redis，只做纯计算。Redis 读写由调用方完成。
"""

from __future__ import annotations

import hashlib
import json


def stable_hash(payload: dict) -> str:
    """对 dict 做确定性 SHA-256，返回前 24 位 hex。

    sort_keys=True 保证相同内容不同 key 顺序产生相同 hash。
    default=str 兜底非 JSON 序列化类型（如 datetime）。
    """
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def make_cache_key(
    namespace: str,
    user_id: str | None,
    payload: dict,
    version: str = "v1",
) -> str | None:
    """构造缓存 key。

    有 user_id → cache:{namespace}:user:{user_id}:{version}:{hash}
    无 user_id → None（第一版不做全局缓存）
    """
    if not user_id:
        return None
    h = stable_hash(payload)
    return f"cache:{namespace}:user:{user_id}:{version}:{h}"


def cacheable_size(value: object, max_bytes: int = 512 * 1024) -> bool:
    """检查序列化后大小是否在限制内（默认 512KB）。"""
    try:
        size = len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
        return size <= max_bytes
    except Exception:
        return False
