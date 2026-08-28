"""共享 Embedding 模块 — 供 Web 情景记忆和 MCP 论文嵌入使用。

优先使用百炼 DashScope text-embedding-v4，测试时可用 numpy hash fallback。
无 API Key 且未显式启用测试 fallback 时抛出 RuntimeError。
异步接口通过 asyncio.to_thread 包装同步调用。

初始化状态机：
  UNINITIALIZED → SUCCESS | FAILED
  reset_embedder() → UNINITIALIZED
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import logging
import os
import threading

logger = logging.getLogger("novare.embedding")

# 嵌入维度常量 — 所有来源统一，不可更改
EMBEDDING_DIMENSION = 1024


class _InitState(enum.Enum):
    UNINITIALIZED = "uninitialized"
    SUCCESS = "success"
    FAILED = "failed"


# ── 全局状态（线程安全通过 _init_lock 保护）─────────────────────

_init_lock = threading.Lock()
_init_state = _InitState.UNINITIALIZED
_embedder = None          # bailian config dict, or None for numpy_fallback
_embedder_type: str | None = None
_init_error: RuntimeError | None = None  # 缓存初始化失败的异常


def _get_bailian_config() -> dict | None:
    """获取百炼 API 配置。"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-v4")
    return {
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "model": model,
    }


def _bailian_embed_sync(config: dict, texts: list[str]) -> list[list[float]]:
    """同步调用百炼 embedding API。"""
    import httpx
    url = f"{config['base_url']}/embeddings"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    results: list[list[float]] = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        body = {
            "model": config["model"],
            "input": batch,
            "dimensions": EMBEDDING_DIMENSION,
        }
        resp = httpx.post(url, json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        results.extend([item["embedding"] for item in sorted_data])
    return results


def _numpy_fallback_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic hash-based 向量化，输出固定 EMBEDDING_DIMENSION 维。
    仅用于测试，受 NOVARE_TEST_EMBEDDING_FALLBACK 控制。
    """
    dim = EMBEDDING_DIMENSION
    results: list[list[float]] = []
    for text in texts:
        raw = hashlib.sha512(text.encode("utf-8")).digest()
        required_bytes = dim * 4
        extended = (raw * ((required_bytes // len(raw)) + 1))[:required_bytes]
        import numpy as np
        arr = np.frombuffer(extended, dtype=np.uint32).astype(np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        results.append(arr.tolist())
    return results


def _init_embedder_unlocked() -> None:
    """在 _init_lock 保护下执行初始化。调用者必须持有 _init_lock。"""
    global _embedder, _embedder_type, _init_state, _init_error

    config = _get_bailian_config()
    if config:
        _embedder = config
        _embedder_type = "bailian"
        _init_state = _InitState.SUCCESS
        _init_error = None
        logger.info("Embedding: Bailian %s (dim=%d)", config["model"], EMBEDDING_DIMENSION)
        return

    # 无 API Key 时，检查是否显式启用了测试 fallback
    test_fallback = os.environ.get("NOVARE_TEST_EMBEDDING_FALLBACK", "").lower() in ("1", "true", "yes")
    if test_fallback:
        _embedder = None
        _embedder_type = "numpy_fallback"
        _init_state = _InitState.SUCCESS
        _init_error = None
        logger.warning("Embedding: numpy hash fallback enabled (NOVARE_TEST_EMBEDDING_FALLBACK=1). "
                       "This is for testing only, do NOT use in production.")
        return

    _init_state = _InitState.FAILED
    _init_error = RuntimeError(
        "Embedding unavailable: DASHSCOPE_API_KEY not set and "
        "NOVARE_TEST_EMBEDDING_FALLBACK not enabled. "
        "Set DASHSCOPE_API_KEY or NOVARE_TEST_EMBEDDING_FALLBACK=true in .env."
    )
    raise _init_error


def _ensure_init() -> tuple[str, dict | None]:
    """确保 embedder 已初始化，返回 (type, config)。

    状态语义：
    - UNINITIALIZED → 尝试初始化
    - SUCCESS → 直接返回缓存
    - FAILED → 抛出缓存的 RuntimeError
    """
    global _init_state, _init_error

    with _init_lock:
        if _init_state == _InitState.SUCCESS:
            return _embedder_type, _embedder

        if _init_state == _InitState.FAILED:
            raise _init_error  # type: ignore[misc]

        # UNINITIALIZED → 尝试初始化
        try:
            _init_embedder_unlocked()
            return _embedder_type, _embedder
        except RuntimeError:
            # _init_state and _init_error already set by _init_embedder_unlocked
            raise


def reset_embedder() -> None:
    """重置 embedder 状态，下次调用时重新初始化。"""
    global _embedder, _embedder_type, _init_state, _init_error
    with _init_lock:
        _embedder = None
        _embedder_type = None
        _init_state = _InitState.UNINITIALIZED
        _init_error = None


def get_embedding_dimension() -> int:
    """获取嵌入维度 — 始终返回 EMBEDDING_DIMENSION (1024)。"""
    _ensure_init()
    return EMBEDDING_DIMENSION


def get_embedding_model_name() -> str:
    """获取当前嵌入模型名称。"""
    etype, config = _ensure_init()
    if etype == "bailian" and config:
        return config.get("model", "text-embedding-v4")
    if etype == "numpy_fallback":
        return "numpy-hash-test-v1"
    return "unknown"


def embed_text(text: str) -> list[float]:
    """同步单条嵌入。"""
    etype, config = _ensure_init()
    if etype == "bailian":
        return _bailian_embed_sync(config, [text])[0]
    return _numpy_fallback_embed([text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """同步批量嵌入。"""
    if not texts:
        return []
    etype, config = _ensure_init()
    if etype == "bailian":
        return _bailian_embed_sync(config, texts)
    return _numpy_fallback_embed(texts)


async def embed_text_async(text: str) -> list[float]:
    """异步单条嵌入 — 同步调用通过 asyncio.to_thread 包装。"""
    return await asyncio.to_thread(embed_text, text)


async def embed_batch_async(texts: list[str]) -> list[list[float]]:
    """异步批量嵌入。"""
    if not texts:
        return []
    return await asyncio.to_thread(embed_batch, texts)
