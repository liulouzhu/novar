"""向量化模块 - 百炼 text-embedding-v4 优先，本地 fallback

安全策略：
- 生产环境没有可用 embedding provider 时抛出明确错误
- numpy hash fallback 只在 NOVARE_TEST_EMBEDDING_FALLBACK=true 时允许
- 日志中不包含 API key
"""

import logging
import os

import httpx

logger = logging.getLogger("research-server.embedding")

# 全局缓存
_embedder = None
_embedder_type = None
_init_attempted = False

# 百炼 API 临时失败时的重试窗口（秒）
_BAILIAN_RETRY_INTERVAL = 120

# 当前使用的 embedding 维度（全局可查询）
_current_dim: int | None = None


class EmbeddingProviderError(Exception):
    """没有可用的 embedding provider 时抛出"""
    pass


def _get_bailian_embedder():
    """百炼 text-embedding-v4（OpenAI 兼容 API）"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    base_url = os.environ.get("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-v4")

    if not api_key:
        return None

    # 日志中不包含 API key，只显示 provider 信息
    logger.info("Using Bailian embedder: model=%s, base_url=%s, dim=1024", model, base_url)
    return ("bailian", {
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "model": model,
        "dim": 1024,
    })


def _get_local_embedder():
    """sentence-transformers all-MiniLM-L6-v2 本地模型"""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Using local sentence-transformers all-MiniLM-L6-v2, dim=384")
        return ("local", model)
    except ImportError:
        logger.warning("sentence-transformers not installed")
        return None


def _get_numpy_fallback():
    """最简单的 hash-based 向量化（仅用于测试）"""
    import numpy as np

    class NumpyFallback:
        dim = 128

        def encode(self, text: str) -> list[float]:
            import hashlib
            h = hashlib.sha512(text.encode()).digest()
            data = h * (self.dim * 4 // len(h) + 1)
            arr = np.frombuffer(data[:self.dim * 4], dtype=np.uint32).astype(np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            return arr.tolist()

        def encode_batch(self, texts: list[str]) -> list[list[float]]:
            return [self.encode(t) for t in texts]

    logger.warning("Using numpy hash fallback embedder (TESTING ONLY, dim=128)")
    return ("numpy_fallback", NumpyFallback())


def _init_embedder():
    """初始化嵌入模型，优先 百炼 → 本地 → 测试 fallback"""
    global _embedder, _embedder_type, _init_attempted, _current_dim

    result = _get_bailian_embedder()
    if result:
        _embedder_type, _embedder = result
        _current_dim = _embedder["dim"]
        _init_attempted = True
        return

    result = _get_local_embedder()
    if result:
        _embedder_type, _embedder = result
        _current_dim = 384
        _init_attempted = True
        return

    # 检查是否允许测试 fallback
    allow_test_fallback = os.environ.get("NOVARE_TEST_EMBEDDING_FALLBACK", "").lower() in ("1", "true", "yes")
    if allow_test_fallback:
        _embedder_type, _embedder = _get_numpy_fallback()
        _current_dim = 128
        _init_attempted = True
    else:
        _init_attempted = True
        raise EmbeddingProviderError(
            "未配置 DASHSCOPE_API_KEY，且本地 sentence-transformers 模型不可用。"
            "请设置 DASHSCOPE_API_KEY 环境变量，或安装 sentence-transformers。"
            "如需在测试中使用 numpy fallback，请设置 NOVARE_TEST_EMBEDDING_FALLBACK=true。"
        )


def reset_embedder():
    """重置 embedder 状态，下次调用时重新初始化（用于百炼 API 恢复后重试）"""
    global _embedder, _embedder_type, _init_attempted, _current_dim
    _embedder = None
    _embedder_type = None
    _init_attempted = False
    _current_dim = None
    logger.info("Embedder reset, will re-initialize on next use")


def get_embedder():
    """获取嵌入模型实例，支持百炼 API 恢复后自动重试"""
    global _embedder, _embedder_type

    if _embedder is not None:
        # 如果当前是 fallback，且百炼 API key 存在，定期重试百炼
        if _embedder_type == "numpy_fallback" and os.environ.get("DASHSCOPE_API_KEY"):
            # 用模块属性记录上次重试时间
            last_retry = getattr(get_embedder, '_last_retry', 0)
            import time
            now = time.time()
            if now - last_retry > _BAILIAN_RETRY_INTERVAL:
                get_embedder._last_retry = now
                result = _get_bailian_embedder()
                if result:
                    logger.info("Bailian API recovered, switching back from fallback")
                    _embedder_type, _embedder = result
        return _embedder_type, _embedder

    _init_embedder()
    return _embedder_type, _embedder


def get_embedding_dim() -> int:
    """获取向量维度"""
    global _current_dim
    if _current_dim is not None:
        return _current_dim
    embedder_type, embedder = get_embedder()
    if embedder_type == "bailian":
        return embedder["dim"]
    elif embedder_type == "local":
        return 384
    else:
        return embedder.dim


def get_embedding_provider_info() -> dict:
    """获取当前 embedding provider 信息（不包含密钥）"""
    embedder_type, embedder = get_embedder()
    info = {"type": embedder_type, "dim": get_embedding_dim()}
    if embedder_type == "bailian":
        info["model"] = embedder.get("model", "unknown")
        info["base_url"] = embedder.get("base_url", "unknown")
    elif embedder_type == "local":
        info["model"] = "all-MiniLM-L6-v2"
    return info


# ── 异步版本（MCP 工具调用）─────────────────────────────────────────────

async def _bailian_embed_async(config: dict, texts: list[str]) -> list[list[float]]:
    """异步调用百炼 embedding API"""
    url = f"{config['base_url']}/embeddings"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    results = []
    batch_size = 10  # Bailian text-embedding-v4 max batch size is 10
    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            body = {
                "model": config["model"],
                "input": batch,
                "dimensions": config["dim"],
            }
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            results.extend([item["embedding"] for item in sorted_data])
    return results


async def embed_text_async(text: str) -> list[float]:
    """异步单条向量化"""
    embedder_type, embedder = get_embedder()
    if embedder_type == "bailian":
        return (await _bailian_embed_async(embedder, [text]))[0]
    elif embedder_type == "local":
        return embedder.encode(text, normalize_embeddings=True).tolist()
    else:
        return embedder.encode(text)


async def embed_batch_async(texts: list[str]) -> list[list[float]]:
    """异步批量向量化"""
    if not texts:
        return []
    embedder_type, embedder = get_embedder()
    if embedder_type == "bailian":
        return await _bailian_embed_async(embedder, texts)
    elif embedder_type == "local":
        vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=64)
        return [v.tolist() for v in vecs]
    else:
        return embedder.encode_batch(texts)


# ── 同步版本（非 MCP 场景）─────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """同步单条向量化"""
    embedder_type, embedder = get_embedder()
    if embedder_type == "bailian":
        url = f"{embedder['base_url']}/embeddings"
        headers = {
            "Authorization": f"Bearer {embedder['api_key']}",
            "Content-Type": "application/json",
        }
        body = {"model": embedder["model"], "input": text, "dimensions": embedder["dim"]}
        resp = httpx.post(url, json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    elif embedder_type == "local":
        return embedder.encode(text, normalize_embeddings=True).tolist()
    else:
        return embedder.encode(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """同步批量向量化"""
    if not texts:
        return []
    embedder_type, embedder = get_embedder()
    if embedder_type == "bailian":
        url = f"{embedder['base_url']}/embeddings"
        headers = {"Authorization": f"Bearer {embedder['api_key']}", "Content-Type": "application/json"}
        results = []
        for i in range(0, len(texts), 10):  # Bailian max batch size is 10
            batch = texts[i:i + 10]
            body = {"model": embedder["model"], "input": batch, "dimensions": embedder["dim"]}
            resp = httpx.post(url, json=body, headers=headers, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            results.extend([item["embedding"] for item in sorted_data])
        return results
    elif embedder_type == "local":
        vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=64)
        return [v.tolist() for v in vecs]
    else:
        return embedder.encode_batch(texts)
