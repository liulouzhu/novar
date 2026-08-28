"""Milvus 向量存储 — 情景记忆的语义索引。

所有同步 Milvus 调用通过 asyncio.to_thread 包装，不阻塞事件循环。
Milvus 不可用时返回空结果，不影响主聊天。

PyMilvus 3.0.0 API:
- connections.has_connection(alias) 检查是否已连接
- connections.connect(alias, host, port) 建立连接
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any

from novare.embedding import EMBEDDING_DIMENSION

logger = logging.getLogger("novare.episodic_memory.vector_store")

# ── 配置 ──────────────────────────────────────────────────────

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION_NAME = os.getenv("NOVARE_EPISODIC_MEMORY_COLLECTION", "episodic_memories")

# 超时（秒）
_MILVUS_TIMEOUT = 10
_CONNECT_TIMEOUT = 5

# 模块级线程锁，保护 Collection 创建竞态
_collection_lock = threading.Lock()

# 必须存在的业务字段
_REQUIRED_FIELDS = frozenset({"id", "user_id", "session_id", "memory_type", "text",
                               "occurred_at", "importance", "confidence"})


class IncompatibleCollectionSchemaError(RuntimeError):
    """Milvus Collection schema 与当前实现不兼容。"""
    pass


def _build_user_filter(user_id: str) -> str:
    """构建 Milvus user_id 过滤表达式。可单独测试。"""
    safe = user_id.replace('"', '\\"')
    return f'user_id == "{safe}"'


def _ensure_connected_sync() -> None:
    """确保 Milvus alias "episodic" 已连接（同步，线程安全）。

    使用 connections.has_connection(alias) 检查（PyMilvus 3.0.0 API）。
    """
    from pymilvus import connections
    if connections.has_connection("episodic"):
        return
    connections.connect(
        alias="episodic",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )
    logger.info("Connected to Milvus at %s:%s (alias=episodic)", MILVUS_HOST, MILVUS_PORT)


def _validate_collection_schema(col) -> None:
    """验证已有 Collection 的 schema 和索引。不兼容时抛出 IncompatibleCollectionSchemaError。

    验证项：
    1. embedding 字段存在且为 FLOAT_VECTOR
    2. embedding dimension 等于 EMBEDDING_DIMENSION
    3. 必要业务字段存在
    4. embedding 索引存在且使用 COSINE metric
    """
    from pymilvus import DataType

    schema = col.schema
    field_map = {f.name: f for f in schema.fields}
    field_names = set(field_map.keys())

    # 1. embedding 字段存在
    if "embedding" not in field_names:
        raise IncompatibleCollectionSchemaError(
            f"Collection '{COLLECTION_NAME}' missing required 'embedding' field"
        )

    emb_field = field_map["embedding"]

    # 2. embedding 是 FLOAT_VECTOR
    if emb_field.dtype != DataType.FLOAT_VECTOR:
        raise IncompatibleCollectionSchemaError(
            f"Collection '{COLLECTION_NAME}' field 'embedding' has type {emb_field.dtype}, "
            f"expected FLOAT_VECTOR"
        )

    # 3. dimension 正确
    dim = emb_field.params.get("dim", 0)
    if dim != EMBEDDING_DIMENSION:
        raise IncompatibleCollectionSchemaError(
            f"Collection '{COLLECTION_NAME}' embedding dim={dim}, expected {EMBEDDING_DIMENSION}"
        )

    # 4. 必要业务字段存在
    missing = _REQUIRED_FIELDS - field_names
    if missing:
        raise IncompatibleCollectionSchemaError(
            f"Collection '{COLLECTION_NAME}' missing required fields: {sorted(missing)}"
        )

    # 5. embedding 索引存在且使用 COSINE
    indexes = col.indexes
    emb_index = None
    for idx in indexes:
        if idx.field_name == "embedding":
            emb_index = idx
            break
    if emb_index is None:
        raise IncompatibleCollectionSchemaError(
            f"Collection '{COLLECTION_NAME}' has no index on 'embedding' field"
        )
    params = emb_index.params
    metric = params.get("metric_type", "")
    if metric != "COSINE":
        raise IncompatibleCollectionSchemaError(
            f"Collection '{COLLECTION_NAME}' embedding index uses metric '{metric}', expected 'COSINE'"
        )


def _get_collection_sync(create_if_missing: bool = True):
    """同步获取 Collection：确保连接 → 检查存在 → 校验 schema → 必要时创建。

    使用 threading.Lock 保护创建流程，防止并发竞态。
    所有获取已有 Collection 的路径都必须通过 schema 校验。
    不兼容时抛出 IncompatibleCollectionSchemaError，不静默返回。
    可在 asyncio.to_thread 的同步函数中直接调用。
    """
    _ensure_connected_sync()
    from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility

    # 先快速检查（无锁）
    if utility.has_collection(COLLECTION_NAME, using="episodic"):
        col = Collection(COLLECTION_NAME, using="episodic")
        _validate_collection_schema(col)  # 不兼容时抛异常
        return col

    if not create_if_missing:
        return None

    # 加锁保护创建流程
    with _collection_lock:
        # 双重检查：另一个线程可能已经创建了
        if utility.has_collection(COLLECTION_NAME, using="episodic"):
            col = Collection(COLLECTION_NAME, using="episodic")
            _validate_collection_schema(col)  # 不兼容时抛异常
            return col

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="occurred_at", dtype=DataType.INT64),
            FieldSchema(name="importance", dtype=DataType.FLOAT),
            FieldSchema(name="confidence", dtype=DataType.FLOAT),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION),
        ]
        schema = CollectionSchema(fields, description="Episodic memories with user isolation")
        try:
            col = Collection(COLLECTION_NAME, schema, using="episodic")
            col.create_index(
                "embedding",
                {
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 128},
                },
                using="episodic",
            )
            logger.info("Created Milvus collection: %s", COLLECTION_NAME)
            return col
        except Exception as e:
            # AlreadyExists 竞态：另一个线程/进程先创建了
            if "AlreadyExists" in str(e) or "already exists" in str(e).lower():
                logger.info("Collection %s already exists (race), fetching existing", COLLECTION_NAME)
                col = Collection(COLLECTION_NAME, using="episodic")
                _validate_collection_schema(col)  # 不兼容时抛异常
                return col
            raise


class EpisodicMemoryVectorStore:
    """Milvus 向量存储，惰性连接，支持 asyncio.to_thread 包装。

    所有公开方法独立可用，不依赖调用顺序。
    进程重启后第一次调用也能正确连接已有 Collection。
    schema 不兼容时抛出明确异常，由调用方决定降级策略。
    """

    def __init__(self):
        pass

    async def _run_sync(self, func, *args, **kwargs):
        """在 asyncio.to_thread 中执行同步函数，带超时。"""
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args, **kwargs),
            timeout=_MILVUS_TIMEOUT,
        )

    async def ensure_collection(self) -> None:
        """确保 episodic_memories collection 存在（惰性）。"""
        try:
            await self._run_sync(_get_collection_sync, True)
        except Exception as e:
            logger.warning("Failed to ensure Milvus collection: %s", e)
            raise

    async def insert_memory(
        self,
        memory_id: str,
        user_id: str,
        session_id: str,
        memory_type: str,
        text: str,
        occurred_at: int,
        importance: float,
        confidence: float,
        embedding: list[float],
    ) -> None:
        """写入单条情景记忆向量。schema 不兼容时抛出异常。"""
        def _do():
            col = _get_collection_sync(create_if_missing=True)
            data = [
                [memory_id],
                [user_id],
                [session_id],
                [memory_type],
                [text[:2048]],
                [occurred_at],
                [importance],
                [confidence],
                [embedding],
            ]
            col.insert(data)
            col.flush()

        await self._run_sync(_do)
        logger.info("Inserted episodic memory vector: %s", memory_id)

    async def search_memories(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """按 user_id 搜索相似情景记忆。schema 不兼容时返回空列表。"""
        def _do():
            col = _get_collection_sync(create_if_missing=False)
            if col is None:
                return []
            col.load()
            expr = _build_user_filter(user_id)
            results = col.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=top_k,
                expr=expr,
                output_fields=["session_id", "memory_type", "text", "occurred_at", "importance", "confidence"],
            )
            hits = []
            for hit in results[0]:
                hits.append({
                    "id": hit.id,
                    "score": hit.score,
                    "session_id": hit.get("session_id"),
                    "memory_type": hit.get("memory_type"),
                    "text": hit.get("text"),
                    "occurred_at": hit.get("occurred_at"),
                    "importance": hit.get("importance"),
                    "confidence": hit.get("confidence"),
                })
            return hits

        try:
            return await self._run_sync(_do)
        except IncompatibleCollectionSchemaError:
            logger.warning("Milvus collection schema incompatible for %s, search degraded to empty", COLLECTION_NAME)
            return []
        except Exception as e:
            logger.warning("Milvus search failed, returning empty: %s", e)
            return []

    async def delete_memory(self, memory_id: str) -> bool:
        """删除单条记忆向量。schema 不兼容时返回 False。"""
        def _do():
            col = _get_collection_sync(create_if_missing=False)
            if col is None:
                return False
            col.delete(f'id == "{memory_id}"')
            col.flush()
            return True

        try:
            result = await self._run_sync(_do)
            if result:
                logger.info("Deleted episodic memory vector: %s", memory_id)
            else:
                logger.warning("Milvus delete: collection not found for %s", memory_id)
            return bool(result)
        except IncompatibleCollectionSchemaError:
            logger.warning("Milvus collection schema incompatible for %s, delete degraded", COLLECTION_NAME)
            return False
        except Exception as e:
            logger.warning("Milvus delete failed for %s: %s", memory_id, e)
            return False

    async def health_check(self) -> bool:
        """检查 Milvus 连接是否可用。"""
        def _do():
            from pymilvus import connections
            try:
                connections.connect(
                    alias="episodic_health_check",
                    host=MILVUS_HOST,
                    port=MILVUS_PORT,
                )
                connections.disconnect("episodic_health_check")
                return True
            except Exception:
                return False

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_do),
                timeout=_CONNECT_TIMEOUT,
            )
        except Exception:
            return False
