"""Milvus vector store for paper chunk embeddings.

Provides insert/search operations with per-user partition key isolation.
Falls back gracefully when Milvus is unavailable.

Dimension safety:
- insert_vectors validates that embedding dimension matches EMBEDDING_DIM
- search_vectors can filter by paper_id in addition to user_id
- Schema compatibility check on collection load
"""

import json
import os
import logging

from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

logger = logging.getLogger(__name__)

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION_NAME = "paper_chunks"
EMBEDDING_DIM = 1024  # Match DashScope text-embedding-v4


class DimensionMismatchError(Exception):
    """向量维度与 collection schema 不一致"""
    pass


def connect_milvus() -> None:
    """Establish connection to Milvus server."""
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    logger.info("Connected to Milvus at %s:%s", MILVUS_HOST, MILVUS_PORT)


def _ensure_connected() -> None:
    """Ensure an active Milvus connection exists, auto-connecting if needed."""
    try:
        active = connections.list_connections()
        # pymilvus returns list of alias strings; default alias is "default"
        if not active or "default" not in active:
            connect_milvus()
    except Exception:
        try:
            connect_milvus()
        except Exception as e:
            logger.warning("Cannot connect to Milvus at %s:%s: %s", MILVUS_HOST, MILVUS_PORT, e)
            raise


def _check_schema_compatibility(collection: Collection) -> int:
    """检查 collection schema 中 embedding 字段的维度，返回维度值。

    如果 schema 不兼容（如缺少 embedding 字段），抛出异常。
    """
    schema = collection.schema
    for field in schema.fields:
        if field.name == "embedding":
            params = field.params
            dim = params.get("dim", 0)
            if dim != EMBEDDING_DIM:
                logger.warning(
                    "Milvus schema dimension mismatch: collection has dim=%d, "
                    "expected dim=%d. Consider rebuilding the collection.",
                    dim, EMBEDDING_DIM,
                )
            return dim
    raise DimensionMismatchError(
        "Milvus collection 'paper_chunks' schema missing 'embedding' field. "
        "The collection may need to be recreated."
    )


def ensure_collection() -> Collection:
    """Get or create the paper_chunks collection with proper schema and index."""
    _ensure_connected()
    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(COLLECTION_NAME)
        # 检查 schema 兼容性
        try:
            actual_dim = _check_schema_compatibility(collection)
            if actual_dim != EMBEDDING_DIM:
                logger.warning(
                    "Milvus collection dimension (%d) differs from expected (%d). "
                    "Queries may fail. Rebuild collection to fix.",
                    actual_dim, EMBEDDING_DIM,
                )
        except Exception as e:
            logger.warning("Failed to check Milvus schema: %s", e)
        return collection

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
        FieldSchema(
            name="user_id", dtype=DataType.VARCHAR, max_length=64,
            is_partition_key=True,
        ),
        FieldSchema(name="paper_id", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    schema = CollectionSchema(
        fields, description="Paper chunks with user partition key"
    )
    collection = Collection(COLLECTION_NAME, schema)
    collection.create_index(
        "embedding",
        {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        },
    )
    logger.info("Created Milvus collection: %s (dim=%d)", COLLECTION_NAME, EMBEDDING_DIM)
    return collection


def insert_vectors(
    user_id: str,
    paper_id: str,
    chunk_ids: list[int],
    texts: list[str],
    embeddings: list[list[float]],
    flush: bool = True,
) -> None:
    """Insert chunk vectors into Milvus.

    Args:
        user_id: Partition key for user isolation.
        paper_id: Paper identifier.
        chunk_ids: Integer chunk IDs (matching SQLite chunks table).
        texts: Chunk text content.
        embeddings: Corresponding embedding vectors.
        flush: Flush the collection immediately after insertion.

    Raises:
        DimensionMismatchError: If any embedding dimension doesn't match EMBEDDING_DIM.
    """
    # 维度校验
    for i, emb in enumerate(embeddings):
        if len(emb) != EMBEDDING_DIM:
            raise DimensionMismatchError(
                f"Embedding dimension mismatch at index {i}: "
                f"got {len(emb)}, expected {EMBEDDING_DIM}. "
                f"Check DASHSCOPE_API_KEY and EMBEDDING_MODEL configuration."
            )

    collection = ensure_collection()
    ids = [f"{user_id}_{paper_id}_{cid}" for cid in chunk_ids]
    data = [
        ids,
        [user_id] * len(ids),
        [paper_id] * len(ids),
        chunk_ids,
        texts,
        embeddings,
    ]
    collection.insert(data)
    if flush:
        collection.flush()
    logger.info(
        "Inserted %d vectors for user=%s paper=%s (dim=%d)",
        len(ids), user_id, paper_id, EMBEDDING_DIM,
    )


def delete_vectors(paper_id: str, user_id: str | None = None) -> int:
    """Delete vectors for one user-paper pair, or all users for a paper."""
    _ensure_connected()
    if not utility.has_collection(COLLECTION_NAME):
        return 0

    collection = Collection(COLLECTION_NAME)
    expr_parts = [f"paper_id == {json.dumps(paper_id)}"]
    if user_id is not None:
        expr_parts.append(f"user_id == {json.dumps(user_id)}")
    result = collection.delete(" and ".join(expr_parts))
    collection.flush()
    deleted = int(getattr(result, "delete_count", 0) or 0)
    logger.info(
        "Deleted %d vectors for paper=%s user=%s",
        deleted,
        paper_id,
        user_id or "*",
    )
    return deleted


def search_vectors(
    user_id: str,
    query_embedding: list[float],
    top_k: int = 5,
    paper_ids: list[str] | None = None,
) -> list[dict]:
    """Search for similar chunk vectors scoped to a user, optionally filtered by paper_ids.

    Args:
        user_id: Partition key to scope search.
        query_embedding: Query vector.
        top_k: Number of results to return.
        paper_ids: Optional list of paper IDs to filter by.

    Returns:
        List of hit dicts with keys: id, score, paper_id, chunk_id, text.

    Raises:
        DimensionMismatchError: If query embedding dimension doesn't match collection schema.
    """
    # 查询向量维度校验
    if len(query_embedding) != EMBEDDING_DIM:
        raise DimensionMismatchError(
            f"Query embedding dimension mismatch: got {len(query_embedding)}, "
            f"expected {EMBEDDING_DIM}. "
            f"Index vectors are {EMBEDDING_DIM}D. "
            f"Please restore the original embedding configuration or rebuild the index."
        )

    collection = ensure_collection()
    collection.load()

    # 构建过滤表达式（使用 json.dumps 安全转义字符串）
    expr_parts = [f"user_id == {json.dumps(user_id)}"]
    if paper_ids:
        # Milvus expression: paper_id in ["id1", "id2", ...]
        encoded_ids = ", ".join(json.dumps(pid) for pid in paper_ids)
        expr_parts.append(f"paper_id in [{encoded_ids}]")
    expr = " and ".join(expr_parts)

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 16}},
        limit=top_k,
        expr=expr,
        output_fields=["paper_id", "chunk_id", "text"],
    )
    hits = []
    for hit in results[0]:
        hits.append(
            {
                "id": hit.id,
                "score": hit.score,
                "paper_id": hit.get("paper_id"),
                "chunk_id": hit.get("chunk_id"),
                "text": hit.get("text"),
            }
        )
    return hits
