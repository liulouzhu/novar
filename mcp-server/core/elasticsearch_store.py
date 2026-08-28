"""Elasticsearch BM25 存储模块

提供关键词检索能力，与 Milvus 向量检索互补。
ES 不可用时静默降级，不阻塞主流程。

环境变量：
    ELASTICSEARCH_URL — ES 地址（默认 http://localhost:9200）
    ELASTICSEARCH_INDEX — 索引名（默认 paper_chunks）
    ELASTICSEARCH_USERNAME — 用户名（可选）
    ELASTICSEARCH_PASSWORD — 密码（可选，日志中不打印）
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from elasticsearch import AsyncElasticsearch as _AsyncElasticsearch
    from elasticsearch.helpers import async_bulk as _async_bulk
except ImportError:
    _AsyncElasticsearch = None
    _async_bulk = None

logger = logging.getLogger("research-server.elasticsearch")

# ── 配置 ──────────────────────────────────────────────────────────────────

ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ELASTICSEARCH_INDEX", "paper_chunks")
ES_USERNAME = os.getenv("ELASTICSEARCH_USERNAME", "")
ES_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "")

# 重试配置
_RETRY_MIN_INTERVAL = 30   # 最小重试间隔（秒）
_RETRY_MAX_INTERVAL = 120  # 最大重试间隔（秒）
_REQUEST_TIMEOUT = 30      # 请求超时（秒）

# 全局客户端（延迟初始化）
_client = None
_last_failure_time: float = 0
_failure_count: int = 0
_client_lock = asyncio.Lock()


# ── 结构化结果 ────────────────────────────────────────────────────────────

@dataclass
class ESSearchResult:
    """ES 搜索结果，区分「不可用」和「无匹配」。"""
    hits: list[dict] = field(default_factory=list)
    available: bool = True
    error: str | None = None


# ── 文本清洗 ──────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """清洗文本用于 BM25 检索。

    去除 HTML 标签、Markdown 图片标记、表格语法等噪声，
    但不改变原始 text 字段。
    """
    if not text:
        return ""
    cleaned = text
    # HTML 标签
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    # Markdown 图片 ![alt](url)
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", cleaned)
    # Markdown 链接 [text](url) → text
    cleaned = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cleaned)
    # Markdown 表格分隔符 |---|
    cleaned = re.sub(r"\|[\s\-:]+\|", " ", cleaned)
    # 表格行首尾的 |
    cleaned = re.sub(r"^\|", " ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\|$", " ", cleaned, flags=re.MULTILINE)
    # 多个空白合并
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ── 客户端管理 ────────────────────────────────────────────────────────────

async def _get_client():
    """获取或创建 AsyncElasticsearch 客户端。

    支持指数退避 retry：失败后等待 30→60→120 秒再重试。
    使用 asyncio.Lock 防止并发创建多个 client。
    """
    global _client, _last_failure_time, _failure_count

    if _client is not None:
        return _client

    # 检查重试间隔（指数退避）
    now = time.monotonic()
    if _last_failure_time > 0 and _failure_count > 0:
        elapsed = now - _last_failure_time
        wait = min(
            _RETRY_MAX_INTERVAL,
            _RETRY_MIN_INTERVAL * 2 ** max(0, _failure_count - 1),
        )
        if elapsed < wait:
            return None

    async with _client_lock:
        # Double-check after acquiring lock
        if _client is not None:
            return _client

        try:
            if _AsyncElasticsearch is None:
                raise ImportError("elasticsearch not installed")

            kwargs: dict[str, Any] = {
                "hosts": [ES_URL],
                "request_timeout": _REQUEST_TIMEOUT,
            }
            if ES_USERNAME and ES_PASSWORD:
                kwargs["basic_auth"] = (ES_USERNAME, ES_PASSWORD)

            client = _AsyncElasticsearch(**kwargs)
            # 验证连接
            if await client.ping():
                logger.info("Connected to Elasticsearch at %s", ES_URL)
                _client = client
                _last_failure_time = 0
                _failure_count = 0
                return _client
            else:
                logger.warning("Elasticsearch ping failed at %s", ES_URL)
                await _safe_close(client)
                _failure_count += 1
                _last_failure_time = time.monotonic()
                return None
        except ImportError:
            logger.warning("elasticsearch[async] not installed, BM25 retrieval unavailable")
            _failure_count += 1
            _last_failure_time = time.monotonic()
            return None
        except Exception as e:
            logger.warning("Cannot connect to Elasticsearch at %s: %s", ES_URL, e)
            _failure_count += 1
            _last_failure_time = time.monotonic()
            return None


async def _safe_close(client) -> None:
    """安全关闭 client，忽略错误。"""
    try:
        await client.close()
    except Exception:
        pass


async def close_client() -> None:
    """关闭 ES 客户端连接。幂等。"""
    global _client, _last_failure_time, _failure_count
    old = _client
    _client = None
    if old is not None:
        await _safe_close(old)
    _last_failure_time = 0
    _failure_count = 0


def reset_client() -> None:
    """重置客户端状态（用于测试）。"""
    global _client, _last_failure_time, _failure_count
    _client = None
    _last_failure_time = 0
    _failure_count = 0


# ── 索引管理 ──────────────────────────────────────────────────────────────

_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "chunk_id": {"type": "long"},
            "paper_id": {"type": "keyword"},
            "title": {
                "type": "text",
                "fields": {
                    "keyword": {"type": "keyword"},
                },
            },
            "section": {"type": "keyword"},
            "text": {"type": "text"},
            "search_text": {"type": "text"},
        }
    }
}

_REQUIRED_FIELDS = {"chunk_id", "paper_id", "title", "section", "text", "search_text"}


async def ensure_index() -> bool:
    """确保 ES 索引存在且 mapping 兼容。返回是否成功。"""
    client = await _get_client()
    if client is None:
        return False

    try:
        if not await client.indices.exists(index=ES_INDEX):
            await client.indices.create(index=ES_INDEX, body=_INDEX_MAPPING)
            logger.info("Created Elasticsearch index: %s", ES_INDEX)
            return True

        # 索引已存在，检查 mapping 兼容性
        try:
            mapping = await client.indices.get_mapping(index=ES_INDEX)
            index_mapping = mapping.get(ES_INDEX, {}).get("mappings", {}).get("properties", {})
            missing = _REQUIRED_FIELDS - set(index_mapping.keys())
            if missing:
                logger.warning(
                    "Elasticsearch index '%s' missing fields: %s. "
                    "Consider reindexing.",
                    ES_INDEX, missing,
                )
        except Exception as e:
            logger.warning("Failed to check ES mapping: %s", e)
        return True
    except Exception as e:
        logger.warning("Failed to ensure Elasticsearch index: %s", e)
        return False


# ── 文档写入 ──────────────────────────────────────────────────────────────

async def bulk_upsert_chunks(documents: list[dict]) -> dict:
    """批量写入 chunk 文档到 ES。

    Args:
        documents: 文档列表，每项包含：
            - chunk_id: int
            - paper_id: str
            - title: str
            - section: str
            - text: str

    Returns:
        {"success": int, "errors": list[str]}
    """
    if not documents:
        return {"success": 0, "errors": []}

    # 确保索引存在
    if not await ensure_index():
        return {"success": 0, "errors": ["ES index not available"]}

    client = await _get_client()
    if client is None:
        return {"success": 0, "errors": ["ES client not available"]}

    # 构造 bulk actions — elasticsearch.helpers.async_bulk 格式
    actions = []
    for doc in documents:
        doc_id = f"{doc['paper_id']}:{doc['chunk_id']}"
        actions.append({
            "_op_type": "index",
            "_index": ES_INDEX,
            "_id": doc_id,
            "_source": {
                "chunk_id": doc["chunk_id"],
                "paper_id": doc["paper_id"],
                "title": doc.get("title", ""),
                "section": doc.get("section", ""),
                "text": doc.get("text", ""),
                "search_text": _clean_text(doc.get("text", "")),
            },
        })

    try:
        if _async_bulk is None:
            return {"success": 0, "errors": ["elasticsearch.helpers.async_bulk not available"]}

        error_messages: list[str] = []

        success, errors = await _async_bulk(
            client, actions, refresh="wait_for",
            raise_on_error=False,
        )
        if errors:
            for err in (errors if isinstance(errors, list) else [errors]):
                error_messages.append(str(err)[:200])
            logger.warning("ES bulk upsert had %d errors (of %d docs)", len(errors), len(documents))
        logger.info("ES bulk upsert: %d/%d succeeded", success, len(documents))
        return {"success": success, "errors": error_messages}
    except Exception as e:
        logger.warning("ES bulk upsert failed (non-fatal): %s", e)
        return {"success": 0, "errors": [str(e)[:200]]}


async def delete_paper_chunks(paper_id: str) -> int:
    """Delete all BM25 documents for a paper; safe to retry."""
    if _AsyncElasticsearch is None:
        return 0
    client = await _get_client()
    if client is None:
        raise RuntimeError("Elasticsearch client not available")
    if not await client.indices.exists(index=ES_INDEX):
        return 0

    response = await client.delete_by_query(
        index=ES_INDEX,
        body={"query": {"term": {"paper_id": paper_id}}},
        conflicts="proceed",
        refresh=True,
    )
    failures = response.get("failures") or []
    if failures:
        raise RuntimeError(f"Elasticsearch delete failures: {failures[:3]}")
    deleted = int(response.get("deleted", 0) or 0)
    logger.info("Deleted %d Elasticsearch chunks for paper=%s", deleted, paper_id)
    return deleted


# ── BM25 搜索 ─────────────────────────────────────────────────────────────

async def search_chunks(
    question: str,
    paper_ids: list[str],
    top_n: int = 50,
) -> ESSearchResult:
    """BM25 关键词检索。

    Args:
        question: 用户查询
        paper_ids: 必须指定，限定检索范围
        top_n: 召回数量

    Returns:
        ESSearchResult 区分「不可用」和「无匹配」
    """
    client = await _get_client()
    if client is None:
        return ESSearchResult(hits=[], available=False, error="ES client not available")

    if not paper_ids:
        return ESSearchResult(hits=[], available=True)

    query_body = {
        "size": top_n,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": question,
                            "fields": [
                                "title^4",
                                "section^2",
                                "search_text^1.5",
                                "text",
                            ],
                            "type": "best_fields",
                        }
                    }
                ],
                "filter": [
                    {
                        "terms": {
                            "paper_id": paper_ids,
                        }
                    }
                ],
            }
        },
    }

    try:
        resp = await client.search(index=ES_INDEX, body=query_body)
        hits = resp.get("hits", {}).get("hits", [])
        results = []
        for hit in hits:
            src = hit.get("_source", {})
            results.append({
                "chunk_id": src.get("chunk_id"),
                "paper_id": src.get("paper_id", ""),
                "title": src.get("title", ""),
                "section": src.get("section", ""),
                "text": src.get("text", ""),
                "score": hit.get("_score", 0),
                "source": "keyword",
            })
        return ESSearchResult(hits=results, available=True)
    except Exception as e:
        # 并发下偶发超时，重试一次
        logger.debug("ES search failed (attempt 1): %s", e)
        try:
            resp = await client.search(index=ES_INDEX, body=query_body)
            hits = resp.get("hits", {}).get("hits", [])
            results = []
            for hit in hits:
                src = hit.get("_source", {})
                results.append({
                    "chunk_id": src.get("chunk_id"),
                    "paper_id": src.get("paper_id", ""),
                    "title": src.get("title", ""),
                    "section": src.get("section", ""),
                    "text": src.get("text", ""),
                    "score": hit.get("_score", 0),
                    "source": "keyword",
                })
            return ESSearchResult(hits=results, available=True)
        except Exception as e2:
            logger.warning("ES search failed after retry: %s", e2)
            return ESSearchResult(hits=[], available=False, error=str(e2))
