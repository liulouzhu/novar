"""RAG 混合检索工具 — Milvus 向量 + Elasticsearch BM25 + RRF 融合

Redis 缓存：明确作用域的 RAG 查询在 TTL 内直接返回缓存结果，按 user_id 隔离。
"""

import asyncio
import logging
import os

import numpy as np

from core.database import get_connection, get_embeddings_by_paper_ids
from core.embedding import embed_text_async, get_embedding_dim, EmbeddingProviderError
from core.paper_id import canonicalize_paper_id
from tools.result import ok, fail, truncate, MAX_CHUNK_TEXT

# 缓存 helper（纯计算，无 Redis 依赖）
from novare.cache import make_cache_key, cacheable_size

# Redis 访问（安全降级）
try:
    from web.backend.redis_service import redis_service
except Exception:
    redis_service = None

logger = logging.getLogger("research-server.rag_query")

_CACHE_TTL = 600  # 10 分钟
_MIN_TOP_K = 1
_MAX_TOP_K = 50

DEFAULT_USER_ID = os.getenv("RAG_DEFAULT_USER", "default")
ALLOW_UNSCOPED = os.getenv("RAG_ALLOW_UNSCOPED", "").lower() in ("1", "true", "yes")

# 混合检索配置
VECTOR_TOP_N = int(os.getenv("RAG_VECTOR_TOP_N", "50"))
KEYWORD_TOP_N = int(os.getenv("RAG_KEYWORD_TOP_N", "50"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))
RERANK_ENABLED = os.getenv("RAG_RERANK_ENABLED", "false").lower() in ("1", "true", "yes")
RERANK_CANDIDATES = max(1, int(os.getenv("RAG_RERANK_CANDIDATES", "20")))
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "qwen3-rerank")


# ── RRF 融合 ──────────────────────────────────────────────────────────────


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    rrf_k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion (RRF) 融合多路检索结果。

    RRF(d) = Σ 1 / (rrf_k + rank(d))

    同一 chunk 被多路召回时合并为一条，保留各路的 rank 和 score。
    """
    # chunk_id → 融合记录
    fused: dict[int, dict] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list, 1):
            cid = item["chunk_id"]
            if cid not in fused:
                fused[cid] = {
                    "chunk_id": cid,
                    "paper_id": item["paper_id"],
                    "title": item.get("title", ""),
                    "section": item.get("section", ""),
                    "text": item.get("text", ""),
                    "vector_rank": None,
                    "keyword_rank": None,
                    "vector_score": None,
                    "keyword_score": None,
                    "fusion_score": 0.0,
                }

            entry = fused[cid]
            source = item.get("source", "vector")
            if source == "vector":
                entry["vector_rank"] = rank
                entry["vector_score"] = round(item.get("score", 0), 4)
            elif source == "keyword":
                entry["keyword_rank"] = rank
                entry["keyword_score"] = round(item.get("score", 0), 4)

            entry["fusion_score"] += 1.0 / (rrf_k + rank)

    # 按 fusion_score 降序排序
    results = sorted(fused.values(), key=lambda x: x["fusion_score"], reverse=True)
    for r in results:
        r["fusion_score"] = round(r["fusion_score"], 6)
    return results


# ── 核心检索函数 ──────────────────────────────────────────────────────────


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


async def _get_user_paper_ids(user_id: str) -> set[str]:
    """Get paper_ids where user has fulltext access.

    Raises on any failure (fail-closed).  Returns an empty set when the
    user simply has no papers — callers treat that as "no results".
    """
    if not user_id:
        raise PermissionError("user_id is required for RAG search")

    from web.backend.db.base import get_session_factory
    from web.backend.db.models import UserPaper
    from uuid import UUID
    from sqlalchemy import select

    async with get_session_factory()() as db:
        result = await db.execute(
            select(UserPaper.paper_id).where(
                UserPaper.user_id == UUID(user_id),
                UserPaper.has_fulltext_access.is_(True),
                UserPaper.deleted_at.is_(None),
            )
        )
        return {str(up[0]) for up in result.all()}


async def _brute_force_search(
    query_vec: np.ndarray, top_k: int, allowed_paper_ids: set[str]
) -> tuple[list[dict], int]:
    """Brute-force cosine similarity scoped to *allowed_paper_ids* only.

    Returns (results, total_scanned).  Never reads the full embeddings table.
    """
    if not allowed_paper_ids:
        return [], 0

    async with get_connection() as conn:
        scoped_embeddings = await get_embeddings_by_paper_ids(conn, allowed_paper_ids)

    if not scoped_embeddings:
        return [], 0

    query_dim = len(query_vec)
    results = []
    skipped_dim = 0
    matched_dim: int | None = None
    for emb in scoped_embeddings:
        emb_dim = len(emb["vec"])
        if matched_dim is None:
            matched_dim = emb_dim
        if emb_dim != query_dim:
            skipped_dim += 1
            continue
        score = _cosine_similarity(query_vec, emb["vec"])
        results.append({
            "score": score,
            "chunk_id": emb["chunk_id"],
            "text": emb["text"],
            "section": emb["section"],
            "paper_id": emb["paper_id"],
            "title": emb["title"],
            "source": "vector",
        })

    if skipped_dim > 0 and matched_dim is not None and matched_dim != query_dim:
        raise DimensionMismatchError(
            f"查询向量为 {query_dim} 维，索引向量为 {matched_dim} 维。"
            f"请恢复原 embedding 配置（DASHSCOPE_API_KEY + text-embedding-v4）或重新构建索引。"
        )
    elif skipped_dim > 0:
        logger.warning(
            "Brute-force search: skipped %d embeddings with mismatched dimension "
            "(query=%d). Papers may need re-arsing.", skipped_dim, query_dim
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k], len(scoped_embeddings)


class DimensionMismatchError(Exception):
    """查询向量与索引向量维度不一致"""
    pass


async def _milvus_search(
    query_vec: list[float], top_k: int, user_id: str,
    paper_ids: list[str] | None = None,
) -> list[dict]:
    """Try Milvus vector search. Uses asyncio.to_thread for non-blocking execution."""
    from core.vector_store import search_vectors

    # Milvus search_vectors 是同步函数，放到线程池执行避免阻塞事件循环
    hits = await asyncio.to_thread(
        search_vectors, user_id, query_vec, top_k, paper_ids=paper_ids,
    )
    if not hits:
        return []

    # 批量查询 PostgreSQL 获取 title/section（一次 WHERE IN 查询）
    chunk_ids = [h["chunk_id"] for h in hits]
    chunk_meta: dict[int, tuple[str, str]] = {}  # chunk_id → (title, section)

    async with get_connection() as conn:
        from web.backend.db.models import Chunk, Paper
        from sqlalchemy import select
        result = await conn.execute(
            select(Chunk.id, Paper.title, Chunk.section)
            .join(Paper, Chunk.paper_id == Paper.id)
            .where(Chunk.id.in_(chunk_ids))
        )
        for row in result.all():
            chunk_meta[row[0]] = (row[1], row[2] or "")

    # 构建结果，保持 Milvus 返回顺序
    id_to_hit = {h["chunk_id"]: h for h in hits}
    results = []
    for cid in chunk_ids:
        h = id_to_hit[cid]
        title, section = chunk_meta.get(cid, (h["paper_id"], ""))
        results.append({
            "score": h["score"],
            "chunk_id": h["chunk_id"],
            "text": h["text"],
            "section": section,
            "paper_id": h["paper_id"],
            "title": title,
            "source": "vector",
        })
    return results


async def _es_search(
    question: str, paper_ids: list[str], top_n: int,
) -> tuple[list[dict], bool, str | None]:
    """Elasticsearch BM25 检索。

    Returns:
        (hits, available, error) — 区分「不可用」和「无匹配」
    """
    try:
        from core.elasticsearch_store import search_chunks
        result = await search_chunks(question, paper_ids, top_n=top_n)
        return result.hits, result.available, result.error
    except Exception as e:
        logger.warning("Elasticsearch search failed: %s", e)
        return [], False, str(e)


async def _rerank_results(
    question: str, candidates: list[dict],
) -> tuple[list[dict], bool, str | None]:
    """Call the optional reranker without making it a retrieval dependency."""
    try:
        from core.reranker import rerank_chunks
        result = await rerank_chunks(question, candidates)
        return result.hits, result.available, result.error
    except Exception as e:
        logger.warning("Rerank failed: %s", e)
        return [], False, str(e)


# ── 缓存作用域判断 helpers ─────────────────────────────────────────────────


def _resolve_paper_ids(args: dict) -> list[str]:
    """从 args 中提取 paper_id/paper_ids，规范化后返回去重排序列表。"""
    ids: list[str] = []
    if "paper_id" in args and args["paper_id"]:
        ids.append(canonicalize_paper_id(str(args["paper_id"])))
    if "paper_ids" in args and args["paper_ids"]:
        ids.extend(canonicalize_paper_id(str(pid)) for pid in args["paper_ids"] if pid)
    return sorted(set(ids))


def _build_canonical_to_stored_map(stored_ids: set[str]) -> dict[str, set[str]]:
    """构建 canonical ID → 实际存储 ID 的映射。

    例如：
    - stored: {"2308.11681", "arxiv:2308.11681"}
    - canonical: "arxiv:2308.11681"
    - 映射: {"arxiv:2308.11681": {"2308.11681", "arxiv:2308.11681"}}
    """
    mapping: dict[str, set[str]] = {}
    for stored_id in stored_ids:
        canonical = canonicalize_paper_id(stored_id)
        if canonical not in mapping:
            mapping[canonical] = set()
        mapping[canonical].add(stored_id)
    return mapping


def _resolve_to_stored_ids(
    requested_canonical: list[str],
    canonical_to_stored: dict[str, set[str]],
) -> set[str]:
    """将请求的 canonical ID 解析为实际存储 ID。

    返回所有匹配的 stored IDs（并集）。
    """
    result: set[str] = set()
    for canonical in requested_canonical:
        stored = canonical_to_stored.get(canonical)
        if stored:
            result.update(stored)
    return result


def _has_explicit_filters(args: dict) -> bool:
    """判断 args 中是否有明确过滤字段（不含 question/top_k/paper_id/paper_ids）。"""
    filter_keys = {"filters", "paper_filter", "source", "year_from", "year_to"}
    return any(k in args and args[k] for k in filter_keys)


def _validate_top_k(top_k: int | None) -> int:
    """校验并规范化 top_k 参数"""
    if top_k is None:
        return 5
    try:
        val = int(top_k)
    except (TypeError, ValueError):
        return 5
    return max(_MIN_TOP_K, min(_MAX_TOP_K, val))


# ── 主入口 ────────────────────────────────────────────────────────────────


async def handle_rag_query(args: dict, user_id: str = None, allow_unscoped: bool | None = None) -> str:
    """RAG 混合检索入口 — Milvus 向量 + ES BM25 + RRF 融合。

    安全策略：默认 fail-closed。
    - 无 user_id 且未显式启用 RAG_ALLOW_UNSCOPED → 拒绝。
    - 用户权限查询失败 → 拒绝（不降级为全库检索）。
    - paper_id/paper_ids 显式过滤与用户权限取交集。
    - allow_unscoped: 可选，per-call 覆盖全局 ALLOW_UNSCOPED（用于并发评测）。
    """
    question = args.get("question", "").strip()
    if not question:
        return fail("rag_query", "请提供查询问题。")

    # per-call 覆盖全局 ALLOW_UNSCOPED（用于并发评测，避免竞态）
    _allow_unscoped = allow_unscoped if allow_unscoped is not None else ALLOW_UNSCOPED

    final_top_k = _validate_top_k(args.get("top_k", 5))

    # ── 明确作用域检测 + Redis 缓存命中 ──
    _scoped_paper_ids = _resolve_paper_ids(args)
    _has_filters = _has_explicit_filters(args)
    _is_scoped = bool(_scoped_paper_ids or _has_filters)

    _rs = redis_service
    cache_key: str | None = None
    if _is_scoped and user_id:
        cache_key = make_cache_key("rag_query", user_id, {
            "question": question,
            "top_k": final_top_k,
            "paper_ids": sorted(_scoped_paper_ids) if _scoped_paper_ids else [],
            "filters": {k: args[k] for k in sorted(args) if k not in ("question", "top_k", "paper_id", "paper_ids")},
            "rerank": {
                "enabled": RERANK_ENABLED,
                "model": RERANK_MODEL if RERANK_ENABLED else None,
                "candidates": RERANK_CANDIDATES if RERANK_ENABLED else 0,
            },
        })
        if cache_key and _rs and _rs.is_available:
            try:
                cached = await _rs.get_json(cache_key)
                if cached and isinstance(cached, dict) and "result" in cached:
                    logger.info("rag_query cache hit: %s", cache_key)
                    return cached["result"]
            except Exception:
                logger.debug("cache read failed (non-fatal)", exc_info=True)

    # ── 权限校验：无 user_id 时默认拒绝 ──
    if not user_id and not _allow_unscoped:
        return fail(
            "rag_query",
            "缺少 user_id，无法执行安全的 RAG 检索。"
            "请设置 NOVARE_USER_ID 环境变量（CLI 模式），或在 Web 模式下使用。"
        )

    # 计算查询向量
    try:
        query_vec_list = await embed_text_async(question)
        query_vec = np.array(query_vec_list, dtype=np.float32)
    except EmbeddingProviderError as e:
        return fail("rag_query", str(e))
    except Exception as e:
        return fail("rag_query", f"查询向量化失败 - {e}")

    # ── 解析用户可访问的 paper_id 集合（fail-closed）──
    allowed_paper_ids: set[str] | None = None
    if user_id:
        try:
            allowed_paper_ids = await _get_user_paper_ids(user_id)
        except Exception as e:
            logger.error("RAG authorization failed for user %s: %s", user_id, e)
            return fail("rag_query", f"无法验证用户论文权限，检索已中止: {e}")
        if not allowed_paper_ids:
            return fail("rag_query", "您没有已解析全文的论文。请先使用 paper_parse 解析至少一篇论文。")

    # ── paper_id/paper_ids 过滤：与用户权限取交集（支持历史 ID 兼容）──
    filtered_paper_ids: list[str] | None = None
    if _scoped_paper_ids and allowed_paper_ids is not None:
        # 构建 canonical → stored ID 映射
        canonical_to_stored = _build_canonical_to_stored_map(allowed_paper_ids)
        # 将请求的 canonical IDs 解析为实际存储 IDs
        stored_match = _resolve_to_stored_ids(_scoped_paper_ids, canonical_to_stored)
        if not stored_match:
            return fail("rag_query", "所指定的论文不在您可访问的范围内。")
        filtered_paper_ids = sorted(stored_match)
    elif _scoped_paper_ids and _allow_unscoped and not user_id:
        filtered_paper_ids = _scoped_paper_ids

    # 实际检索用的 paper_ids 集合
    effective_paper_ids: set[str] | None = None
    if filtered_paper_ids is not None:
        effective_paper_ids = set(filtered_paper_ids)
    elif allowed_paper_ids is not None:
        effective_paper_ids = allowed_paper_ids
    elif _allow_unscoped and not user_id:
        effective_paper_ids = None  # None 表示全库

    # ── 双路并发召回 ──
    vector_hits: list[dict] = []
    keyword_hits: list[dict] = []
    warnings: list[str] = []
    milvus_ok = True
    es_available = True

    # 用于 ES 过滤的 paper_ids（必须是非空列表）
    es_paper_ids = filtered_paper_ids or (sorted(effective_paper_ids) if effective_paper_ids else [])
    # unscoped 全库模式：从数据库取所有 paper_ids 给 ES
    if not es_paper_ids and effective_paper_ids is None and not user_id and _allow_unscoped:
        try:
            from core.database import get_connection
            from web.backend.db.models import Paper
            from sqlalchemy import select
            async with get_connection() as conn:
                result = await conn.execute(
                    select(Paper.id).where(Paper.deleted_at.is_(None))
                )
                es_paper_ids = [r[0] for r in result.all()]
        except Exception as e:
            logger.warning("Failed to fetch all paper_ids for ES: %s", e)

    # 并发执行两路检索
    async def _vector_search():
        nonlocal vector_hits, milvus_ok
        try:
            milvus_user = user_id or DEFAULT_USER_ID
            vector_hits = await _milvus_search(
                query_vec_list, VECTOR_TOP_N, milvus_user,
                paper_ids=filtered_paper_ids,
            )
        except Exception as e:
            milvus_ok = False
            logger.warning("Milvus search failed: %s", e)
            warnings.append(f"Milvus unavailable: {e}")

    async def _keyword_search():
        nonlocal keyword_hits, es_available
        if not es_paper_ids:
            # 无 paper_ids 时跳过 ES（ES 不允许无范围全库检索）
            es_available = False
            return
        hits, available, error = await _es_search(question, es_paper_ids, KEYWORD_TOP_N)
        keyword_hits = hits
        if not available:
            es_available = False
            if error:
                warnings.append(f"Elasticsearch unavailable: {error}")

    await asyncio.gather(_vector_search(), _keyword_search())

    # ── Brute-force fallback：当两路都为空时 ──
    if not vector_hits and not keyword_hits:
        if effective_paper_ids is not None:
            # 有明确 paper_ids 范围，尝试 scoped brute-force
            try:
                bf_results, bf_total = await _brute_force_search(
                    query_vec, VECTOR_TOP_N, effective_paper_ids,
                )
                vector_hits = bf_results
            except DimensionMismatchError as e:
                return fail("rag_query", str(e))
        elif effective_paper_ids is None and not user_id and _allow_unscoped:
            # 无作用域全库 brute-force fallback（仅在显式允许时）
            from core.database import get_all_embeddings
            async with get_connection() as conn:
                all_embeddings = await get_all_embeddings(conn)
            if not all_embeddings:
                return fail("rag_query", "论文库为空。请先使用 paper_parse 解析至少一篇论文。")
            results = []
            for emb in all_embeddings:
                if len(emb["vec"]) != len(query_vec):
                    continue
                score = _cosine_similarity(query_vec, emb["vec"])
                results.append({
                    "score": score,
                    "chunk_id": emb["chunk_id"],
                    "text": emb["text"],
                    "section": emb["section"],
                    "paper_id": emb["paper_id"],
                    "title": emb["title"],
                    "source": "vector",
                })
            results.sort(key=lambda x: x["score"], reverse=True)
            vector_hits = results[:VECTOR_TOP_N]

    # 降级日志（只在 ES 真正不可用时报告，不误报空结果）
    if not vector_hits and keyword_hits:
        logger.info("Degraded to Elasticsearch BM25 only")
        if not milvus_ok and "Milvus unavailable" not in " ".join(warnings):
            warnings.append("Milvus unavailable; degraded to keyword retrieval")
    elif vector_hits and not keyword_hits:
        logger.info("Degraded to Milvus vector only")
        if not es_available and "Elasticsearch unavailable" not in " ".join(warnings):
            warnings.append("Elasticsearch unavailable; degraded to vector retrieval")

    if not vector_hits and not keyword_hits:
        if _scoped_paper_ids:
            return fail("rag_query", "在指定的论文中未找到相关内容。")
        return fail("rag_query", "未找到相关内容。")

    # ── RRF 融合 ──
    fused_results = reciprocal_rank_fusion(
        [vector_hits, keyword_hits] if keyword_hits else [vector_hits],
        rrf_k=RRF_K,
    )

    # ── Qwen3 rerank（失败时保留 RRF 顺序）──
    rerank_applied = False
    rerank_candidate_count = 0
    if RERANK_ENABLED:
        rerank_candidate_count = min(
            len(fused_results), max(final_top_k, RERANK_CANDIDATES),
        )
        rerank_candidates = fused_results[:rerank_candidate_count]
        reranked, rerank_available, rerank_error = await _rerank_results(
            question, rerank_candidates,
        )
        if rerank_available and reranked:
            final_results = reranked[:final_top_k]
            rerank_applied = True
        else:
            final_results = fused_results[:final_top_k]
            message = "Rerank unavailable; kept RRF ordering"
            if rerank_error:
                message += f" ({rerank_error})"
            warnings.append(message)
    else:
        final_results = fused_results[:final_top_k]

    # ── 构建结构化结果 ──
    results_json = []
    sources = []
    for i, r in enumerate(final_results, 1):
        results_json.append({
            "rank": i,
            "chunk_id": r["chunk_id"],
            "paper_id": r["paper_id"],
            "title": r["title"],
            "section": r["section"],
            "text": truncate(r.get("text", ""), MAX_CHUNK_TEXT),
            "vector_rank": r.get("vector_rank"),
            "keyword_rank": r.get("keyword_rank"),
            "vector_score": r.get("vector_score"),
            "keyword_score": r.get("keyword_score"),
            "fusion_score": r.get("fusion_score"),
            "rerank_score": r.get("rerank_score"),
            "rerank_rank": r.get("rerank_rank"),
        })
        sources.append({
            "id": r["paper_id"],
            "title": r["title"],
            "section": r["section"],
            "chunk_id": r["chunk_id"],
        })

    unique_papers = len(set(r["paper_id"] for r in final_results))

    # 确定搜索方法
    if vector_hits and keyword_hits:
        search_method = "hybrid_rrf"
    elif vector_hits:
        search_method = "Milvus"
    else:
        search_method = "Elasticsearch BM25"
    base_search_method = search_method
    if rerank_applied:
        search_method = f"{search_method} + {RERANK_MODEL}"

    result = ok(
        "rag_query",
        {
            "question": question,
            "search_method": search_method,
            "vector_candidates": len(vector_hits),
            "keyword_candidates": len(keyword_hits),
            "fused_candidates": len(fused_results),
            "rerank_candidates": rerank_candidate_count,
            "rerank_applied": rerank_applied,
            "rerank_model": RERANK_MODEL if RERANK_ENABLED else None,
            "returned_results": len(final_results),
            "unique_papers": unique_papers,
            "results": results_json,
        },
        summary=f"检索到 {len(final_results)} 条相关片段（来自 {unique_papers} 篇论文, "
                f"向量候选 {len(vector_hits)}, 关键词候选 {len(keyword_hits)}, "
                f"融合候选 {len(fused_results)}）",
        sources=sources,
        providers=[base_search_method, RERANK_MODEL] if rerank_applied else [base_search_method],
        warnings=warnings,
    )

    # ── Redis 缓存：写入成功结果（仅限明确作用域查询） ──
    if cache_key and _rs and _rs.is_available:
        try:
            if cacheable_size(result):
                await _rs.set_json(cache_key, {"result": result}, ttl=_CACHE_TTL)
        except Exception:
            logger.debug("cache write failed (non-fatal)", exc_info=True)

    return result
