"""tests/test_cache.py — 缓存 helper + paper_search + rag_query Redis 缓存测试

覆盖：
1. stable_hash 确定性
2. make_cache_key 按 user_id 隔离
3. cacheable_size 大小检查
4. paper_search cache hit / miss / unavailable / 用户隔离
5. rag_query cache: scoped detection / hit / miss / unavailable / isolation / error / oversized
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

import pytest

# mcp-server 工具需要 sys.path 中有 mcp-server 目录
_MCP_SERVER = str(Path(__file__).resolve().parent.parent / "mcp-server")
if _MCP_SERVER not in sys.path:
    sys.path.insert(0, _MCP_SERVER)

from novare.cache import stable_hash, make_cache_key, cacheable_size


# ══════════════════════════════════════════════════════════════════════════════
# 1. stable_hash
# ══════════════════════════════════════════════════════════════════════════════


class TestStableHash:
    def test_same_payload_same_hash(self):
        a = stable_hash({"query": "llm", "limit": 10})
        b = stable_hash({"query": "llm", "limit": 10})
        assert a == b

    def test_key_order_irrelevant(self):
        a = stable_hash({"query": "llm", "limit": 10})
        b = stable_hash({"limit": 10, "query": "llm"})
        assert a == b

    def test_different_payload_different_hash(self):
        a = stable_hash({"query": "llm"})
        b = stable_hash({"query": "rag"})
        assert a != b

    def test_returns_24_hex_chars(self):
        h = stable_hash({"x": 1})
        assert len(h) == 24
        assert all(c in "0123456789abcdef" for c in h)


# ══════════════════════════════════════════════════════════════════════════════
# 2. make_cache_key
# ══════════════════════════════════════════════════════════════════════════════


class TestMakeCacheKey:
    def test_with_user_id(self):
        key = make_cache_key("paper_search", "u-1", {"query": "llm"})
        assert key is not None
        assert "paper_search" in key
        assert "u-1" in key
        assert "v1" in key

    def test_without_user_id_returns_none(self):
        key = make_cache_key("paper_search", None, {"query": "llm"})
        assert key is None

    def test_empty_user_id_returns_none(self):
        key = make_cache_key("paper_search", "", {"query": "llm"})
        assert key is None

    def test_different_user_different_key(self):
        k1 = make_cache_key("paper_search", "u-a", {"query": "llm"})
        k2 = make_cache_key("paper_search", "u-b", {"query": "llm"})
        assert k1 != k2

    def test_no_query_in_key(self):
        key = make_cache_key("paper_search", "u-1", {"query": "secret topic"})
        assert "secret topic" not in key

    def test_custom_version(self):
        key = make_cache_key("ps", "u-1", {"q": "x"}, version="v2")
        assert "v2" in key


# ══════════════════════════════════════════════════════════════════════════════
# 3. cacheable_size
# ══════════════════════════════════════════════════════════════════════════════


class TestCacheableSize:
    def test_small_value(self):
        assert cacheable_size({"a": 1}) is True

    def test_large_value(self):
        big = "x" * (600 * 1024)
        assert cacheable_size(big, max_bytes=512 * 1024) is False

    def test_custom_limit(self):
        assert cacheable_size("x" * 100, max_bytes=50) is False
        assert cacheable_size("x" * 10, max_bytes=50) is True


# ══════════════════════════════════════════════════════════════════════════════
# 4–9. paper_search 缓存集成测试
# ══════════════════════════════════════════════════════════════════════════════


def _fake_search_result(query: str, count: int = 2) -> str:
    """Build a valid paper_search success JSON result."""
    papers = [
        {"paper_id": f"p{i}", "title": f"Paper {i} about {query}", "authors": ["A"], "year": 2024}
        for i in range(count)
    ]
    return json.dumps({
        "schema_version": 1, "tool": "paper_search", "ok": True,
        "summary": f"搜索 '{query}' 找到 {count} 篇论文",
        "data": {"query": query, "total": count, "papers": papers},
        "sources": [], "providers": ["semantic_scholar"], "warnings": [], "error": None,
    }, ensure_ascii=False)


class TestPaperSearchCache:
    """paper_search 的 Redis 缓存行为。"""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached(self):
        """命中缓存时直接返回，不调用搜索 API。"""
        cached_result = _fake_search_result("llm", 3)
        mock_rs = MagicMock()
        mock_rs.is_available = True
        mock_rs.get_json = AsyncMock(return_value={"result": cached_result})
        mock_rs.set_json = AsyncMock(return_value=True)

        with patch("tools.paper_search.redis_service", mock_rs), \
             patch("tools.paper_search._search_semantic_scholar") as mock_s2, \
             patch("tools.paper_search._search_arxiv") as mock_arxiv:
            from tools.paper_search import handle_paper_search
            result = await handle_paper_search({"query": "llm"}, user_id="u-1")

        assert result == cached_result
        mock_s2.assert_not_called()
        mock_arxiv.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_then_write(self):
        """缓存未命中时执行搜索，成功后写缓存，TTL=1800。"""
        mock_rs = MagicMock()
        mock_rs.is_available = True
        mock_rs.get_json = AsyncMock(return_value=None)
        mock_rs.set_json = AsyncMock(return_value=True)

        fake_papers = [{"id": "p1", "title": "T", "authors": [], "abstract": "", "year": 2024,
                        "source": "s2", "url": "", "pdf_url": None, "citation_count": 0}]

        with patch("tools.paper_search.redis_service", mock_rs), \
             patch("tools.paper_search._search_semantic_scholar",
                   new_callable=AsyncMock, return_value=(fake_papers, "")), \
             patch("tools.paper_search._search_arxiv",
                   new_callable=AsyncMock, return_value=([], "")), \
             patch("tools.paper_search.get_connection"), \
             patch("tools.paper_search.upsert_paper"):
            from tools.paper_search import handle_paper_search
            result = await handle_paper_search({"query": "transformer"}, user_id="u-1")

        # 搜索成功
        parsed = json.loads(result)
        assert parsed["ok"] is True

        # 写缓存被调用，TTL=1800
        mock_rs.set_json.assert_awaited_once()
        call_args = mock_rs.set_json.call_args
        assert call_args.kwargs.get("ttl") == 1800 or (len(call_args.args) >= 3 and call_args.args[2] == 1800)
        # 值中包含 result 字段
        cache_value = call_args.kwargs.get("value") or call_args.args[1]
        assert "result" in cache_value

    @pytest.mark.asyncio
    async def test_redis_unavailable_no_cache(self):
        """Redis 不可用时行为不变，不调用 get_json/set_json。"""
        mock_rs = MagicMock()
        mock_rs.is_available = False
        mock_rs.get_json = AsyncMock(return_value=None)
        mock_rs.set_json = AsyncMock(return_value=True)

        fake_papers = [{"id": "p1", "title": "T", "authors": [], "abstract": "", "year": 2024,
                        "source": "s2", "url": "", "pdf_url": None, "citation_count": 0}]

        with patch("tools.paper_search.redis_service", mock_rs), \
             patch("tools.paper_search._search_semantic_scholar",
                   new_callable=AsyncMock, return_value=(fake_papers, "")), \
             patch("tools.paper_search._search_arxiv",
                   new_callable=AsyncMock, return_value=([], "")), \
             patch("tools.paper_search.get_connection"), \
             patch("tools.paper_search.upsert_paper"):
            from tools.paper_search import handle_paper_search
            result = await handle_paper_search({"query": "bert"}, user_id="u-1")

        parsed = json.loads(result)
        assert parsed["ok"] is True
        mock_rs.get_json.assert_not_called()
        mock_rs.set_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_redis_none_no_cache(self):
        """redis_service 为 None（import 失败）时行为不变。"""
        fake_papers = [{"id": "p1", "title": "T", "authors": [], "abstract": "", "year": 2024,
                        "source": "s2", "url": "", "pdf_url": None, "citation_count": 0}]

        with patch("tools.paper_search.redis_service", None), \
             patch("tools.paper_search._search_semantic_scholar",
                   new_callable=AsyncMock, return_value=(fake_papers, "")), \
             patch("tools.paper_search._search_arxiv",
                   new_callable=AsyncMock, return_value=([], "")), \
             patch("tools.paper_search.get_connection"), \
             patch("tools.paper_search.upsert_paper"):
            from tools.paper_search import handle_paper_search
            result = await handle_paper_search({"query": "gpt"}, user_id="u-1")

        parsed = json.loads(result)
        assert parsed["ok"] is True

    @pytest.mark.asyncio
    async def test_no_user_id_no_cache(self):
        """无 user_id 时不缓存。"""
        mock_rs = MagicMock()
        mock_rs.is_available = True
        mock_rs.get_json = AsyncMock(return_value=None)
        mock_rs.set_json = AsyncMock(return_value=True)

        fake_papers = [{"id": "p1", "title": "T", "authors": [], "abstract": "", "year": 2024,
                        "source": "s2", "url": "", "pdf_url": None, "citation_count": 0}]

        with patch("tools.paper_search.redis_service", mock_rs), \
             patch("tools.paper_search._search_semantic_scholar",
                   new_callable=AsyncMock, return_value=(fake_papers, "")), \
             patch("tools.paper_search._search_arxiv",
                   new_callable=AsyncMock, return_value=([], "")), \
             patch("tools.paper_search.get_connection"), \
             patch("tools.paper_search.upsert_paper"):
            from tools.paper_search import handle_paper_search
            result = await handle_paper_search({"query": "llm"}, user_id=None)

        parsed = json.loads(result)
        assert parsed["ok"] is True
        # 不应调用缓存读写
        mock_rs.get_json.assert_not_called()
        mock_rs.set_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_result_not_cached(self):
        """搜索失败时不写缓存。"""
        mock_rs = MagicMock()
        mock_rs.is_available = True
        mock_rs.get_json = AsyncMock(return_value=None)
        mock_rs.set_json = AsyncMock(return_value=True)

        with patch("tools.paper_search.redis_service", mock_rs), \
             patch("tools.paper_search._search_semantic_scholar",
                   new_callable=AsyncMock, return_value=([], "S2 down")), \
             patch("tools.paper_search._search_arxiv",
                   new_callable=AsyncMock, return_value=([], "arXiv down")):
            from tools.paper_search import handle_paper_search
            result = await handle_paper_search({"query": "xyz"}, user_id="u-1")

        parsed = json.loads(result)
        assert parsed["ok"] is False
        mock_rs.set_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_oversized_result_not_cached(self):
        """cacheable_size 返回 False 时不缓存（mock cacheable_size）。"""
        mock_rs = MagicMock()
        mock_rs.is_available = True
        mock_rs.get_json = AsyncMock(return_value=None)
        mock_rs.set_json = AsyncMock(return_value=True)

        fake_papers = [{"id": "p1", "title": "T", "authors": [], "abstract": "", "year": 2024,
                        "source": "s2", "url": "", "pdf_url": None, "citation_count": 0}]

        with patch("tools.paper_search.redis_service", mock_rs), \
             patch("tools.paper_search._search_semantic_scholar",
                   new_callable=AsyncMock, return_value=(fake_papers, "")), \
             patch("tools.paper_search._search_arxiv",
                   new_callable=AsyncMock, return_value=([], "")), \
             patch("tools.paper_search.get_connection"), \
             patch("tools.paper_search.upsert_paper"), \
             patch("tools.paper_search.cacheable_size", return_value=False):
            from tools.paper_search import handle_paper_search
            result = await handle_paper_search({"query": "big"}, user_id="u-1")

        parsed = json.loads(result)
        assert parsed["ok"] is True
        # 超大结果不缓存
        mock_rs.set_json.assert_not_called()

    def test_user_isolation_key_differs(self):
        """相同 query 不同 user_id 生成不同缓存 key。"""
        payload = {"query": "attention mechanism", "limit": 10, "year_from": None, "year_to": None}
        k1 = make_cache_key("paper_search", "user-aaa", payload)
        k2 = make_cache_key("paper_search", "user-bbb", payload)
        assert k1 is not None and k2 is not None
        assert k1 != k2
        # 两个 key 都不包含明文 query
        assert "attention" not in k1
        assert "attention" not in k2


# ══════════════════════════════════════════════════════════════════════════════
# RAG query 缓存
# ══════════════════════════════════════════════════════════════════════════════


def _fake_rag_result(question: str, count: int = 3) -> str:
    """构造一个合法的 rag_query 成功 JSON 结果。"""
    results = [
        {"rank": i, "score": 0.9 - i * 0.1, "chunk_id": f"c{i}", "paper_id": "p1",
         "title": "T", "section": "Abstract", "text": f"chunk {i}"}
        for i in range(1, count + 1)
    ]
    return json.dumps({
        "schema_version": 1, "tool": "rag_query", "ok": True,
        "summary": f"检索到 {count} 条相关片段",
        "data": {"question": question, "results": results, "unique_papers": 1,
                 "total_chunks_searched": 10, "search_method": "brute-force"},
        "sources": [], "providers": ["brute-force"], "warnings": [], "error": None,
    }, ensure_ascii=False)


class TestRagScopedDetection:
    """rag_query 明确作用域判断。"""

    def test_paper_id_is_scoped(self):
        from tools.rag_query import _resolve_paper_ids, _has_explicit_filters
        ids = _resolve_paper_ids({"question": "q", "paper_id": "arxiv:1234"})
        assert ids == ["arxiv:1234"]
        assert _has_explicit_filters({"question": "q", "paper_id": "arxiv:1234"}) is False

    def test_paper_ids_is_scoped(self):
        from tools.rag_query import _resolve_paper_ids
        ids = _resolve_paper_ids({"question": "q", "paper_ids": ["p2", "p1"]})
        assert ids == ["p1", "p2"]  # sorted

    def test_filters_is_scoped(self):
        from tools.rag_query import _has_explicit_filters
        assert _has_explicit_filters({"question": "q", "filters": {"source": "arxiv"}}) is True

    def test_source_filter_is_scoped(self):
        from tools.rag_query import _has_explicit_filters
        assert _has_explicit_filters({"question": "q", "source": "arxiv"}) is True

    def test_year_filter_is_scoped(self):
        from tools.rag_query import _has_explicit_filters
        assert _has_explicit_filters({"question": "q", "year_from": 2020}) is True

    def test_only_query_not_scoped(self):
        from tools.rag_query import _resolve_paper_ids, _has_explicit_filters
        ids = _resolve_paper_ids({"question": "q"})
        assert ids == []
        assert _has_explicit_filters({"question": "q"}) is False


class TestRagCacheKeyNormalization:
    """rag_query cache key 归一化。"""

    def test_same_params_same_key(self):
        payload = {"question": "q", "top_k": 5, "paper_ids": ["p1", "p2"], "filters": {}}
        k1 = make_cache_key("rag_query", "u-1", payload)
        k2 = make_cache_key("rag_query", "u-1", payload)
        assert k1 == k2

    def test_paper_ids_order_irrelevant(self):
        """paper_ids sorted by _resolve_paper_ids before cache key generation."""
        from tools.rag_query import _resolve_paper_ids
        ids_a = _resolve_paper_ids({"question": "q", "paper_ids": ["p1", "p2"]})
        ids_b = _resolve_paper_ids({"question": "q", "paper_ids": ["p2", "p1"]})
        assert ids_a == ids_b
        k1 = make_cache_key("rag_query", "u-1", {"question": "q", "top_k": 5, "paper_ids": ids_a, "filters": {}})
        k2 = make_cache_key("rag_query", "u-1", {"question": "q", "top_k": 5, "paper_ids": ids_b, "filters": {}})
        assert k1 == k2

    def test_different_user_different_key(self):
        payload = {"question": "q", "top_k": 5, "paper_ids": ["p1"], "filters": {}}
        k1 = make_cache_key("rag_query", "u-a", payload)
        k2 = make_cache_key("rag_query", "u-b", payload)
        assert k1 != k2

    def test_different_top_k_different_key(self):
        k1 = make_cache_key("rag_query", "u-1", {"question": "q", "top_k": 5, "paper_ids": ["p1"], "filters": {}})
        k2 = make_cache_key("rag_query", "u-1", {"question": "q", "top_k": 10, "paper_ids": ["p1"], "filters": {}})
        assert k1 != k2

    def test_different_filters_different_key(self):
        k1 = make_cache_key("rag_query", "u-1", {"question": "q", "top_k": 5, "paper_ids": ["p1"], "filters": {"source": "arxiv"}})
        k2 = make_cache_key("rag_query", "u-1", {"question": "q", "top_k": 5, "paper_ids": ["p1"], "filters": {"source": "s2"}})
        assert k1 != k2

    def test_no_query_in_key(self):
        key = make_cache_key("rag_query", "u-1", {"question": "secret medical query", "top_k": 5, "paper_ids": ["p1"], "filters": {}})
        assert "secret" not in key
        assert "medical" not in key


class TestRagQueryCache:
    """rag_query Redis 缓存行为。"""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached(self):
        """命中缓存时直接返回，不执行 RAG 逻辑。"""
        cached_result = _fake_rag_result("what is attention", 3)
        mock_rs = MagicMock()
        mock_rs.is_available = True
        mock_rs.get_json = AsyncMock(return_value={"result": cached_result})
        mock_rs.set_json = AsyncMock(return_value=True)

        with patch("tools.rag_query.redis_service", mock_rs), \
             patch("tools.rag_query.embed_text_async") as mock_embed, \
             patch("tools.rag_query._get_user_paper_ids", return_value={"p1"}):
            from tools.rag_query import handle_rag_query
            result = await handle_rag_query(
                {"question": "what is attention", "paper_id": "p1"}, user_id="u-1")

        assert result == cached_result
        mock_embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_then_write(self):
        """缓存未命中时执行 RAG 查询，成功后写缓存，TTL=600。"""
        mock_rs = MagicMock()
        mock_rs.is_available = True
        mock_rs.get_json = AsyncMock(return_value=None)
        mock_rs.set_json = AsyncMock(return_value=True)

        with patch("tools.rag_query.redis_service", mock_rs), \
             patch("tools.rag_query.embed_text_async", return_value=[0.1] * 384), \
             patch("tools.rag_query._get_user_paper_ids", return_value={"p1"}), \
             patch("tools.rag_query._milvus_search", new_callable=AsyncMock, side_effect=Exception("Milvus unavailable")), \
             patch("tools.rag_query._brute_force_search", return_value=([
                 {"score": 0.9, "chunk_id": "c1", "text": "hello", "section": "Abs",
                  "paper_id": "p1", "title": "T"}
             ], 10)):
            from tools.rag_query import handle_rag_query
            result = await handle_rag_query(
                {"question": "attention", "paper_id": "p1"}, user_id="u-1")

        parsed = json.loads(result)
        assert parsed["ok"] is True

        mock_rs.set_json.assert_awaited_once()
        _, kwargs = mock_rs.set_json.call_args
        assert kwargs.get("ttl") == 600

    @pytest.mark.asyncio
    async def test_redis_unavailable_no_cache(self):
        """Redis 不可用时行为不变。"""
        mock_rs = MagicMock()
        mock_rs.is_available = False

        with patch("tools.rag_query.redis_service", mock_rs), \
             patch("tools.rag_query.embed_text_async", return_value=[0.1] * 384), \
             patch("tools.rag_query._get_user_paper_ids", return_value={"p1"}), \
             patch("tools.rag_query._milvus_search", new_callable=AsyncMock, side_effect=Exception("Milvus unavailable")), \
             patch("tools.rag_query._brute_force_search", return_value=([
                 {"score": 0.9, "chunk_id": "c1", "text": "hello", "section": "Abs",
                  "paper_id": "p1", "title": "T"}
             ], 10)):
            from tools.rag_query import handle_rag_query
            result = await handle_rag_query(
                {"question": "attention", "paper_id": "p1"}, user_id="u-1")

        parsed = json.loads(result)
        assert parsed["ok"] is True
        mock_rs.get_json.assert_not_called()
        mock_rs.set_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_redis_none_no_cache(self):
        """redis_service 为 None 时行为不变。"""
        with patch("tools.rag_query.redis_service", None), \
             patch("tools.rag_query.embed_text_async", return_value=[0.1] * 384), \
             patch("tools.rag_query._get_user_paper_ids", return_value={"p1"}), \
             patch("tools.rag_query._milvus_search", new_callable=AsyncMock, side_effect=Exception("Milvus unavailable")), \
             patch("tools.rag_query._brute_force_search", return_value=([
                 {"score": 0.9, "chunk_id": "c1", "text": "hello", "section": "Abs",
                  "paper_id": "p1", "title": "T"}
             ], 10)):
            from tools.rag_query import handle_rag_query
            result = await handle_rag_query(
                {"question": "attention", "paper_id": "p1"}, user_id="u-1")

        parsed = json.loads(result)
        assert parsed["ok"] is True

    @pytest.mark.asyncio
    async def test_no_user_id_no_cache(self):
        """No user_id means no cache."""
        mock_rs = MagicMock()
        mock_rs.is_available = True

        with patch("tools.rag_query.redis_service", mock_rs), \
             patch("tools.rag_query.ALLOW_UNSCOPED", True), \
             patch("tools.rag_query.embed_text_async", return_value=[0.1] * 384), \
             patch("tools.rag_query._milvus_search", new_callable=AsyncMock, side_effect=Exception("Milvus unavailable")), \
             patch("tools.rag_query.get_embeddings_by_paper_ids", new_callable=AsyncMock, return_value=[
                 {"chunk_id": "c1", "text": "hello", "section": "Abs",
                  "paper_id": "p1", "title": "T", "vec": np.array([0.1] * 384, dtype=np.float32)}
             ]), \
             patch("tools.rag_query.get_connection"):
            from tools.rag_query import handle_rag_query
            result = await handle_rag_query(
                {"question": "attention", "paper_id": "p1"}, user_id=None)

        parsed = json.loads(result)
        assert parsed["ok"] is True
        mock_rs.get_json.assert_not_called()
        mock_rs.set_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_unscoped_query_no_cache(self):
        """全库搜索（无 paper_id/paper_ids/filter）不缓存。"""
        mock_rs = MagicMock()
        mock_rs.is_available = True

        with patch("tools.rag_query.redis_service", mock_rs), \
             patch("tools.rag_query.embed_text_async", return_value=[0.1] * 384), \
             patch("tools.rag_query._get_user_paper_ids", return_value={"p1"}), \
             patch("tools.rag_query._milvus_search", new_callable=AsyncMock, side_effect=Exception("Milvus unavailable")), \
             patch("tools.rag_query._brute_force_search", return_value=([
                 {"score": 0.9, "chunk_id": "c1", "text": "hello", "section": "Abs",
                  "paper_id": "p1", "title": "T"}
             ], 10)):
            from tools.rag_query import handle_rag_query
            result = await handle_rag_query({"question": "attention"}, user_id="u-1")

        parsed = json.loads(result)
        assert parsed["ok"] is True
        mock_rs.get_json.assert_not_called()
        mock_rs.set_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_result_not_cached(self):
        """RAG 返回错误时不写缓存。"""
        mock_rs = MagicMock()
        mock_rs.is_available = True
        mock_rs.get_json = AsyncMock(return_value=None)
        mock_rs.set_json = AsyncMock(return_value=True)

        with patch("tools.rag_query.redis_service", mock_rs), \
             patch("tools.rag_query.embed_text_async", side_effect=Exception("embed failed")):
            from tools.rag_query import handle_rag_query
            result = await handle_rag_query(
                {"question": "attention", "paper_id": "p1"}, user_id="u-1")

        parsed = json.loads(result)
        assert parsed["ok"] is False
        mock_rs.set_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_oversized_result_not_cached(self):
        """cacheable_size 返回 False 时不缓存。"""
        mock_rs = MagicMock()
        mock_rs.is_available = True
        mock_rs.get_json = AsyncMock(return_value=None)
        mock_rs.set_json = AsyncMock(return_value=True)

        with patch("tools.rag_query.redis_service", mock_rs), \
             patch("tools.rag_query.embed_text_async", return_value=[0.1] * 384), \
             patch("tools.rag_query._get_user_paper_ids", return_value={"p1"}), \
             patch("tools.rag_query._milvus_search", new_callable=AsyncMock, side_effect=Exception("Milvus unavailable")), \
             patch("tools.rag_query._brute_force_search", return_value=([
                 {"score": 0.9, "chunk_id": "c1", "text": "hello", "section": "Abs",
                  "paper_id": "p1", "title": "T"}
             ], 10)), \
             patch("tools.rag_query.cacheable_size", return_value=False):
            from tools.rag_query import handle_rag_query
            result = await handle_rag_query(
                {"question": "attention", "paper_id": "p1"}, user_id="u-1")

        parsed = json.loads(result)
        assert parsed["ok"] is True
        mock_rs.set_json.assert_not_called()

    def test_cross_user_isolation(self):
        """user A 和 user B 相同 query/paper_ids 生成不同 key。"""
        payload = {"question": "attention", "top_k": 5, "paper_ids": ["p1"], "filters": {}}
        k1 = make_cache_key("rag_query", "user-a", payload)
        k2 = make_cache_key("rag_query", "user-b", payload)
        assert k1 is not None and k2 is not None
        assert k1 != k2

    @pytest.mark.asyncio
    async def test_cache_hit_does_not_affect_other_user(self):
        """user A 的 cache hit 不影响 user B（不同 key）。"""
        cached_a = _fake_rag_result("q", 2)
        mock_rs = MagicMock()
        mock_rs.is_available = True
        # 只有 user A 的 key 有缓存
        async def selective_get(key):
            if "user-a" in key:
                return {"result": cached_a}
            return None
        mock_rs.get_json = AsyncMock(side_effect=selective_get)
        mock_rs.set_json = AsyncMock(return_value=True)

        with patch("tools.rag_query.redis_service", mock_rs), \
             patch("tools.rag_query.embed_text_async", return_value=[0.1] * 384), \
             patch("tools.rag_query._get_user_paper_ids", return_value={"p1"}), \
             patch("tools.rag_query._milvus_search", new_callable=AsyncMock, side_effect=Exception("Milvus unavailable")), \
             patch("tools.rag_query._brute_force_search", return_value=([
                 {"score": 0.8, "chunk_id": "c1", "text": "b", "section": "Abs",
                  "paper_id": "p1", "title": "T"}
             ], 5)):
            from tools.rag_query import handle_rag_query
            # user A: cache hit
            result_a = await handle_rag_query(
                {"question": "q", "paper_id": "p1"}, user_id="user-a")
            # user B: cache miss → 执行搜索
            result_b = await handle_rag_query(
                {"question": "q", "paper_id": "p1"}, user_id="user-b")

        assert result_a == cached_a
        assert result_b != cached_a
        parsed_b = json.loads(result_b)
        assert parsed_b["ok"] is True
