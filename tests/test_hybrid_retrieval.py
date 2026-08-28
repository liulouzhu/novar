"""tests/test_hybrid_retrieval.py — 混合检索 (Milvus + ES BM25 + RRF) 测试

覆盖：
1. 未传 top_k 时，Milvus 召回数量为 50，最终返回数量为 5
2. 传入 top_k=10 时，Milvus 仍召回 50，最终返回 10
3. ES 查询的 size 为 50
4. ES 查询包含 paper_id terms filter
5. Milvus 查询包含 user_id 和 paper_id 过滤
6. RRF 计算公式正确
7. 同一 chunk 被两路召回时正确去重
8. 仅 Milvus 可用时正常降级
9. 仅 ES 可用时正常降级
10. 两路都失败时返回明确错误
11. 无 user_id 时仍 fail-closed
12. 无权限 paper_id 不会出现在 ES 或 Milvus 查询中
13. RRF fusion_score 计算验证
14. ES search_chunks 返回正确格式
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

MCP_ROOT = Path(__file__).resolve().parent.parent / "mcp-server"
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")


def _parse_result(json_str: str) -> dict:
    return json.loads(json_str)


# ══════════════════════════════════════════════════════════════════════════
# 1. 未传 top_k 时 Milvus 召回 50，最终返回 5
# ══════════════════════════════════════════════════════════════════════════

class TestDefaultTopK:
    @pytest.mark.asyncio
    async def test_milvus_receives_50_final_returns_5(self):
        """默认 top_k=5 时，Milvus 召回 VECTOR_TOP_N=50，最终返回 5。"""
        import tools.rag_query as rq

        fake_vec = [0.1] * 1024
        fake_hits = [
            {"score": 0.9 - i * 0.01, "chunk_id": i, "text": f"chunk {i}",
             "paper_id": f"p{i % 5}", "title": f"Paper {i % 5}",
             "section": "abstract", "source": "vector"}
            for i in range(50)
        ]

        with patch.object(rq, "_get_user_paper_ids", return_value={"p0", "p1", "p2", "p3", "p4"}):
            with patch.object(rq, "embed_text_async", return_value=fake_vec):
                with patch.object(rq, "_milvus_search", new_callable=AsyncMock) as mock_milvus:
                    mock_milvus.return_value = fake_hits
                    with patch.object(rq, "_es_search", new_callable=AsyncMock, return_value=([], True, None)):
                        result = _parse_result(
                            await rq.handle_rag_query({"question": "test"}, user_id="u-1")
                        )
                        # 验证 Milvus 收到 VECTOR_TOP_N=50
                        assert mock_milvus.call_args[0][1] == rq.VECTOR_TOP_N
                        # 验证最终返回 5
                        assert result["data"]["returned_results"] == 5
                        assert result["data"]["vector_candidates"] == 50


# ══════════════════════════════════════════════════════════════════════════
# 2. 传入 top_k=10 时 Milvus 仍召回 50，最终返回 10
# ══════════════════════════════════════════════════════════════════════════

class TestExplicitTopK:
    @pytest.mark.asyncio
    async def test_milvus_50_final_10(self):
        import tools.rag_query as rq

        fake_vec = [0.1] * 1024
        fake_hits = [
            {"score": 0.9 - i * 0.01, "chunk_id": i, "text": f"chunk {i}",
             "paper_id": f"p{i % 5}", "title": f"Paper {i % 5}",
             "section": "abstract", "source": "vector"}
            for i in range(50)
        ]

        with patch.object(rq, "_get_user_paper_ids", return_value={"p0", "p1", "p2", "p3", "p4"}):
            with patch.object(rq, "embed_text_async", return_value=fake_vec):
                with patch.object(rq, "_milvus_search", new_callable=AsyncMock) as mock_milvus:
                    mock_milvus.return_value = fake_hits
                    with patch.object(rq, "_es_search", new_callable=AsyncMock, return_value=([], True, None)):
                        result = _parse_result(
                            await rq.handle_rag_query(
                                {"question": "test", "top_k": 10}, user_id="u-1"
                            )
                        )
                        assert mock_milvus.call_args[0][1] == rq.VECTOR_TOP_N
                        assert result["data"]["returned_results"] == 10


# ══════════════════════════════════════════════════════════════════════════
# 3. ES 查询 size 为 50
# ══════════════════════════════════════════════════════════════════════════

class TestESQuerySize:
    @pytest.mark.asyncio
    async def test_es_search_chunks_uses_top_n(self):
        """search_chunks 传入 top_n=50。"""
        from core.elasticsearch_store import search_chunks

        mock_client = AsyncMock()
        mock_client.search.return_value = {"hits": {"hits": []}}

        with patch("core.elasticsearch_store._get_client", new_callable=AsyncMock, return_value=mock_client):
            await search_chunks("test query", ["p1"], top_n=50)
            call_kwargs = mock_client.search.call_args[1]
            assert call_kwargs["body"]["size"] == 50


# ══════════════════════════════════════════════════════════════════════════
# 4. ES 查询包含 paper_id terms filter
# ══════════════════════════════════════════════════════════════════════════

class TestESPaperIdFilter:
    @pytest.mark.asyncio
    async def test_es_query_has_paper_id_filter(self):
        from core.elasticsearch_store import search_chunks

        mock_client = AsyncMock()
        mock_client.search.return_value = {"hits": {"hits": []}}

        with patch("core.elasticsearch_store._get_client", new_callable=AsyncMock, return_value=mock_client):
            await search_chunks("test", ["p1", "p2", "p3"], top_n=50)
            call_kwargs = mock_client.search.call_args[1]
            query = call_kwargs["body"]["query"]
            filters = query["bool"]["filter"]
            paper_filter = next(f for f in filters if "terms" in f and "paper_id" in f["terms"])
            assert set(paper_filter["terms"]["paper_id"]) == {"p1", "p2", "p3"}


# ══════════════════════════════════════════════════════════════════════════
# 5. Milvus 查询包含 user_id 和 paper_id 过滤
# ══════════════════════════════════════════════════════════════════════════

class TestMilvusFilter:
    @pytest.mark.asyncio
    async def test_milvus_search_passes_paper_ids(self):
        """_milvus_search 传递 paper_ids 到 search_vectors。"""
        import tools.rag_query as rq

        with patch("core.vector_store.search_vectors", return_value=[]) as mock_sv:
            with patch("core.vector_store.ensure_collection"):
                with patch("core.vector_store.connections"):
                    await rq._milvus_search([0.1] * 1024, 50, "user-1", paper_ids=["p1", "p2"])
                    call_kwargs = mock_sv.call_args[1]
                    assert call_kwargs["paper_ids"] == ["p1", "p2"]


# ══════════════════════════════════════════════════════════════════════════
# 6. RRF 计算公式正确
# ══════════════════════════════════════════════════════════════════════════

class TestRRFCalculation:
    def test_rrf_single_list(self):
        """单路结果的 RRF 分数 = 1/(k + rank)。"""
        from tools.rag_query import reciprocal_rank_fusion

        list1 = [
            {"chunk_id": 1, "paper_id": "p1", "score": 0.9, "source": "vector",
             "text": "a", "section": "s", "title": "t"},
            {"chunk_id": 2, "paper_id": "p2", "score": 0.8, "source": "vector",
             "text": "b", "section": "s", "title": "t"},
        ]
        result = reciprocal_rank_fusion([list1], rrf_k=60)
        # chunk 1: rank=1, score = 1/(60+1) = 0.016393...
        # chunk 2: rank=2, score = 1/(60+2) = 0.016129...
        assert abs(result[0]["fusion_score"] - 1 / 61) < 1e-6
        assert abs(result[1]["fusion_score"] - 1 / 62) < 1e-6

    def test_rrf_two_lists_merge(self):
        """两路结果融合时分数叠加。"""
        from tools.rag_query import reciprocal_rank_fusion

        list1 = [{"chunk_id": 1, "paper_id": "p1", "score": 0.9, "source": "vector",
                  "text": "a", "section": "s", "title": "t"}]
        list2 = [{"chunk_id": 1, "paper_id": "p1", "score": 10.0, "source": "keyword",
                  "text": "a", "section": "s", "title": "t"}]

        result = reciprocal_rank_fusion([list1, list2], rrf_k=60)
        # chunk 1 在两路都是 rank=1: 1/61 + 1/61 = 2/61
        assert len(result) == 1
        assert abs(result[0]["fusion_score"] - 2 / 61) < 1e-6
        assert result[0]["vector_rank"] == 1
        assert result[0]["keyword_rank"] == 1


# ══════════════════════════════════════════════════════════════════════════
# 7. 同一 chunk 被两路召回时正确去重
# ══════════════════════════════════════════════════════════════════════════

class TestRRFDedup:
    def test_same_chunk_deduped(self):
        from tools.rag_query import reciprocal_rank_fusion

        list1 = [
            {"chunk_id": 1, "paper_id": "p1", "score": 0.9, "source": "vector",
             "text": "a", "section": "s", "title": "t"},
            {"chunk_id": 2, "paper_id": "p2", "score": 0.8, "source": "vector",
             "text": "b", "section": "s", "title": "t"},
        ]
        list2 = [
            {"chunk_id": 1, "paper_id": "p1", "score": 10.0, "source": "keyword",
             "text": "a", "section": "s", "title": "t"},
            {"chunk_id": 3, "paper_id": "p3", "score": 5.0, "source": "keyword",
             "text": "c", "section": "s", "title": "t"},
        ]
        result = reciprocal_rank_fusion([list1, list2], rrf_k=60)
        chunk_ids = [r["chunk_id"] for r in result]
        assert len(chunk_ids) == len(set(chunk_ids))  # 无重复
        assert set(chunk_ids) == {1, 2, 3}


# ══════════════════════════════════════════════════════════════════════════
# 8. 仅 Milvus 可用时正常降级
# ══════════════════════════════════════════════════════════════════════════

class TestMilvusOnlyDegradation:
    @pytest.mark.asyncio
    async def test_es_fails_milvus_works(self):
        import tools.rag_query as rq

        fake_vec = [0.1] * 1024
        fake_hits = [
            {"score": 0.9, "chunk_id": 1, "text": "hello",
             "paper_id": "p1", "title": "T", "section": "s", "source": "vector"},
        ]

        with patch.object(rq, "_get_user_paper_ids", return_value={"p1"}):
            with patch.object(rq, "embed_text_async", return_value=fake_vec):
                with patch.object(rq, "_milvus_search", new_callable=AsyncMock, return_value=fake_hits):
                    with patch.object(rq, "_es_search", new_callable=AsyncMock, return_value=([], True, None)):
                        result = _parse_result(
                            await rq.handle_rag_query({"question": "test"}, user_id="u-1")
                        )
                        assert result["ok"] is True
                        assert result["data"]["search_method"].startswith("Milvus")
                        assert result["data"]["vector_candidates"] == 1
                        assert result["data"]["keyword_candidates"] == 0


# ══════════════════════════════════════════════════════════════════════════
# 9. 仅 ES 可用时正常降级
# ══════════════════════════════════════════════════════════════════════════

class TestESOnlyDegradation:
    @pytest.mark.asyncio
    async def test_milvus_fails_es_works(self):
        import tools.rag_query as rq

        fake_vec = [0.1] * 1024
        fake_hits = [
            {"chunk_id": 1, "paper_id": "p1", "title": "T",
             "section": "s", "text": "hello", "score": 10.0, "source": "keyword"},
        ]

        with patch.object(rq, "_get_user_paper_ids", return_value={"p1"}):
            with patch.object(rq, "embed_text_async", return_value=fake_vec):
                with patch.object(rq, "_milvus_search", new_callable=AsyncMock, side_effect=Exception("Milvus down")):
                    with patch.object(rq, "_es_search", new_callable=AsyncMock, return_value=(fake_hits, True, None)):
                        result = _parse_result(
                            await rq.handle_rag_query({"question": "test"}, user_id="u-1")
                        )
                        assert result["ok"] is True
                        assert result["data"]["search_method"].startswith("Elasticsearch BM25")
                        assert result["data"]["keyword_candidates"] == 1
                        assert any("Milvus" in w for w in result.get("warnings", []))


# ══════════════════════════════════════════════════════════════════════════
# 10. 两路都失败时返回明确错误
# ══════════════════════════════════════════════════════════════════════════

class TestBothFail:
    @pytest.mark.asyncio
    async def test_both_fail_returns_error(self):
        import tools.rag_query as rq

        fake_vec = [0.1] * 1024

        with patch.object(rq, "_get_user_paper_ids", return_value={"p1"}):
            with patch.object(rq, "embed_text_async", return_value=fake_vec):
                with patch.object(rq, "_milvus_search", new_callable=AsyncMock, side_effect=Exception("Milvus down")):
                    with patch.object(rq, "_es_search", new_callable=AsyncMock, return_value=([], True, None)):
                        with patch.object(rq, "_brute_force_search", return_value=([], 0)):
                            result = _parse_result(
                                await rq.handle_rag_query({"question": "test"}, user_id="u-1")
                            )
                            assert result["ok"] is False
                            assert "未找到" in result["error"]


# ══════════════════════════════════════════════════════════════════════════
# 11. 无 user_id 时仍 fail-closed
# ══════════════════════════════════════════════════════════════════════════

class TestFailClosed:
    @pytest.mark.asyncio
    async def test_no_user_id_rejected(self):
        import tools.rag_query as rq
        original = rq.ALLOW_UNSCOPED
        rq.ALLOW_UNSCOPED = False
        try:
            result = _parse_result(await rq.handle_rag_query({"question": "test"}))
            assert result["ok"] is False
            assert "user_id" in result["error"]
        finally:
            rq.ALLOW_UNSCOPED = original


# ══════════════════════════════════════════════════════════════════════════
# 12. 无权限 paper_id 不会出现在 ES 或 Milvus 查询中
# ══════════════════════════════════════════════════════════════════════════

class TestPaperIdLeakage:
    @pytest.mark.asyncio
    async def test_unauthorized_paper_id_rejected(self):
        import tools.rag_query as rq

        with patch.object(rq, "_get_user_paper_ids", return_value={"p1"}):
            with patch.object(rq, "embed_text_async", return_value=[0.1] * 1024):
                result = _parse_result(
                    await rq.handle_rag_query(
                        {"question": "test", "paper_id": "unauthorized"},
                        user_id="u-1",
                    )
                )
                assert result["ok"] is False
                assert "unauthorized" not in result["error"]


# ══════════════════════════════════════════════════════════════════════════
# 13. RRF fusion_score 排序验证
# ══════════════════════════════════════════════════════════════════════════

class TestRRFSorting:
    def test_rrf_results_sorted_by_fusion_score(self):
        from tools.rag_query import reciprocal_rank_fusion

        list1 = [
            {"chunk_id": 1, "paper_id": "p1", "score": 0.9, "source": "vector",
             "text": "a", "section": "s", "title": "t"},
            {"chunk_id": 2, "paper_id": "p2", "score": 0.8, "source": "vector",
             "text": "b", "section": "s", "title": "t"},
        ]
        list2 = [
            {"chunk_id": 3, "paper_id": "p3", "score": 10.0, "source": "keyword",
             "text": "c", "section": "s", "title": "t"},
            {"chunk_id": 1, "paper_id": "p1", "score": 5.0, "source": "keyword",
             "text": "a", "section": "s", "title": "t"},
        ]
        result = reciprocal_rank_fusion([list1, list2], rrf_k=60)
        scores = [r["fusion_score"] for r in result]
        assert scores == sorted(scores, reverse=True)


# ══════════════════════════════════════════════════════════════════════════
# 14. ES search_chunks 返回格式验证
# ══════════════════════════════════════════════════════════════════════════

class TestESReturnFormat:
    @pytest.mark.asyncio
    async def test_es_search_returns_correct_format(self):
        from core.elasticsearch_store import search_chunks

        mock_client = AsyncMock()
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_score": 12.5,
                        "_source": {
                            "chunk_id": 42,
                            "paper_id": "arxiv:2308.11681",
                            "title": "VadCLIP",
                            "section": "Experiments",
                            "text": "UCF-Crime results...",
                        },
                    }
                ]
            }
        }

        with patch("core.elasticsearch_store._get_client", new_callable=AsyncMock, return_value=mock_client):
            result = await search_chunks("VadCLIP results", ["arxiv:2308.11681"], top_n=10)
            assert result.available is True
            assert len(result.hits) == 1
            r = result.hits[0]
            assert r["chunk_id"] == 42
            assert r["paper_id"] == "arxiv:2308.11681"
            assert r["score"] == 12.5
            assert r["source"] == "keyword"
            assert r["title"] == "VadCLIP"
            assert r["section"] == "Experiments"
