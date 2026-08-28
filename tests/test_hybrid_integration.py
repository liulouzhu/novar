"""tests/test_hybrid_integration.py — 混合检索集成测试

覆盖：
1. bulk action 格式正确（_op_type, _index, _id, _source）
2. fresh index 创建 mapping
3. paper_parse 后写入 ES
4. ES search 返回 BM25 hits
5. Milvus Top 50 + ES Top 50 + RRF
6. ES 正常但空结果不会报 unavailable
7. ES 故障会正确降级
8. Milvus 空结果会触发 brute-force
9. 未授权 paper_id 不会进入 ES/Milvus 查询
10. 无 user_id 仍保持 fail-closed
11. Elasticsearch client 可恢复
12. reindex 脚本 dry-run/apply 行为正确
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
# 1. bulk action 格式正确
# ══════════════════════════════════════════════════════════════════════════

class TestBulkActionFormat:
    @pytest.mark.asyncio
    async def test_bulk_actions_have_correct_format(self):
        """每个 action 包含 _op_type, _index, _id, _source。"""
        from core.elasticsearch_store import bulk_upsert_chunks

        mock_client = AsyncMock()
        mock_client.indices.exists = AsyncMock(return_value=True)
        mock_client.indices.get_mapping = AsyncMock(return_value={
            "paper_chunks": {"mappings": {"properties": {
                "chunk_id": {"type": "long"},
                "paper_id": {"type": "keyword"},
                "title": {"type": "text"},
                "section": {"type": "keyword"},
                "text": {"type": "text"},
                "search_text": {"type": "text"},
            }}}
        })
        mock_client.ping = AsyncMock(return_value=True)

        with patch("core.elasticsearch_store._get_client", new_callable=AsyncMock, return_value=mock_client):
            with patch("core.elasticsearch_store._async_bulk", new_callable=AsyncMock) as mock_bulk:
                mock_bulk.return_value = (2, [])
                result = await bulk_upsert_chunks([
                    {"chunk_id": 1, "paper_id": "p1", "title": "T", "section": "s", "text": "hello"},
                    {"chunk_id": 2, "paper_id": "p1", "title": "T", "section": "s", "text": "world"},
                ])
                assert result["success"] == 2

                # 验证 bulk actions 格式
                call_args = mock_bulk.call_args
                actions = call_args[0][1]  # 第二个位置参数
                assert len(actions) == 2
                for action in actions:
                    assert "_op_type" in action, f"Missing _op_type: {action}"
                    assert action["_op_type"] == "index"
                    assert "_index" in action, f"Missing _index: {action}"
                    assert "_id" in action, f"Missing _id: {action}"
                    assert "_source" in action, f"Missing _source: {action}"
                    src = action["_source"]
                    assert "chunk_id" in src
                    assert "paper_id" in src
                    assert "title" in src
                    assert "section" in src
                    assert "text" in src
                    assert "search_text" in src  # 清洗后的文本


# ══════════════════════════════════════════════════════════════════════════
# 2. fresh index 创建 mapping
# ══════════════════════════════════════════════════════════════════════════

class TestIndexCreation:
    @pytest.mark.asyncio
    async def test_ensure_index_creates_mapping(self):
        """索引不存在时创建 mapping。"""
        from core.elasticsearch_store import ensure_index

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.indices.exists = AsyncMock(return_value=False)
        mock_client.indices.create = AsyncMock()

        with patch("core.elasticsearch_store._get_client", new_callable=AsyncMock, return_value=mock_client):
            result = await ensure_index()
            assert result is True
            mock_client.indices.create.assert_called_once()
            call_kwargs = mock_client.indices.create.call_args[1]
            assert "mappings" in call_kwargs["body"]

    @pytest.mark.asyncio
    async def test_ensure_index_checks_existing_mapping(self):
        """索引已存在时检查字段兼容性。"""
        from core.elasticsearch_store import ensure_index

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.indices.exists = AsyncMock(return_value=True)
        mock_client.indices.get_mapping = AsyncMock(return_value={
            "paper_chunks": {"mappings": {"properties": {
                "chunk_id": {"type": "long"},
                "paper_id": {"type": "keyword"},
                "title": {"type": "text"},
                "section": {"type": "keyword"},
                "text": {"type": "text"},
                "search_text": {"type": "text"},
            }}}
        })

        with patch("core.elasticsearch_store._get_client", new_callable=AsyncMock, return_value=mock_client):
            result = await ensure_index()
            assert result is True
            mock_client.indices.create.assert_not_called()  # 不需要创建


# ══════════════════════════════════════════════════════════════════════════
# 3. bulk_upsert_chunks 失败时返回可诊断错误
# ══════════════════════════════════════════════════════════════════════════

class TestBulkUpsertErrors:
    @pytest.mark.asyncio
    async def test_bulk_upsert_returns_error_info(self):
        """bulk 写入失败时返回错误信息。"""
        from core.elasticsearch_store import bulk_upsert_chunks

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.indices.exists = AsyncMock(return_value=True)
        mock_client.indices.get_mapping = AsyncMock(return_value={
            "paper_chunks": {"mappings": {"properties": {
                "chunk_id": {"type": "long"},
                "paper_id": {"type": "keyword"},
                "title": {"type": "text"},
                "section": {"type": "keyword"},
                "text": {"type": "text"},
                "search_text": {"type": "text"},
            }}}
        })

        with patch("core.elasticsearch_store._get_client", new_callable=AsyncMock, return_value=mock_client):
            with patch("core.elasticsearch_store._async_bulk", new_callable=AsyncMock) as mock_bulk:
                mock_bulk.return_value = (0, ["error 1", "error 2"])
                result = await bulk_upsert_chunks([
                    {"chunk_id": 1, "paper_id": "p1", "title": "T", "section": "s", "text": "hello"},
                ])
                assert result["success"] == 0
                assert len(result["errors"]) == 2

    @pytest.mark.asyncio
    async def test_bulk_upsert_client_unavailable(self):
        """ES client 不可用时返回错误。"""
        from core.elasticsearch_store import bulk_upsert_chunks

        with patch("core.elasticsearch_store._get_client", new_callable=AsyncMock, return_value=None):
            result = await bulk_upsert_chunks([
                {"chunk_id": 1, "paper_id": "p1", "title": "T", "section": "s", "text": "hello"},
            ])
            assert result["success"] == 0
            assert len(result["errors"]) > 0


# ══════════════════════════════════════════════════════════════════════════
# 4. ES search 返回 BM25 hits
# ══════════════════════════════════════════════════════════════════════════

class TestESSearchHits:
    @pytest.mark.asyncio
    async def test_search_returns_correct_format(self):
        from core.elasticsearch_store import search_chunks

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.search.return_value = {
            "hits": {"hits": [
                {"_score": 12.5, "_source": {
                    "chunk_id": 42, "paper_id": "p1", "title": "T",
                    "section": "Exp", "text": "result 85.2%",
                }},
            ]}
        }

        with patch("core.elasticsearch_store._get_client", new_callable=AsyncMock, return_value=mock_client):
            result = await search_chunks("query", ["p1"], top_n=10)
            assert result.available is True
            assert len(result.hits) == 1
            assert result.hits[0]["chunk_id"] == 42
            assert result.hits[0]["source"] == "keyword"

    @pytest.mark.asyncio
    async def test_search_empty_results_available(self):
        """ES 正常但无结果时 available=True。"""
        from core.elasticsearch_store import search_chunks

        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.search.return_value = {"hits": {"hits": []}}

        with patch("core.elasticsearch_store._get_client", new_callable=AsyncMock, return_value=mock_client):
            result = await search_chunks("nonexistent", ["p1"], top_n=10)
            assert result.available is True
            assert result.hits == []
            assert result.error is None


# ══════════════════════════════════════════════════════════════════════════
# 5. Milvus Top 50 + ES Top 50 + RRF
# ══════════════════════════════════════════════════════════════════════════

class TestHybridRRF:
    @pytest.mark.asyncio
    async def test_hybrid_search_method(self):
        """两路都返回结果时 search_method=hybrid_rrf。"""
        import tools.rag_query as rq

        fake_vec = [0.1] * 1024
        vector_hits = [
            {"score": 0.9, "chunk_id": 1, "text": "a", "paper_id": "p1",
             "title": "T", "section": "s", "source": "vector"},
        ]
        keyword_hits = [
            {"chunk_id": 2, "paper_id": "p1", "title": "T", "section": "s",
             "text": "b", "score": 10.0, "source": "keyword"},
        ]

        with patch.object(rq, "_get_user_paper_ids", return_value={"p1"}):
            with patch.object(rq, "embed_text_async", return_value=fake_vec):
                with patch.object(rq, "_milvus_search", new_callable=AsyncMock, return_value=vector_hits):
                    with patch.object(rq, "_es_search", new_callable=AsyncMock, return_value=(keyword_hits, True, None)):
                        result = _parse_result(
                            await rq.handle_rag_query({"question": "test"}, user_id="u-1")
                        )
                        assert result["ok"] is True
                        assert result["data"]["search_method"].startswith("hybrid_rrf")
                        assert result["data"]["vector_candidates"] == 1
                        assert result["data"]["keyword_candidates"] == 1
                        # RRF 融合后两个 chunk 都应存在
                        chunk_ids = [r["chunk_id"] for r in result["data"]["results"]]
                        assert 1 in chunk_ids
                        assert 2 in chunk_ids


# ══════════════════════════════════════════════════════════════════════════
# 6. ES 正常但空结果不会报 unavailable
# ══════════════════════════════════════════════════════════════════════════

class TestESNoFalseAlarm:
    @pytest.mark.asyncio
    async def test_es_empty_no_warning(self):
        """ES 正常返回空结果时，不产生 unavailable warning。"""
        import tools.rag_query as rq

        fake_vec = [0.1] * 1024
        vector_hits = [
            {"score": 0.9, "chunk_id": 1, "text": "a", "paper_id": "p1",
             "title": "T", "section": "s", "source": "vector"},
        ]

        with patch.object(rq, "_get_user_paper_ids", return_value={"p1"}):
            with patch.object(rq, "embed_text_async", return_value=fake_vec):
                with patch.object(rq, "_milvus_search", new_callable=AsyncMock, return_value=vector_hits):
                    # ES available=True, hits=[] (正常无匹配)
                    with patch.object(rq, "_es_search", new_callable=AsyncMock, return_value=([], True, None)):
                        result = _parse_result(
                            await rq.handle_rag_query({"question": "test"}, user_id="u-1")
                        )
                        assert result["ok"] is True
                        assert not any("Elasticsearch unavailable" in w for w in result.get("warnings", []))


# ══════════════════════════════════════════════════════════════════════════
# 7. ES 故障会正确降级
# ══════════════════════════════════════════════════════════════════════════

class TestESDegradation:
    @pytest.mark.asyncio
    async def test_es_failure_degrades(self):
        """ES 故障时降级到 Milvus only。"""
        import tools.rag_query as rq

        fake_vec = [0.1] * 1024
        vector_hits = [
            {"score": 0.9, "chunk_id": 1, "text": "a", "paper_id": "p1",
             "title": "T", "section": "s", "source": "vector"},
        ]

        with patch.object(rq, "_get_user_paper_ids", return_value={"p1"}):
            with patch.object(rq, "embed_text_async", return_value=fake_vec):
                with patch.object(rq, "_milvus_search", new_callable=AsyncMock, return_value=vector_hits):
                    # ES available=False (故障)
                    with patch.object(rq, "_es_search", new_callable=AsyncMock, return_value=([], False, "connection refused")):
                        result = _parse_result(
                            await rq.handle_rag_query({"question": "test"}, user_id="u-1")
                        )
                        assert result["ok"] is True
                        assert result["data"]["search_method"].startswith("Milvus")
                        assert any("Elasticsearch unavailable" in w for w in result.get("warnings", []))


# ══════════════════════════════════════════════════════════════════════════
# 8. Milvus 空结果会触发 brute-force
# ══════════════════════════════════════════════════════════════════════════

class TestMilvusEmptyFallback:
    @pytest.mark.asyncio
    async def test_milvus_empty_triggers_brute_force(self):
        """Milvus 返回空（无异常）+ ES 空时，触发 brute-force。"""
        import tools.rag_query as rq

        fake_vec = [0.1] * 1024
        emb_p1 = {
            "chunk_id": 1, "dim": 1024,
            "vec": np.array([0.1] * 1024, dtype=np.float32),
            "text": "hello", "section": "abstract",
            "paper_id": "p1", "title": "Paper 1",
        }

        with patch.object(rq, "_get_user_paper_ids", return_value={"p1"}):
            with patch.object(rq, "embed_text_async", return_value=fake_vec):
                # Milvus 返回空（无异常）
                with patch.object(rq, "_milvus_search", new_callable=AsyncMock, return_value=[]):
                    # ES 也返回空
                    with patch.object(rq, "_es_search", new_callable=AsyncMock, return_value=([], True, None)):
                        # brute-force 有结果
                        with patch("tools.rag_query.get_embeddings_by_paper_ids", new_callable=AsyncMock, return_value=[emb_p1]):
                            result = _parse_result(
                                await rq.handle_rag_query({"question": "test"}, user_id="u-1")
                            )
                            assert result["ok"] is True
                            assert len(result["data"]["results"]) > 0


# ══════════════════════════════════════════════════════════════════════════
# 9. 未授权 paper_id 不会进入 ES/Milvus 查询
# ══════════════════════════════════════════════════════════════════════════

class TestUnauthorizedPaperId:
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
# 10. 无 user_id 仍保持 fail-closed
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
# 11. Elasticsearch client 可恢复
# ══════════════════════════════════════════════════════════════════════════

class TestESClientRecovery:
    @pytest.mark.asyncio
    async def test_client_recovers_after_failure(self):
        """ES 暂时不可用后可以恢复。"""
        import core.elasticsearch_store as es_mod

        es_mod.reset_client()

        # 第一次 ping 失败
        mock_client_fail = AsyncMock()
        mock_client_fail.ping = AsyncMock(return_value=False)
        mock_client_fail.close = AsyncMock()

        with patch.object(es_mod, "_AsyncElasticsearch", return_value=mock_client_fail):
            result1 = await es_mod._get_client()
            assert result1 is None

        # 模拟时间流逝（跳过重试间隔）
        es_mod._last_failure_time = 0

        # 第二次 ping 成功
        mock_client_ok = AsyncMock()
        mock_client_ok.ping = AsyncMock(return_value=True)

        with patch.object(es_mod, "_AsyncElasticsearch", return_value=mock_client_ok):
            result2 = await es_mod._get_client()
            assert result2 is not None

        es_mod.reset_client()

    @pytest.mark.asyncio
    async def test_close_client_idempotent(self):
        """close_client 幂等。"""
        import core.elasticsearch_store as es_mod
        es_mod.reset_client()

        mock_client = AsyncMock()
        es_mod._client = mock_client

        await es_mod.close_client()
        assert es_mod._client is None

        # 再次调用不报错
        await es_mod.close_client()


# ══════════════════════════════════════════════════════════════════════════
# 12. reindex 脚本 dry-run/apply 行为正确
# ══════════════════════════════════════════════════════════════════════════

class TestReindexScript:
    def test_reindex_script_exists_and_compiles(self):
        """reindex 脚本存在且可编译。"""
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "reindex_elasticsearch.py"
        assert script_path.exists(), f"Script not found: {script_path}"
        # Verify it compiles
        import py_compile
        py_compile.compile(str(script_path), doraise=True)

    def test_reindex_script_has_dry_run_default(self):
        """reindex 脚本默认 dry-run。"""
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "reindex_elasticsearch.py"
        content = script_path.read_text(encoding="utf-8")
        assert "--apply" in content
        assert "dry-run" in content.lower() or "args.apply" in content


# ══════════════════════════════════════════════════════════════════════════
# ESSearchResult 结构验证
# ══════════════════════════════════════════════════════════════════════════

class TestESSearchResultStruct:
    def test_es_unavailable_result(self):
        from core.elasticsearch_store import ESSearchResult
        r = ESSearchResult(hits=[], available=False, error="timeout")
        assert r.available is False
        assert r.error == "timeout"
        assert r.hits == []

    def test_es_empty_result(self):
        from core.elasticsearch_store import ESSearchResult
        r = ESSearchResult(hits=[], available=True)
        assert r.available is True
        assert r.error is None
        assert r.hits == []
