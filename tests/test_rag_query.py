"""tests/test_rag_query.py — RAG multi-user isolation tests.

Verifies fail-closed behavior:
  - No user_id → reject (unless RAG_ALLOW_UNSCOPED)
  - Auth query failure → reject (never degrade to full-library scan)
  - brute-force fallback uses scoped SQL, never get_all_embeddings()
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# mcp-server/ 不在 sys.path 中，需要手动加入
MCP_ROOT = Path(__file__).resolve().parent.parent / "mcp-server"
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")


# ── Helpers ──────────────────────────────────────────────────────────────

def _parse_result(json_str: str) -> dict:
    """解析工具返回的 JSON 字符串。"""
    return json.loads(json_str)


# ── _get_user_paper_ids ──────────────────────────────────────────────────

class TestGetUserPaperIds:
    """_get_user_paper_ids 必须在查询失败时抛异常，而不是返回 None。"""

    @pytest.mark.asyncio
    async def test_no_user_id_raises(self):
        from tools.rag_query import _get_user_paper_ids
        with pytest.raises(PermissionError):
            await _get_user_paper_ids("")

    @pytest.mark.asyncio
    async def test_no_user_id_none_raises(self):
        from tools.rag_query import _get_user_paper_ids
        with pytest.raises(PermissionError):
            await _get_user_paper_ids(None)

    @pytest.mark.asyncio
    async def test_db_query_failure_raises(self):
        """DB 连接/查询异常时必须抛出，不能静默返回 None。"""
        from tools.rag_query import _get_user_paper_ids
        with patch("web.backend.db.base.get_session_factory") as mock_factory:
            mock_factory.return_value.return_value.__aenter__ = AsyncMock(side_effect=RuntimeError("connection refused"))
            mock_factory.return_value.return_value.__aexit__ = AsyncMock(return_value=False)
            with pytest.raises(RuntimeError):
                await _get_user_paper_ids("00000000-0000-0000-0000-000000000001")

    @pytest.mark.asyncio
    async def test_returns_empty_set_when_no_papers(self):
        """用户没有论文时返回空集合（不是 None）。"""
        from tools.rag_query import _get_user_paper_ids
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        with patch("web.backend.db.base.get_session_factory") as mock_factory:
            mock_factory.return_value.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_factory.return_value.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await _get_user_paper_ids("00000000-0000-0000-0000-000000000001")
        assert result == set()


# ── _brute_force_search ──────────────────────────────────────────────────

class TestBruteForceSearch:
    """_brute_force_search 必须使用 scoped 查询，不得调用 get_all_embeddings。"""

    @pytest.mark.asyncio
    async def test_calls_scoped_query_not_all(self):
        """只调用 get_embeddings_by_paper_ids，不调用 get_all_embeddings。"""
        from tools.rag_query import _brute_force_search
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        with patch("tools.rag_query.get_embeddings_by_paper_ids", new_callable=AsyncMock, return_value=[]) as mock_scoped:
            query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            await _brute_force_search(query_vec, top_k=5, allowed_paper_ids={"p1"})
            mock_scoped.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_paper_ids_returns_empty(self):
        """allowed_paper_ids 为空时直接返回空，不查数据库。"""
        from tools.rag_query import _brute_force_search
        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results, total = await _brute_force_search(query_vec, top_k=5, allowed_paper_ids=set())
        assert results == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_returns_total_scanned(self):
        """返回值包含 total_scanned（scoped embeddings 数量）。"""
        fake_emb = {
            "chunk_id": 1, "dim": 3,
            "vec": np.array([1, 0, 0], dtype=np.float32),
            "text": "t", "section": "s", "paper_id": "p1", "title": "T",
        }
        with patch("tools.rag_query.get_embeddings_by_paper_ids", new_callable=AsyncMock, return_value=[fake_emb]):
            from tools.rag_query import _brute_force_search
            query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            results, total = await _brute_force_search(query_vec, top_k=5, allowed_paper_ids={"p1"})
            assert total == 1
            assert len(results) == 1


# ── handle_rag_query fail-closed ─────────────────────────────────────────

class TestHandleRagQueryFailClosed:
    """handle_rag_query 默认路径必须 fail-closed。"""

    @pytest.mark.asyncio
    async def test_no_user_id_fails_closed(self):
        """无 user_id 时默认返回错误。"""
        import tools.rag_query as rq
        original = rq.ALLOW_UNSCOPED
        rq.ALLOW_UNSCOPED = False
        try:
            result = _parse_result(await rq.handle_rag_query({"question": "test"}))
            assert result["ok"] is False
            assert "user_id" in result["error"]
        finally:
            rq.ALLOW_UNSCOPED = original

    @pytest.mark.asyncio
    async def test_auth_failure_fails_closed(self):
        """_get_user_paper_ids 抛异常时，不降级为全库检索。"""
        import tools.rag_query as rq
        with patch.object(rq, "_get_user_paper_ids", side_effect=RuntimeError("db down")):
            with patch.object(rq, "embed_text_async", return_value=[0.1, 0.2, 0.3]):
                result = _parse_result(
                    await rq.handle_rag_query({"question": "test"}, user_id="u-1")
                )
                assert result["ok"] is False
                assert "中止" in result["error"]

    @pytest.mark.asyncio
    async def test_no_papers_returns_fail(self):
        """用户没有论文时返回明确失败，不查全库。"""
        import tools.rag_query as rq
        with patch.object(rq, "_get_user_paper_ids", return_value=set()):
            with patch.object(rq, "embed_text_async", return_value=[0.1, 0.2, 0.3]):
                result = _parse_result(
                    await rq.handle_rag_query({"question": "test"}, user_id="u-1")
                )
                assert result["ok"] is False
                assert "论文" in result["error"]

    @pytest.mark.asyncio
    async def test_brute_force_never_calls_get_all_embeddings(self):
        """正常有 user_id 的 brute-force 路径不得调用 get_all_embeddings。"""
        import tools.rag_query as rq
        fake_vec = [1.0, 0.0, 0.0]
        fake_emb = {
            "chunk_id": 1, "dim": 3,
            "vec": np.array(fake_vec, dtype=np.float32),
            "text": "hello", "section": "abstract",
            "paper_id": "p1", "title": "Paper 1",
        }
        with patch.object(rq, "_get_user_paper_ids", return_value={"p1"}):
            with patch.object(rq, "embed_text_async", return_value=fake_vec):
                with patch.object(rq, "_milvus_search", new_callable=AsyncMock, side_effect=Exception("Milvus down")):
                    with patch("tools.rag_query.get_embeddings_by_paper_ids", new_callable=AsyncMock, return_value=[fake_emb]):
                        with patch("core.database.get_all_embeddings", new_callable=AsyncMock) as mock_all:
                            result = _parse_result(
                                await rq.handle_rag_query({"question": "test"}, user_id="u-1")
                            )
                            assert result["ok"] is True
                            mock_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_allow_unscoped_permits_no_user_id(self):
        """显式 RAG_ALLOW_UNSCOPED=true 时允许无 user_id 全库检索。"""
        import tools.rag_query as rq
        original = rq.ALLOW_UNSCOPED
        rq.ALLOW_UNSCOPED = True
        fake_vec = [1.0, 0.0, 0.0]
        fake_emb = {
            "chunk_id": 1, "dim": 3,
            "vec": np.array(fake_vec, dtype=np.float32),
            "text": "hello", "section": "abstract",
            "paper_id": "p1", "title": "Paper 1",
        }
        try:
            with patch.object(rq, "embed_text_async", return_value=fake_vec):
                with patch.object(rq, "_milvus_search", new_callable=AsyncMock, side_effect=Exception("Milvus down")):
                    with patch("core.database.get_all_embeddings", new_callable=AsyncMock, return_value=[fake_emb]):
                        with patch("tools.rag_query.get_connection") as mock_conn:
                            mock_conn.return_value.__aenter__ = AsyncMock()
                            mock_conn.return_value.__aexit__ = AsyncMock(return_value=False)
                            result = _parse_result(
                                await rq.handle_rag_query({"question": "test"})
                            )
                            assert result["ok"] is True
        finally:
            rq.ALLOW_UNSCOPED = original

    @pytest.mark.asyncio
    async def test_brute_force_scoped_to_user_papers(self):
        """brute-force 结果只包含用户有权访问的论文。"""
        import tools.rag_query as rq
        fake_vec = [1.0, 0.0, 0.0]
        emb_p1 = {
            "chunk_id": 1, "dim": 3,
            "vec": np.array(fake_vec, dtype=np.float32),
            "text": "paper1 chunk", "section": "abstract",
            "paper_id": "p1", "title": "Paper 1",
        }
        with patch.object(rq, "_get_user_paper_ids", return_value={"p1"}):
            with patch.object(rq, "embed_text_async", return_value=fake_vec):
                with patch.object(rq, "_milvus_search", new_callable=AsyncMock, side_effect=Exception("Milvus down")):
                    with patch(
                        "tools.rag_query.get_embeddings_by_paper_ids", new_callable=AsyncMock, return_value=[emb_p1]
                    ) as mock_scoped:
                        result = _parse_result(
                            await rq.handle_rag_query({"question": "test"}, user_id="u-1")
                        )
                        assert result["ok"] is True
                        for r in result["data"]["results"]:
                            assert r["paper_id"] == "p1"


# ── get_embeddings_by_paper_ids (database layer) ─────────────────────────

class TestGetEmbeddingsByPaperIds:
    """get_embeddings_by_paper_ids 只查询指定 paper_ids 的 embeddings。"""

    @pytest.mark.asyncio
    async def test_empty_paper_ids_returns_empty(self):
        """空 paper_ids 直接返回空列表。"""
        from core.database import get_embeddings_by_paper_ids
        result = await get_embeddings_by_paper_ids(MagicMock(), set())
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        from core.database import get_embeddings_by_paper_ids
        result = await get_embeddings_by_paper_ids(MagicMock(), [])
        assert result == []
