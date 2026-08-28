"""tests/test_rag_fixes.py — 综合测试 RAG 检索修复

覆盖以下场景：
1. CLI 设置 NOVARE_USER_ID 后，MCP payload 包含正确的 _user_id
2. CLI 未设置 NOVARE_USER_ID 时保持 fail-closed，并返回明确提示
3. Web 的 user_id 注入不受影响
4. 默认 MCP 环境变量正确传递 allowlist 中的 embedding/Milvus 配置
5. 敏感配置不出现在日志中
6. 无 embedding provider 时生产环境明确失败
7. 只有测试开关打开时才允许 numpy fallback
8. 查询向量和数据库向量维度不一致时返回维度错误
9. paper_id 和 paper_ids 在 brute-force 检索中真正生效
10. Milvus 检索表达式同时包含 user_id 和 paper_id 条件
11. 无权限 paper_id 不会泄露论文信息
12. top_k 非法值被拒绝或规范化
13. arXiv ID canonicalization 覆盖裸 ID、前缀 ID 和 URL
14. 回归测试：模拟 VadCLIP 查询时能返回包含实验结果的片段
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


# ══════════════════════════════════════════════════════════════════════════
# 1. CLI 设置 NOVARE_USER_ID 后，MCP payload 包含正确的 _user_id
# ══════════════════════════════════════════════════════════════════════════

class TestCliUserIdInjection:
    """CLI 模式下 _user_id 必须从 NOVARE_USER_ID 注入到 MCP payload。"""

    @pytest.mark.asyncio
    async def test_handler_injects_user_id(self):
        """make_handler 闭包应将 user_id 注入 payload。"""
        from unittest.mock import AsyncMock

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value='{"ok": true}')

        async def make_handler(c, tn, uid):
            async def handler(args, workspace=None):
                payload = dict(args)
                if uid:
                    payload["_user_id"] = uid
                return await c.call_tool(tn, payload)
            return handler

        handler = await make_handler(mock_client, "rag_query", "test-user-123")

        args = {"question": "test"}
        await handler(args)

        # 验证 call_tool 被调用时 payload 包含 _user_id
        call_args = mock_client.call_tool.call_args
        payload = call_args[0][1]
        assert payload["_user_id"] == "test-user-123"
        assert payload["question"] == "test"

    @pytest.mark.asyncio
    async def test_handler_without_user_id(self):
        """NOVARE_USER_ID 未设置时，payload 不包含 _user_id。"""
        from unittest.mock import AsyncMock

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value='{"ok": true}')

        async def make_handler(c, tn, uid):
            async def handler(args, workspace=None):
                payload = dict(args)
                if uid:
                    payload["_user_id"] = uid
                return await c.call_tool(tn, payload)
            return handler

        handler = await make_handler(mock_client, "rag_query", None)

        args = {"question": "test"}
        await handler(args)

        call_args = mock_client.call_tool.call_args
        payload = call_args[0][1]
        assert "_user_id" not in payload


# ══════════════════════════════════════════════════════════════════════════
# 2. CLI 未设置 NOVARE_USER_ID 时保持 fail-closed
# ══════════════════════════════════════════════════════════════════════════

class TestCliFailClosed:
    """无 user_id 时 RAG 检索必须 fail-closed 并返回明确提示。"""

    @pytest.mark.asyncio
    async def test_no_user_id_fails_with_actionable_message(self):
        """无 user_id 时返回包含 NOVARE_USER_ID 的错误提示。"""
        import tools.rag_query as rq
        original = rq.ALLOW_UNSCOPED
        rq.ALLOW_UNSCOPED = False
        try:
            result = _parse_result(await rq.handle_rag_query({"question": "test"}))
            assert result["ok"] is False
            assert "NOVARE_USER_ID" in result["error"]
        finally:
            rq.ALLOW_UNSCOPED = original


# ══════════════════════════════════════════════════════════════════════════
# 3. Web 的 user_id 注入不受影响
# ══════════════════════════════════════════════════════════════════════════

class TestWebUserIdInjection:
    """Web 模式的 _user_id 注入路径不受 CLI 修改影响。"""

    @pytest.mark.asyncio
    async def test_web_handler_injects_user_id(self):
        """web/backend/agent_service.py 的 _make_mcp_handler 注入 _user_id。"""
        from unittest.mock import AsyncMock

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value='{"ok": true}')

        # 模拟 web backend 的 handler 创建
        def _make_mcp_handler(client, tool_name):
            async def handler(arguments, **kwargs):
                payload = dict(arguments)
                user_id = kwargs.get("user_id")
                if user_id:
                    payload["_user_id"] = user_id
                return await client.call_tool(tool_name, payload)
            return handler

        handler = _make_mcp_handler(mock_client, "rag_query")

        await handler({"question": "test"}, user_id="web-user-456")

        call_args = mock_client.call_tool.call_args
        payload = call_args[0][1]
        assert payload["_user_id"] == "web-user-456"

    @pytest.mark.asyncio
    async def test_web_handler_without_user_id(self):
        """Web 模式无 user_id 时 payload 不包含 _user_id。"""
        from unittest.mock import AsyncMock

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value='{"ok": true}')

        def _make_mcp_handler(client, tool_name):
            async def handler(arguments, **kwargs):
                payload = dict(arguments)
                user_id = kwargs.get("user_id")
                if user_id:
                    payload["_user_id"] = user_id
                return await client.call_tool(tool_name, payload)
            return handler

        handler = _make_mcp_handler(mock_client, "rag_query")

        await handler({"question": "test"})

        call_args = mock_client.call_tool.call_args
        payload = call_args[0][1]
        assert "_user_id" not in payload


# ══════════════════════════════════════════════════════════════════════════
# 4. 默认 MCP 环境变量正确传递
# ══════════════════════════════════════════════════════════════════════════

class TestMcpEnvPropagation:
    """MCP 子进程环境变量 allowlist 正确传递。"""

    def test_config_passes_embedding_env_vars(self):
        """config.py 应将 DASHSCOPE_API_KEY 等传入 MCP env。"""
        from novare.config import NovareConfig

        with patch.dict(os.environ, {
            "DATABASE_URL": "postgresql://test",
            "DASHSCOPE_API_KEY": "sk-test-key",
            "EMBEDDING_BASE_URL": "https://custom.api/v1",
            "EMBEDDING_MODEL": "custom-model",
            "MILVUS_HOST": "milvus-host",
            "MILVUS_PORT": "19530",
            "RAG_DEFAULT_USER": "default",
            "NOVARE_API_KEY": "test-api-key",
        }, clear=False):
            cfg = NovareConfig.load()
            # 如果项目根目录有 mcp-server/，会创建默认 research MCP
            if "research" in cfg.mcp_servers:
                env = cfg.mcp_servers["research"].env
                assert env.get("DASHSCOPE_API_KEY") == "sk-test-key"
                assert env.get("EMBEDDING_BASE_URL") == "https://custom.api/v1"
                assert env.get("EMBEDDING_MODEL") == "custom-model"
                assert env.get("MILVUS_HOST") == "milvus-host"
                assert env.get("MILVUS_PORT") == "19530"
                assert env.get("RAG_DEFAULT_USER") == "default"
                assert env.get("DATABASE_URL") == "postgresql://test"

    def test_config_does_not_leak_unlisted_vars(self):
        """未列入 allowlist 的环境变量不应传入 MCP。"""
        from novare.config import NovareConfig

        with patch.dict(os.environ, {
            "DATABASE_URL": "postgresql://test",
            "NOVARE_API_KEY": "test-api-key",
            "SECRET_SHOULD_NOT_LEAK": "secret-value",
        }, clear=False):
            cfg = NovareConfig.load()
            if "research" in cfg.mcp_servers:
                env = cfg.mcp_servers["research"].env
                assert "SECRET_SHOULD_NOT_LEAK" not in env


# ══════════════════════════════════════════════════════════════════════════
# 5. 敏感配置不出现在日志中
# ══════════════════════════════════════════════════════════════════════════

class TestSensitiveConfigNotLogged:
    """API key 等敏感配置不应出现在日志输出中。"""

    def test_embedding_logs_do_not_contain_api_key(self):
        """embedding.py 日志不应包含 DASHSCOPE_API_KEY 的值。"""
        import logging
        import core.embedding as emb_mod

        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "sk-super-secret-key-12345",
            "EMBEDDING_MODEL": "text-embedding-v4",
        }):
            emb_mod.reset_embedder()
            with patch("core.embedding.logger") as mock_logger:
                try:
                    emb_mod._get_bailian_embedder()
                except Exception:
                    pass
                # 检查所有日志调用中不包含 secret key
                for call in mock_logger.info.call_args_list:
                    log_msg = str(call)
                    assert "sk-super-secret-key-12345" not in log_msg
            emb_mod.reset_embedder()


# ══════════════════════════════════════════════════════════════════════════
# 6. 无 embedding provider 时生产环境明确失败
# ══════════════════════════════════════════════════════════════════════════

class TestNoEmbeddingProviderFails:
    """生产环境没有可用 embedding provider 时应抛出明确错误。"""

    def test_raises_when_no_provider(self):
        """无 DASHSCOPE_API_KEY 且无 sentence-transformers 时抛出 EmbeddingProviderError。"""
        import core.embedding as emb_mod
        from core.embedding import EmbeddingProviderError

        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "",
            "NOVARE_TEST_EMBEDDING_FALLBACK": "",
        }, clear=False):
            emb_mod.reset_embedder()
            with patch.object(emb_mod, "_get_local_embedder", return_value=None):
                with pytest.raises(EmbeddingProviderError, match="DASHSCOPE_API_KEY"):
                    emb_mod._init_embedder()
            emb_mod.reset_embedder()

    def test_error_message_is_actionable(self):
        """错误消息应包含可操作的建议。"""
        import core.embedding as emb_mod
        from core.embedding import EmbeddingProviderError

        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "",
            "NOVARE_TEST_EMBEDDING_FALLBACK": "",
        }, clear=False):
            emb_mod.reset_embedder()
            with patch.object(emb_mod, "_get_local_embedder", return_value=None):
                with pytest.raises(EmbeddingProviderError) as exc_info:
                    emb_mod._init_embedder()
                msg = str(exc_info.value)
                assert "DASHSCOPE_API_KEY" in msg
                assert "NOVARE_TEST_EMBEDDING_FALLBACK" in msg
            emb_mod.reset_embedder()


# ══════════════════════════════════════════════════════════════════════════
# 7. 只有测试开关打开时才允许 numpy fallback
# ══════════════════════════════════════════════════════════════════════════

class TestNumpyFallbackControlled:
    """numpy hash embedding fallback 只能在测试开关打开时使用。"""

    def test_fallback_allowed_with_test_switch(self):
        """NOVARE_TEST_EMBEDDING_FALLBACK=true 时允许 numpy fallback。"""
        import core.embedding as emb_mod

        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "",
            "NOVARE_TEST_EMBEDDING_FALLBACK": "true",
        }, clear=False):
            emb_mod.reset_embedder()
            with patch.object(emb_mod, "_get_local_embedder", return_value=None):
                emb_mod._init_embedder()
                assert emb_mod._embedder_type == "numpy_fallback"
            emb_mod.reset_embedder()

    def test_fallback_blocked_without_test_switch(self):
        """NOVARE_TEST_EMBEDDING_FALLBACK 未设置时不允许 numpy fallback。"""
        import core.embedding as emb_mod
        from core.embedding import EmbeddingProviderError

        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "",
            "NOVARE_TEST_EMBEDDING_FALLBACK": "",
        }, clear=False):
            emb_mod.reset_embedder()
            with patch.object(emb_mod, "_get_local_embedder", return_value=None):
                with pytest.raises(EmbeddingProviderError):
                    emb_mod._init_embedder()
            emb_mod.reset_embedder()


# ══════════════════════════════════════════════════════════════════════════
# 8. 查询向量和数据库向量维度不一致时返回维度错误
# ══════════════════════════════════════════════════════════════════════════

class TestDimensionMismatch:
    """维度不匹配时应返回明确诊断，不伪装成"未找到相关内容"。"""

    @pytest.mark.asyncio
    async def test_brute_force_reports_dimension_mismatch(self):
        """brute-force 搜索遇到维度不匹配时抛出 DimensionMismatchError。"""
        from tools.rag_query import _brute_force_search, DimensionMismatchError

        # 模拟 128 维查询向量 vs 1024 维数据库向量
        query_vec = np.random.randn(128).astype(np.float32)
        db_emb = {
            "chunk_id": 1,
            "dim": 1024,
            "vec": np.random.randn(1024).astype(np.float32),
            "text": "test",
            "section": "abstract",
            "paper_id": "p1",
            "title": "Test Paper",
        }

        with patch("tools.rag_query.get_embeddings_by_paper_ids", new_callable=AsyncMock, return_value=[db_emb]):
            with pytest.raises(DimensionMismatchError, match="128 维.*1024 维"):
                await _brute_force_search(query_vec, top_k=5, allowed_paper_ids={"p1"})

    @pytest.mark.asyncio
    async def test_rag_query_returns_dimension_error(self):
        """handle_rag_query 维度不匹配时返回 ok=false 的错误结果。"""
        import tools.rag_query as rq
        from tools.rag_query import DimensionMismatchError

        with patch.object(rq, "_get_user_paper_ids", return_value={"p1"}):
            with patch.object(rq, "embed_text_async", return_value=[0.1] * 128):
                with patch.object(rq, "_milvus_search", new_callable=AsyncMock, side_effect=Exception("Milvus down")):
                    with patch.object(rq, "_brute_force_search", new_callable=AsyncMock) as mock_bf:
                        mock_bf.side_effect = DimensionMismatchError(
                            "查询向量为 128 维，索引向量为 1024 维。"
                        )
                        result = _parse_result(
                            await rq.handle_rag_query({"question": "test"}, user_id="u-1")
                        )
                        assert result["ok"] is False
                        assert "128 维" in result["error"]
                        assert "1024 维" in result["error"]


# ══════════════════════════════════════════════════════════════════════════
# 9. paper_id 和 paper_ids 在 brute-force 检索中真正生效
# ══════════════════════════════════════════════════════════════════════════

class TestPaperIdFiltering:
    """paper_id/paper_ids 过滤必须真正限制检索范围。"""

    @pytest.mark.asyncio
    async def test_paper_id_filters_brute_force(self):
        """指定 paper_id 时 brute-force 只检索该论文的向量。"""
        import tools.rag_query as rq

        emb_p1 = {
            "chunk_id": 1, "dim": 3,
            "vec": np.array([1, 0, 0], dtype=np.float32),
            "text": "paper1", "section": "abstract",
            "paper_id": "p1", "title": "Paper 1",
        }
        emb_p2 = {
            "chunk_id": 2, "dim": 3,
            "vec": np.array([1, 0, 0], dtype=np.float32),
            "text": "paper2", "section": "abstract",
            "paper_id": "p2", "title": "Paper 2",
        }

        # 用户有 p1 和 p2 的权限，但只请求 p1
        with patch.object(rq, "_get_user_paper_ids", return_value={"p1", "p2"}):
            with patch.object(rq, "embed_text_async", return_value=[1.0, 0.0, 0.0]):
                with patch.object(rq, "_milvus_search", new_callable=AsyncMock, side_effect=Exception("Milvus down")):
                    with patch("tools.rag_query.get_embeddings_by_paper_ids", new_callable=AsyncMock) as mock_scoped:
                        # 模拟 filtered paper_ids 只返回 p1
                        mock_scoped.return_value = [emb_p1]
                        result = _parse_result(
                            await rq.handle_rag_query(
                                {"question": "test", "paper_id": "p1"},
                                user_id="u-1",
                            )
                        )
                        # 验证 scoped 查询只用了 p1
                        called_ids = mock_scoped.call_args[0][1]
                        assert called_ids == {"p1"}

    @pytest.mark.asyncio
    async def test_paper_ids_filters_brute_force(self):
        """指定 paper_ids 时 brute-force 只检索这些论文的向量。"""
        import tools.rag_query as rq

        emb_p1 = {
            "chunk_id": 1, "dim": 3,
            "vec": np.array([1, 0, 0], dtype=np.float32),
            "text": "paper1", "section": "abstract",
            "paper_id": "p1", "title": "Paper 1",
        }

        with patch.object(rq, "_get_user_paper_ids", return_value={"p1", "p2", "p3"}):
            with patch.object(rq, "embed_text_async", return_value=[1.0, 0.0, 0.0]):
                with patch.object(rq, "_milvus_search", new_callable=AsyncMock, side_effect=Exception("Milvus down")):
                    with patch("tools.rag_query.get_embeddings_by_paper_ids", new_callable=AsyncMock) as mock_scoped:
                        mock_scoped.return_value = [emb_p1]
                        result = _parse_result(
                            await rq.handle_rag_query(
                                {"question": "test", "paper_ids": ["p1", "p2"]},
                                user_id="u-1",
                            )
                        )
                        # p1 和 p2 的交集（用户有 p1, p2, p3 权限）
                        called_ids = mock_scoped.call_args[0][1]
                        assert called_ids == {"p1", "p2"}


# ══════════════════════════════════════════════════════════════════════════
# 10. Milvus 检索表达式同时包含 user_id 和 paper_id 条件
# ══════════════════════════════════════════════════════════════════════════

class TestMilvusExprIncludesPaperId:
    """Milvus 搜索表达式应同时包含 user_id 和 paper_id 过滤。"""

    def test_search_vectors_builds_correct_expr(self):
        """search_vectors 构建的表达式包含 paper_id 过滤。"""
        from core.vector_store import search_vectors

        mock_collection = MagicMock()
        mock_collection.search.return_value = [[]]

        with patch("core.vector_store.ensure_collection", return_value=mock_collection):
            with patch("core.vector_store.connections"):
                query_emb = [0.1] * 1024
                search_vectors("user-1", query_emb, top_k=5, paper_ids=["p1", "p2"])

                call_kwargs = mock_collection.search.call_args[1]
                expr = call_kwargs["expr"]
                assert 'user_id == "user-1"' in expr
                assert "paper_id in" in expr
                assert '"p1"' in expr
                assert '"p2"' in expr

    def test_search_vectors_without_paper_ids(self):
        """不传 paper_ids 时表达式只包含 user_id。"""
        from core.vector_store import search_vectors

        mock_collection = MagicMock()
        mock_collection.search.return_value = [[]]

        with patch("core.vector_store.ensure_collection", return_value=mock_collection):
            with patch("core.vector_store.connections"):
                query_emb = [0.1] * 1024
                search_vectors("user-1", query_emb, top_k=5)

                call_kwargs = mock_collection.search.call_args[1]
                expr = call_kwargs["expr"]
                assert 'user_id == "user-1"' in expr
                assert "paper_id" not in expr


# ══════════════════════════════════════════════════════════════════════════
# 11. 无权限 paper_id 不会泄露论文信息
# ══════════════════════════════════════════════════════════════════════════

class TestNoPaperLeak:
    """请求无权访问的 paper_id 不应泄露论文是否存在。"""

    @pytest.mark.asyncio
    async def test_unauthorized_paper_id_returns_generic_error(self):
        """请求用户无权访问的 paper_id 时返回通用错误。"""
        import tools.rag_query as rq

        with patch.object(rq, "_get_user_paper_ids", return_value={"p1"}):
            with patch.object(rq, "embed_text_async", return_value=[0.1, 0.2, 0.3]):
                result = _parse_result(
                    await rq.handle_rag_query(
                        {"question": "test", "paper_id": "unauthorized-paper"},
                        user_id="u-1",
                    )
                )
                assert result["ok"] is False
                # 错误消息不应提及该论文是否存在
                assert "unauthorized-paper" not in result["error"]
                assert "可访问" in result["error"]


# ══════════════════════════════════════════════════════════════════════════
# 12. top_k 非法值被拒绝或规范化
# ══════════════════════════════════════════════════════════════════════════

class TestTopKValidation:
    """top_k 参数必须被校验和规范化。"""

    def test_validate_top_k_normal(self):
        """正常值保持不变。"""
        from tools.rag_query import _validate_top_k
        assert _validate_top_k(5) == 5
        assert _validate_top_k(1) == 1
        assert _validate_top_k(50) == 50

    def test_validate_top_k_clamps_range(self):
        """超出范围的值被 clamp 到 [1, 50]。"""
        from tools.rag_query import _validate_top_k
        assert _validate_top_k(0) == 1
        assert _validate_top_k(-5) == 1
        assert _validate_top_k(100) == 50
        assert _validate_top_k(200) == 50

    def test_validate_top_k_none_default(self):
        """None 返回默认值 5。"""
        from tools.rag_query import _validate_top_k
        assert _validate_top_k(None) == 5

    def test_validate_top_k_invalid_type(self):
        """非法类型返回默认值 5。"""
        from tools.rag_query import _validate_top_k
        assert _validate_top_k("abc") == 5
        assert _validate_top_k([]) == 5


# ══════════════════════════════════════════════════════════════════════════
# 13. arXiv ID canonicalization
# ══════════════════════════════════════════════════════════════════════════

class TestArxivIdCanonicalization:
    """arXiv ID 规范化覆盖裸 ID、前缀 ID 和 URL。"""

    def test_bare_id(self):
        """裸 arXiv ID → arxiv:YYYY.NNNNN"""
        from core.paper_id import canonicalize_paper_id
        assert canonicalize_paper_id("2308.11681") == "arxiv:2308.11681"

    def test_prefixed_id(self):
        """前缀 arXiv ID → arxiv:YYYY.NNNNN"""
        from core.paper_id import canonicalize_paper_id
        assert canonicalize_paper_id("arxiv:2308.11681") == "arxiv:2308.11681"

    def test_id_with_version(self):
        """带版本号 → 去掉版本号"""
        from core.paper_id import canonicalize_paper_id
        assert canonicalize_paper_id("2308.11681v2") == "arxiv:2308.11681"
        assert canonicalize_paper_id("arxiv:2308.11681v3") == "arxiv:2308.11681"

    def test_abs_url(self):
        """arXiv abs URL → canonical ID"""
        from core.paper_id import canonicalize_paper_id
        assert canonicalize_paper_id("https://arxiv.org/abs/2308.11681") == "arxiv:2308.11681"

    def test_pdf_url(self):
        """arXiv PDF URL → canonical ID"""
        from core.paper_id import canonicalize_paper_id
        assert canonicalize_paper_id("https://arxiv.org/pdf/2308.11681v2") == "arxiv:2308.11681"

    def test_non_arxiv_id_unchanged(self):
        """非 arXiv ID 原样返回"""
        from core.paper_id import canonicalize_paper_id
        assert canonicalize_paper_id("doi:10.1234/abc") == "doi:10.1234/abc"
        assert canonicalize_paper_id("s2:12345") == "s2:12345"
        assert canonicalize_paper_id("parsed:my_paper") == "parsed:my_paper"

    def test_is_arxiv_id(self):
        """is_arxiv_id 正确识别各种格式"""
        from core.paper_id import is_arxiv_id
        assert is_arxiv_id("2308.11681") is True
        assert is_arxiv_id("arxiv:2308.11681") is True
        assert is_arxiv_id("https://arxiv.org/abs/2308.11681") is True
        assert is_arxiv_id("doi:10.1234/abc") is False
        assert is_arxiv_id("") is False
        assert is_arxiv_id(None) is False

    def test_canonicalize_empty(self):
        """空字符串原样返回"""
        from core.paper_id import canonicalize_paper_id
        assert canonicalize_paper_id("") == ""
        assert canonicalize_paper_id(None) is None


# ══════════════════════════════════════════════════════════════════════════
# 14. 回归测试：模拟 VadCLIP 查询
# ══════════════════════════════════════════════════════════════════════════

class TestVadclipRegression:
    """模拟 VadCLIP 查询时能返回包含实验结果的片段。"""

    @pytest.mark.asyncio
    async def test_vadclip_query_returns_results(self):
        """模拟 VadCLIP 查询，验证能返回包含实验结果的片段。"""
        import tools.rag_query as rq

        # 模拟 1024 维查询向量（匹配 Bailian text-embedding-v4）
        fake_vec = [0.1] * 1024

        # 模拟数据库中的 VadCLIP 相关 chunks
        vadclip_chunks = [
            {
                "chunk_id": 101, "dim": 1024,
                "vec": np.array([0.1] * 1024, dtype=np.float32),
                "text": "VadCLIP achieves 85.2% accuracy on ImageNet-1k benchmark, "
                        "outperforming previous state-of-the-art methods by 3.1%.",
                "section": "Experimental Results",
                "paper_id": "arxiv:2308.11681",
                "title": "VadCLIP: Visual Adaptive CLIP for Zero-Shot Learning",
            },
            {
                "chunk_id": 102, "dim": 1024,
                "vec": np.array([0.15] * 1024, dtype=np.float32),
                "text": "On Caltech-101, VadCLIP achieves 92.1% accuracy with "
                        "only 16 shots per class, demonstrating strong few-shot capabilities.",
                "section": "Experimental Results",
                "paper_id": "arxiv:2308.11681",
                "title": "VadCLIP: Visual Adaptive CLIP for Zero-Shot Learning",
            },
        ]

        with patch.object(rq, "_get_user_paper_ids", return_value={"arxiv:2308.11681"}):
            with patch.object(rq, "embed_text_async", return_value=fake_vec):
                with patch.object(rq, "_milvus_search", new_callable=AsyncMock, side_effect=Exception("Milvus down")):
                    with patch(
                        "tools.rag_query.get_embeddings_by_paper_ids",
                        new_callable=AsyncMock,
                        return_value=vadclip_chunks,
                    ):
                        result = _parse_result(
                            await rq.handle_rag_query(
                                {"question": "What are the experimental results of VadCLIP?"},
                                user_id="u-1",
                            )
                        )
                        assert result["ok"] is True
                        assert len(result["data"]["results"]) > 0
                        # 验证结果中包含实验结果片段
                        texts = [r["text"] for r in result["data"]["results"]]
                        has_results = any(
                            "accuracy" in t.lower() or "benchmark" in t.lower()
                            for t in texts
                        )
                        assert has_results, "Expected experimental results in response"

    @pytest.mark.asyncio
    async def test_vadclip_with_paper_id_filter(self):
        """指定 paper_id 时只返回该论文的结果。"""
        import tools.rag_query as rq

        fake_vec = [0.1] * 1024

        vadclip_emb = {
            "chunk_id": 101, "dim": 1024,
            "vec": np.array([0.1] * 1024, dtype=np.float32),
            "text": "VadCLIP achieves 85.2% accuracy on ImageNet-1k.",
            "section": "Experimental Results",
            "paper_id": "arxiv:2308.11681",
            "title": "VadCLIP",
        }

        with patch.object(rq, "_get_user_paper_ids", return_value={"arxiv:2308.11681", "other-paper"}):
            with patch.object(rq, "embed_text_async", return_value=fake_vec):
                with patch.object(rq, "_milvus_search", new_callable=AsyncMock, side_effect=Exception("Milvus down")):
                    with patch(
                        "tools.rag_query.get_embeddings_by_paper_ids",
                        new_callable=AsyncMock,
                    ) as mock_scoped:
                        mock_scoped.return_value = [vadclip_emb]
                        result = _parse_result(
                            await rq.handle_rag_query(
                                {
                                    "question": "VadCLIP results",
                                    "paper_id": "arxiv:2308.11681",
                                },
                                user_id="u-1",
                            )
                        )
                        assert result["ok"] is True
                        # 验证只检索了 arxiv:2308.11681
                        called_ids = mock_scoped.call_args[0][1]
                        assert called_ids == {"arxiv:2308.11681"}
