"""tests/test_innovation_search.py — innovation_search 回归测试

验证：
- 使用 async context manager
- upsert_paper 被 await
- associate_user_paper 被 await
- 数据库异常时返回结构化 warning 而非伪装成功
- warning 不包含敏感信息
"""

import json
import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_innovation_search_uses_async_context_manager():
    """innovation_search 使用 async with get_connection()。"""
    with patch("tools.innovation_search.get_connection") as mock_get_conn:
        mock_conn = AsyncMock()
        mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("tools.innovation_search._search_multi_source", new_callable=AsyncMock, return_value=[
            {"paper_id": "arxiv:1234", "title": "Test Paper", "authors": ["Author"],
             "abstract": "Abstract", "year": 2024, "citation_count": 10, "source": "arxiv"},
        ]):
            with patch("tools.innovation_search.upsert_paper", new_callable=AsyncMock) as mock_upsert:
                with patch("tools.paper_parse.associate_user_paper", new_callable=AsyncMock) as mock_associate:
                    from tools.innovation_search import handle_innovation_search
                    result = await handle_innovation_search(
                        {"action": "landscape", "topic": "test"}, user_id="user-1"
                    )

        # get_connection 应该被作为 async context manager 调用
        mock_get_conn.return_value.__aenter__.assert_awaited_once()
        mock_get_conn.return_value.__aexit__.assert_awaited_once()

        # upsert_paper 应该被 await
        mock_upsert.assert_awaited_once()

        # associate_user_paper 应该被 await
        mock_associate.assert_awaited_once()

        # 返回结果应该是成功
        parsed = json.loads(result)
        assert parsed["ok"] is True
        assert parsed["warnings"] == []


@pytest.mark.asyncio
async def test_innovation_search_db_error_returns_warning():
    """数据库异常时返回结构化 warning，搜索结果仍存在。"""
    with patch("tools.innovation_search.get_connection") as mock_get_conn:
        mock_conn = AsyncMock()
        mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("tools.innovation_search._search_multi_source", new_callable=AsyncMock, return_value=[
            {"paper_id": "arxiv:1234", "title": "Test Paper", "authors": ["Author"],
             "abstract": "Abstract", "year": 2024, "citation_count": 10, "source": "arxiv"},
        ]):
            with patch("tools.innovation_search.upsert_paper", new_callable=AsyncMock) as mock_upsert:
                mock_upsert.side_effect = RuntimeError("DB connection refused")
                with patch("tools.paper_parse.associate_user_paper", new_callable=AsyncMock) as mock_associate:
                    from tools.innovation_search import handle_innovation_search
                    result = await handle_innovation_search(
                        {"action": "landscape", "topic": "test"}, user_id="user-1"
                    )

        # upsert_paper 应该被调用（在异常之前）
        mock_upsert.assert_awaited_once()

        # 解析返回结果
        parsed = json.loads(result)
        # 搜索结果仍然存在
        assert parsed["ok"] is True
        assert parsed["data"]["total_papers"] == 1
        assert len(parsed["data"]["papers"]) == 1

        # warnings 非空
        assert len(parsed["warnings"]) == 1
        assert "持久化失败" in parsed["warnings"][0]

        # warning 不包含敏感信息
        assert "password" not in parsed["warnings"][0].lower()
        assert "localhost" not in parsed["warnings"][0]
        assert "connection" not in parsed["warnings"][0].lower()


@pytest.mark.asyncio
async def test_innovation_search_associate_failure_returns_warning():
    """associate_user_paper 失败时也返回 warning。"""
    with patch("tools.innovation_search.get_connection") as mock_get_conn:
        mock_conn = AsyncMock()
        mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("tools.innovation_search._search_multi_source", new_callable=AsyncMock, return_value=[
            {"paper_id": "arxiv:1234", "title": "Test Paper", "authors": ["Author"],
             "abstract": "Abstract", "year": 2024, "citation_count": 10, "source": "arxiv"},
        ]):
            with patch("tools.innovation_search.upsert_paper", new_callable=AsyncMock):
                with patch("tools.paper_parse.associate_user_paper", new_callable=AsyncMock) as mock_associate:
                    mock_associate.side_effect = RuntimeError("Constraint violation")
                    from tools.innovation_search import handle_innovation_search
                    result = await handle_innovation_search(
                        {"action": "landscape", "topic": "test"}, user_id="user-1"
                    )

        mock_associate.assert_awaited_once()

        parsed = json.loads(result)
        assert parsed["ok"] is True
        assert len(parsed["warnings"]) == 1
        assert "持久化失败" in parsed["warnings"][0]


@pytest.mark.asyncio
async def test_innovation_search_no_unawaited_coroutines():
    """验证没有未 await 的协程。"""
    with patch("tools.innovation_search.get_connection") as mock_get_conn:
        mock_conn = AsyncMock()
        mock_get_conn.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_get_conn.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("tools.innovation_search._search_multi_source", new_callable=AsyncMock, return_value=[]):
            with patch("tools.innovation_search.upsert_paper", new_callable=AsyncMock) as mock_upsert:
                with patch("tools.paper_parse.associate_user_paper", new_callable=AsyncMock) as mock_associate:
                    from tools.innovation_search import handle_innovation_search
                    result = await handle_innovation_search(
                        {"action": "landscape", "topic": "test"}, user_id="user-1"
                    )

        # 空结果不会调用 upsert/associate
        mock_upsert.assert_not_awaited()
        mock_associate.assert_not_awaited()

        parsed = json.loads(result)
        assert parsed["ok"] is True
        assert parsed["data"]["total_papers"] == 0
