"""tests/test_mcp_client.py"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from novare.mcp_client import McpClient


class TestMcpClientInit:
    def test_create_client(self):
        client = McpClient(command="python", args=["server.py"])
        assert client._params.command == "python"
        assert client._params.args == ["server.py"]
        assert client._session is None


class TestMcpClientMock:
    @pytest.mark.asyncio
    async def test_list_tools(self):
        client = McpClient(command="echo", args=[])
        client._tools = [
            {"name": "paper_search", "description": "Search papers", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}},
            {"name": "paper_parse", "description": "Parse paper", "inputSchema": {"type": "object", "properties": {"paper_id": {"type": "string"}}}},
        ]
        tools = await client.list_tools()
        assert len(tools) == 2
        assert tools[0]["name"] == "paper_search"

    @pytest.mark.asyncio
    async def test_call_tool(self):
        client = McpClient(command="echo", args=[])
        mock_session = AsyncMock()
        mock_content = MagicMock()
        mock_content.text = "search result"
        mock_session.call_tool = AsyncMock(return_value=MagicMock(content=[mock_content]))
        client._session = mock_session

        result = await client.call_tool("paper_search", {"query": "test"})
        assert result == "search result"
        mock_session.call_tool.assert_called_once_with("paper_search", {"query": "test"})

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self):
        client = McpClient(command="echo", args=[])
        with pytest.raises(RuntimeError, match="not connected"):
            await client.call_tool("test", {})


class TestMcpClientConnectGuard:
    """C3: connect() 异常保护 + close() 幂等"""

    @pytest.mark.asyncio
    async def test_connect_cleans_up_on_session_init_failure(self):
        """ClientSession.__aenter__ 或 initialize 失败时，close() 被调用清理资源。"""
        client = McpClient(command="echo", args=[])

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=(AsyncMock(), AsyncMock()))
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch("novare.mcp_client.stdio_client", return_value=mock_cm):
            with patch("novare.mcp_client.ClientSession") as MockSession:
                mock_session_instance = AsyncMock()
                mock_session_instance.__aenter__ = AsyncMock(side_effect=RuntimeError("init boom"))
                MockSession.return_value = mock_session_instance

                with pytest.raises(RuntimeError, match="init boom"):
                    await client.connect()

        # close() 应该被调用了，状态应已清理
        assert client._session is None
        assert client._cm is None
        assert client._tools == []

    @pytest.mark.asyncio
    async def test_close_is_idempotent(self):
        """重复调用 close() 不抛异常。"""
        client = McpClient(command="echo", args=[])
        client._session = AsyncMock()
        client._cm = AsyncMock()
        client._tools = [{"name": "x"}]

        await client.close()
        assert client._session is None
        assert client._cm is None
        assert client._tools == []

        # 第二次调用不抛异常
        await client.close()
        assert client._session is None

    @pytest.mark.asyncio
    async def test_close_clears_state(self):
        """close() 后状态完全清理。"""
        client = McpClient(command="echo", args=[])
        client._session = AsyncMock()
        client._cm = AsyncMock()
        client._tools = [{"name": "a"}]

        await client.close()
        assert client._session is None
        assert client._cm is None
        assert client._tools == []
