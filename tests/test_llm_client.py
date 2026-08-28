"""tests/test_llm_client.py"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from novare.llm_client import LLMClient, LLMResponse, ToolCall


class TestLLMResponse:
    def test_text_response(self):
        resp = LLMResponse(
            content="Hello world",
            tool_calls=[],
            stop_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        assert resp.content == "Hello world"
        assert resp.tool_calls == []
        assert resp.stop_reason == "stop"

    def test_tool_call_response(self):
        tc = ToolCall(id="call_1", name="paper_search", arguments={"query": "test"})
        resp = LLMResponse(
            content="",
            tool_calls=[tc],
            stop_reason="tool_calls",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "paper_search"
        assert resp.tool_calls[0].arguments == {"query": "test"}


class TestParseResponse:
    def test_parse_text_only(self):
        from novare.llm_client import parse_chat_response
        raw = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        resp = parse_chat_response(raw)
        assert resp.content == "Hello"
        assert resp.tool_calls == []
        assert resp.stop_reason == "stop"

    def test_parse_tool_calls(self):
        from novare.llm_client import parse_chat_response
        raw = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "paper_search",
                            "arguments": json.dumps({"query": "test"}),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        resp = parse_chat_response(raw)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].id == "call_abc"
        assert resp.tool_calls[0].name == "paper_search"
        assert resp.stop_reason == "tool_calls"

    def test_parse_empty_content_with_tool_calls(self):
        from novare.llm_client import parse_chat_response
        raw = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "/tmp/test.txt"}),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
        resp = parse_chat_response(raw)
        assert resp.content == ""
        assert len(resp.tool_calls) == 1

    def test_parse_reasoning_content_separately(self):
        from novare.llm_client import parse_chat_response
        raw = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "{\"claims\": []}",
                    "reasoning_content": "internal reasoning",
                },
                "finish_reason": "stop",
            }],
        }
        resp = parse_chat_response(raw)
        assert resp.content == '{"claims": []}'
        assert resp.reasoning_content == "internal reasoning"


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_chat_makes_http_request(self):
        client = LLMClient(api_key="test-key", base_url="https://api.test.com/v1", model="test-model")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hi there"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }

        with patch.object(client._http, "post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
            resp = await client.chat([{"role": "user", "content": "Hello"}])
            assert resp.content == "Hi there"
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "/chat/completions" in call_args[0][0]
            body = call_args[1]["json"]
            assert body["model"] == "test-model"
            assert body["messages"][0]["content"] == "Hello"


class TestLLMClientLifecycle:
    """C2: async context manager + close 幂等 + closed 后报错"""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """async with 返回 self，退出时调用 aclose。"""
        client = LLMClient(api_key="k", base_url="https://api.test.com/v1", model="m")
        client._http.aclose = AsyncMock()

        async with client as c:
            assert c is client
            assert not client._closed

        client._http.aclose.assert_awaited_once()
        assert client._closed

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        """重复调用 close() 只调用一次 aclose。"""
        client = LLMClient(api_key="k", base_url="https://api.test.com/v1", model="m")
        client._http.aclose = AsyncMock()

        await client.close()
        await client.close()

        client._http.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_chat_after_close_raises(self):
        """close 后调用 chat 抛 RuntimeError。"""
        client = LLMClient(api_key="k", base_url="https://api.test.com/v1", model="m")
        client._http.aclose = AsyncMock()
        await client.close()

        with pytest.raises(RuntimeError, match="LLMClient is closed"):
            await client.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_collect_stream_after_close_raises(self):
        """close 后调用 collect_stream 抛 RuntimeError。"""
        client = LLMClient(api_key="k", base_url="https://api.test.com/v1", model="m")
        client._http.aclose = AsyncMock()
        await client.close()

        with pytest.raises(RuntimeError, match="LLMClient is closed"):
            await client.collect_stream([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_context_manager_on_closed_client_raises(self):
        """对已关闭的 client 使用 async with 抛 RuntimeError。"""
        client = LLMClient(api_key="k", base_url="https://api.test.com/v1", model="m")
        client._http.aclose = AsyncMock()
        await client.close()

        with pytest.raises(RuntimeError, match="LLMClient is closed"):
            async with client:
                pass
