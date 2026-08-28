"""tests/test_agent_loop.py"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from novare.agent_loop import AgentLoop
from novare.hallucination_verifier import VerificationResult
from novare.llm_client import LLMResponse, ToolCall
from novare.tools.registry import ToolRegistry, ToolDef


class TestAgentLoop:
    def _make_loop(self, responses: list[LLMResponse], tool_handler=None):
        llm = AsyncMock()
        # agent_loop 现在调用 collect_stream（非生成器，直接返回 LLMResponse）
        llm.collect_stream = AsyncMock(side_effect=responses)
        llm.close = AsyncMock()

        registry = ToolRegistry()
        if tool_handler:
            for name, handler in tool_handler.items():
                registry.register_tool(ToolDef(
                    name=name,
                    description=f"test {name}",
                    parameters={"type": "object", "properties": {}},
                    handler=handler,
                ))

        return AgentLoop(
            llm_client=llm,
            tool_registry=registry,
            system_prompt="You are a test assistant.",
        )

    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        loop = self._make_loop([
            LLMResponse(content="Hello!", tool_calls=[], stop_reason="stop", usage={}),
        ])
        from novare.session import Session
        session = Session()
        result = await loop.run_turn(session, "Hi")
        assert result == "Hello!"
        assert len(session.messages) == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_verification_report_is_turn_state_not_message_metadata(self):
        async def rag_query(args, workspace=None):
            return json.dumps({"ok": True, "data": {"results": []}})

        loop = self._make_loop(
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="call_1", name="rag_query", arguments={})
                ], stop_reason="tool_calls", usage={}),
                LLMResponse(content="draft", tool_calls=[], stop_reason="stop", usage={}),
            ],
            tool_handler={"rag_query": rag_query},
        )
        verifier = AsyncMock()
        verifier.enabled = True
        verifier.verify = AsyncMock(return_value=VerificationResult(
            original_answer="draft",
            corrected_answer="verified",
            status="revised",
            did_revise=True,
        ))
        loop.hallucination_verifier = verifier

        from novare.session import Session
        session = Session()
        result = await loop.run_turn(session, "question")

        assert result == "verified"
        assert session.verification["status"] == "revised"
        assert "_verification" not in session.messages[-1]

    @pytest.mark.asyncio
    async def test_tool_call_then_response(self):
        async def mock_echo(args, workspace=None):
            return f"Echo: {args.get('message', '')}"

        loop = self._make_loop(
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="call_1", name="echo", arguments={"message": "test"})
                ], stop_reason="tool_calls", usage={}),
                LLMResponse(content="Done!", tool_calls=[], stop_reason="stop", usage={}),
            ],
            tool_handler={"echo": mock_echo},
        )
        from novare.session import Session
        session = Session()
        result = await loop.run_turn(session, "Echo test")
        assert result == "Done!"
        # user + assistant(tool_calls) + tool_result + assistant(final)
        assert len(session.messages) == 4

    @pytest.mark.asyncio
    async def test_max_iterations_returns_fallback(self):
        async def forever_loop(args, workspace=None):
            return "ok"

        responses = []
        for i in range(25):  # more than max_iterations=20
            responses.append(LLMResponse(content="", tool_calls=[
                ToolCall(id=f"call_{i}", name="echo", arguments={"message": str(i)})
            ], stop_reason="tool_calls", usage={}))

        loop = self._make_loop(responses, tool_handler={"echo": forever_loop})
        loop.max_iterations = 20

        from novare.session import Session
        session = Session()
        result = await loop.run_turn(session, "go")
        assert "最大迭代" in result or "迭代" in result or "重试" in result

    @pytest.mark.asyncio
    async def test_tool_error_is_reported_to_llm(self):
        async def failing_tool(args, workspace=None):
            raise ValueError("something broke")

        loop = self._make_loop(
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="call_1", name="fail_tool", arguments={})
                ], stop_reason="tool_calls", usage={}),
                LLMResponse(content="Tool had an error", tool_calls=[], stop_reason="stop", usage={}),
            ],
            tool_handler={"fail_tool": failing_tool},
        )
        from novare.session import Session
        session = Session()
        result = await loop.run_turn(session, "break it")
        assert result == "Tool had an error"
        # Check tool result message contains error info
        tool_msg = session.messages[2]
        assert "Error" in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_on_tool_callback_receives_events(self):
        """验证 on_tool 回调收到 start 和 end 事件"""
        async def mock_echo(args, workspace=None):
            return "echo result"

        loop = self._make_loop(
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="call_1", name="echo", arguments={"message": "hi"})
                ], stop_reason="tool_calls", usage={}),
                LLMResponse(content="Done", tool_calls=[], stop_reason="stop", usage={}),
            ],
            tool_handler={"echo": mock_echo},
        )

        events: list[tuple] = []

        def on_tool(event, name, args, result, elapsed):
            events.append((event, name, args, result, elapsed))

        from novare.session import Session
        session = Session()
        await loop.run_turn(session, "test", on_tool=on_tool)

        assert len(events) == 2
        assert events[0][0] == "start"
        assert events[0][1] == "echo"
        assert events[0][3] is None       # result_preview is None on start
        assert events[0][4] is None       # elapsed is None on start
        assert events[1][0] == "end"
        assert events[1][1] == "echo"
        assert "echo result" in events[1][3]  # result_preview
        assert events[1][4] is not None       # elapsed >= 0

    @pytest.mark.asyncio
    async def test_on_tool_callback_on_error(self):
        """验证 on_tool 回调在工具出错时收到 error 事件"""
        async def failing(args, workspace=None):
            raise RuntimeError("boom")

        loop = self._make_loop(
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="call_1", name="fail", arguments={})
                ], stop_reason="tool_calls", usage={}),
                LLMResponse(content="error handled", tool_calls=[], stop_reason="stop", usage={}),
            ],
            tool_handler={"fail": failing},
        )

        events: list[tuple] = []

        def on_tool(event, name, args, result, elapsed):
            events.append((event, name, args, result, elapsed))

        from novare.session import Session
        session = Session()
        await loop.run_turn(session, "test", on_tool=on_tool)

        assert len(events) == 2
        assert events[0][0] == "start"
        assert events[1][0] == "error"
        assert "boom" in events[1][3]

    @pytest.mark.asyncio
    async def test_should_cancel_stops_before_tool_execution(self):
        """should_cancel=True 在工具调用前停止，不执行工具。"""
        tool_called = False

        async def spy_handler(args, **kwargs):
            nonlocal tool_called
            tool_called = True
            return "tool result"

        loop = self._make_loop(
            [
                LLMResponse(content="", tool_calls=[
                    ToolCall(id="tc1", name="spy_tool", arguments={}),
                ], stop_reason="tool_calls", usage={}),
            ],
            tool_handler={"spy_tool": spy_handler},
        )

        from novare.session import Session
        session = Session()
        result = await loop.run_turn(session, "go", should_cancel=lambda: True)
        assert result == "任务已取消。"
        assert tool_called is False  # 工具未执行

    @pytest.mark.asyncio
    async def test_should_cancel_async_callback(self):
        """should_cancel 支持 async 回调。"""
        async def async_cancel():
            return True

        loop = self._make_loop([
            LLMResponse(content="Hello!", tool_calls=[], stop_reason="stop", usage={}),
        ])

        from novare.session import Session
        session = Session()
        result = await loop.run_turn(session, "Hi", should_cancel=async_cancel)
        assert result == "任务已取消。"
