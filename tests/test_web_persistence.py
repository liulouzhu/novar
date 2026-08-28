"""tests/test_web_persistence.py — Web 模式持久化分离测试

验证：
- autosave=False 时 compact 不调用 session.save()
- autosave=True 时保持旧行为
- on_compact 回调在 compact 发生时被调用
- on_message 回调独立收集未压缩的原始消息
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from novare.agent_loop import AgentLoop
from novare.context_manager import UsageTracker, TokenUsage
from novare.llm_client import LLMResponse, ToolCall


def _make_loop(responses: list[LLMResponse], auto_compact_threshold: int = 100_000):
    llm = AsyncMock()
    llm.collect_stream = AsyncMock(side_effect=responses)
    llm.close = AsyncMock()

    registry = MagicMock()
    registry.to_openai_tools = MagicMock(return_value=[])
    registry.execute = AsyncMock(return_value="ok")

    return AgentLoop(
        llm_client=llm,
        tool_registry=registry,
        system_prompt="You are a test assistant.",
        auto_compact_threshold=auto_compact_threshold,
        context_llm_enabled=False,
    )


def _make_session_with_messages(count: int = 10):
    """创建一个有足够消息的 session，使 compact 能触发。"""
    session = MagicMock()
    session.session_id = "test-session"
    session.messages = []
    for i in range(count):
        if i % 2 == 0:
            session.messages.append({"role": "user", "content": f"message {i} " * 20})
        else:
            session.messages.append({"role": "assistant", "content": f"response {i} " * 20})
    session.usage_tracker = UsageTracker()
    session.save = MagicMock()
    session.add_user_message = MagicMock(side_effect=lambda content: session.messages.append({"role": "user", "content": content}))
    session.add_assistant_message = MagicMock(side_effect=lambda content, tool_calls=None: session.messages.append({"role": "assistant", "content": content, **({"tool_calls": tool_calls} if tool_calls else {})}))
    session.add_tool_result = MagicMock(side_effect=lambda tc_id, content: session.messages.append({"role": "tool", "tool_call_id": tc_id, "content": content}))
    return session


def _inject_usage(session, input_tokens: int = 50_000):
    """向 usage_tracker 注入大量 token 以触发 should_compact。"""
    session.usage_tracker.add(TokenUsage(input_tokens=input_tokens, output_tokens=100))


# ── autosave=False: compact 不调用 session.save() ──────────────────────

class TestAutosaveFalse:
    @pytest.mark.asyncio
    async def test_compact_does_not_call_save_when_autosave_false(self):
        """autosave=False 时，compact 发生后 session.save() 不被调用。"""
        session = _make_session_with_messages(10)
        _inject_usage(session, input_tokens=200_000)

        loop = _make_loop(
            [LLMResponse(content="ok", tool_calls=[], stop_reason="stop", usage={})],
            auto_compact_threshold=100,
        )

        await loop.run_turn(session, "hi", autosave=False)

        session.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_compact_does_not_call_save_no_compact_autosave_false(self):
        """autosave=False 且未发生 compact 时，session.save() 也不被调用。"""
        session = _make_session_with_messages(2)
        session.usage_tracker = UsageTracker()

        loop = _make_loop(
            [LLMResponse(content="ok", tool_calls=[], stop_reason="stop", usage={})],
            auto_compact_threshold=100_000,
        )

        await loop.run_turn(session, "hi", autosave=False)

        session.save.assert_not_called()


# ── autosave=True: 保持旧行为 ──────────────────────────────────────────

class TestAutosaveTrue:
    @pytest.mark.asyncio
    async def test_compact_calls_save_when_autosave_true(self):
        """autosave=True 时，compact 发生后 session.save() 被调用。"""
        session = _make_session_with_messages(10)
        _inject_usage(session, input_tokens=200_000)

        loop = _make_loop(
            [LLMResponse(content="ok", tool_calls=[], stop_reason="stop", usage={})],
            auto_compact_threshold=100,
        )

        await loop.run_turn(session, "hi", autosave=True)

        session.save.assert_called()


# ── on_compact 回调 ─────────────────────────────────────────────────────

class TestOnCompactCallback:
    @pytest.mark.asyncio
    async def test_on_compact_called_when_compact_happens(self):
        """compact 发生时 on_compact 回调被调用，传入 session。"""
        session = _make_session_with_messages(10)
        _inject_usage(session, input_tokens=200_000)

        on_compact = MagicMock()
        loop = _make_loop(
            [LLMResponse(content="ok", tool_calls=[], stop_reason="stop", usage={})],
            auto_compact_threshold=100,
        )

        await loop.run_turn(session, "hi", autosave=False, on_compact=on_compact)

        on_compact.assert_called_once()
        assert on_compact.call_args[0][0] is session

    @pytest.mark.asyncio
    async def test_on_compact_not_called_when_no_compact(self):
        """未发生 compact 时 on_compact 不被调用。"""
        session = _make_session_with_messages(2)

        on_compact = MagicMock()
        loop = _make_loop(
            [LLMResponse(content="ok", tool_calls=[], stop_reason="stop", usage={})],
            auto_compact_threshold=100_000,
        )

        await loop.run_turn(session, "hi", autosave=False, on_compact=on_compact)

        on_compact.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_compact_async_callback(self):
        """on_compact 是 async 函数时也能正确 await。"""
        session = _make_session_with_messages(10)
        _inject_usage(session, input_tokens=200_000)

        on_compact = AsyncMock()
        loop = _make_loop(
            [LLMResponse(content="ok", tool_calls=[], stop_reason="stop", usage={})],
            auto_compact_threshold=100,
        )

        await loop.run_turn(session, "hi", autosave=False, on_compact=on_compact)

        on_compact.assert_awaited_once()


# ── 原始消息回调 ────────────────────────────────────────────────────────

class TestRawMessageCallback:
    @pytest.mark.asyncio
    async def test_collects_all_new_messages_when_compaction_replaces_session_list(self):
        session = _make_session_with_messages(10)
        _inject_usage(session, input_tokens=200_000)
        tool_call = ToolCall(id="tc1", name="search", arguments={"q": "x"})
        loop = _make_loop(
            [
                LLMResponse(content="checking", tool_calls=[tool_call], stop_reason="tool_calls", usage={}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop", usage={}),
            ],
            auto_compact_threshold=100,
        )
        loop.tool_registry.execute = AsyncMock(return_value="tool result")
        captured = []

        await loop.run_turn(
            session,
            "new request",
            autosave=False,
            on_message=lambda message: captured.append(message),
        )

        assert [message["role"] for message in captured] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert captured[0]["content"] == "new request"
        assert captured[2]["tool_call_id"] == "tc1"


def test_build_messages_strips_internal_compaction_metadata():
    loop = _make_loop([], auto_compact_threshold=100_000)
    session = _make_session_with_messages(0)
    session.messages = [{
        "role": "tool",
        "content": "compressed result",
        "tool_call_id": "tc1",
        "_compacted_tool_result": True,
        "_compaction_meta": {"schema_version": 2},
    }]

    messages = loop._build_messages(session)

    assert messages[-1] == {
        "role": "tool",
        "content": "compressed result",
        "tool_call_id": "tc1",
    }
    assert "_compaction_meta" not in messages[-1]
    assert "_compacted_tool_result" not in messages[-1]
