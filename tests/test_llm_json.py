"""tests/test_llm_json.py — 统一 LLM JSON 解析/重试 + 本次重构的回归测试"""

import json

import pytest

from novare.llm_json import call_llm_json, parse_json_object, parse_model_json
from novare.task_state import TaskStateManager, _TOOL_HANDLERS, register_tool_handler


class _FakeResponse:
    def __init__(self, content: str, stop_reason="stop", reasoning=""):
        self.content = content
        self.stop_reason = stop_reason
        self.reasoning_content = reasoning


class _FakeLLM:
    """依次返回预置响应，记录收到的 messages。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=None, max_tokens=4096):
        self.calls.append(messages)
        return self.responses.pop(0)


# ── parse_json_object ────────────────────────────────────────


class TestParseJsonObject:
    def test_plain_object(self):
        assert parse_json_object('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}

    def test_trailing_text_rejected_strict(self):
        with pytest.raises(ValueError):
            parse_json_object('{"a": 1}\n说明文字')

    def test_two_objects_rejected_strict(self):
        with pytest.raises(ValueError):
            parse_json_object('{"a": 1}{"b": 2}')

    def test_tolerant_extracts_first_block(self):
        assert parse_json_object('开头文字 {"a": 1} 结尾', tolerant=True) == {"a": 1}

    def test_tolerant_returns_none_on_garbage(self):
        assert parse_json_object("not json at all", tolerant=True) is None

    def test_parse_model_json_alias(self):
        assert parse_model_json('```json\n{"ok": true}\n```') == {"ok": True}
        assert parse_model_json("") is None
        assert parse_model_json("no json") is None


# ── call_llm_json 重试循环 ────────────────────────────────────


class TestCallLlmJson:
    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        llm = _FakeLLM([_FakeResponse('{"claims": []}')])
        payload, attempts = await call_llm_json(
            llm, system_prompt="s", user_prompt="u", max_tokens=100,
        )
        assert payload == {"claims": []}
        assert attempts == 1

    @pytest.mark.asyncio
    async def test_retry_with_error_feedback(self):
        llm = _FakeLLM([
            _FakeResponse("这不是 JSON"),
            _FakeResponse('{"ok": true}'),
        ])
        payload, attempts = await call_llm_json(
            llm, system_prompt="s", user_prompt="u", max_tokens=100,
        )
        assert payload == {"ok": True}
        assert attempts == 2
        # 第二次调用带错误反馈消息
        feedback = [m for m in llm.calls[1] if "Previous output was rejected" in str(m.get("content"))]
        assert feedback, "重试时应附带上次错误"

    @pytest.mark.asyncio
    async def test_validate_failure_exhausts_attempts(self):
        llm = _FakeLLM([
            _FakeResponse('{"a": 1}'),
            _FakeResponse('{"a": 1}'),
        ])

        def _reject(payload):
            raise ValueError("bad payload")

        with pytest.raises(ValueError):
            await call_llm_json(
                llm, system_prompt="s", user_prompt="u", max_tokens=100,
                validate=_reject, max_attempts=2,
            )
        assert len(llm.calls) == 2

    @pytest.mark.asyncio
    async def test_empty_content_diagnoses_reasoning(self):
        llm = _FakeLLM([_FakeResponse("", stop_reason="length", reasoning="想了很多")])
        with pytest.raises(ValueError, match="reasoning_chars=4"):
            await call_llm_json(
                llm, system_prompt="s", user_prompt="u", max_tokens=100,
                max_attempts=1,
            )

    @pytest.mark.asyncio
    async def test_convert_transforms_result(self):
        llm = _FakeLLM([_FakeResponse('{"n": 2}')])
        result, _ = await call_llm_json(
            llm, system_prompt="s", user_prompt="u", max_tokens=100,
            convert=lambda p: p["n"] * 10,
        )
        assert result == 20


# ── task_state handler 注册 ───────────────────────────────────


class TestTaskStateHandlers:
    def test_builtin_handlers_registered(self):
        assert "paper_search" in _TOOL_HANDLERS
        assert "rag_query" in _TOOL_HANDLERS

    def test_update_from_tool_unknown_tool_silent(self):
        mgr = TaskStateManager()
        mgr.init_turn("do things")
        mgr.update_from_tool("mcp_dynamic_tool", {}, '{"ok": true}')
        assert mgr.state.tools_used == ["mcp_dynamic_tool"]

    def test_register_custom_handler(self):
        calls = []

        def _handler(state, arguments, result):
            calls.append(arguments)
            state.key_findings.append("custom finding")

        register_tool_handler("custom_tool", _handler)
        try:
            mgr = TaskStateManager()
            mgr.init_turn("goal")
            mgr.update_from_tool("custom_tool", {"x": 1}, "{}")
            assert calls == [{"x": 1}]
            assert "custom finding" in mgr.state.key_findings
        finally:
            _TOOL_HANDLERS.pop("custom_tool", None)


# ── subagent await 超时参数化 ─────────────────────────────────


class TestSubagentAwaitTimeout:
    def test_default_follows_turn_timeout(self):
        """默认值从 tool_context 读取 turn_timeout 对齐，不再硬编码 300s。"""
        from novare.subagents import tools as sub_tools
        import inspect

        source = inspect.getsource(sub_tools.handle_spawn_subagent)
        assert "timeout=300" not in source, "await 超时不应再硬编码 300s"
        assert "subagent_await_timeout" in source
