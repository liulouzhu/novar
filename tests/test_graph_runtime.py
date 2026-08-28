"""tests/test_graph_runtime.py — LangGraph 运行时测试

覆盖 GraphRunner 与 legacy AgentLoop 的行为对齐点：
- 纯文本回答 / 工具调用循环 / 达到最大迭代 / 协作式取消
- RAG buffering + 幻觉校验路径
- usage 记账、事件回调集合、RecoveryState 快照
- ModelPort 双实现（Legacy / LangChain）与消息转换
- registry 结构化错误 + timeout 生效
"""

import json

import pytest

from novare.context_compactor import HybridContextCompactor
from novare.graph.adapters.model import (
    LangChainModelPort,
    LegacyModelPort,
    ModelResult,
    StreamDelta,
    ToolCallSpec,
    build_model_port,
)
from novare.graph.runner import GraphRunner
from novare.graph.state import ReplaceMessages, merge_messages
from novare.recovery.types import Outcome
from novare.session import Session
from novare.tools.registry import ToolDef, ToolRegistry


# ── 测试替身 ──────────────────────────────────────────────────


class FakeModelPort:
    """依次返回预置 ModelResult，记录调用。"""

    def __init__(self, responses: list[ModelResult]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    async def astream_collect(self, messages, tools=None, on_delta=None):
        self.calls.append(messages)
        result = self.responses.pop(0)
        if on_delta is not None and result.content:
            on_delta(StreamDelta(text=result.content))
        return result


class FakeVerifier:
    enabled = True

    def __init__(self, corrected: str = "已修正的回答"):
        self.corrected = corrected
        self.calls: list[dict] = []

    async def verify(self, answer, user_question, tool_context=None):
        self.calls.append({"answer": answer, "question": user_question})
        return VerificationResultLike(corrected=self.corrected)


class VerificationResultLike:
    def __init__(self, corrected: str):
        self.corrected_answer = corrected
        self.report = {"risk": "high", "claims": []}

    def to_dict(self):
        return dict(self.report)


def _make_registry(handler_map: dict | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    if handler_map:
        for name, handler in handler_map.items():
            registry.register_tool(ToolDef(
                name=name,
                description=f"test {name}",
                parameters={"type": "object", "properties": {}},
                handler=handler,
            ))
    return registry


def _make_runner(model, registry, **kwargs) -> GraphRunner:
    defaults = dict(
        system_prompt="You are a test assistant.",
        max_iterations=5,
        turn_timeout=30,
        auto_compact_threshold=0,  # 测试默认关闭压缩
    )
    defaults.update(kwargs)
    return GraphRunner(model=model, tool_registry=registry, **defaults)


def _text(content: str) -> ModelResult:
    return ModelResult(content=content, tool_calls=[], stop_reason="stop", usage={})


def _tool_call(id_: str, name: str, args: dict) -> ModelResult:
    return ModelResult(
        content="",
        tool_calls=[ToolCallSpec(id=id_, name=name, arguments=args)],
        stop_reason="tool_calls",
        usage={},
    )


async def _noop_handler(args, **kwargs) -> str:
    return json.dumps({"ok": True, "summary": "done"})


# ── 基本流程 ──────────────────────────────────────────────────


class TestGraphRunnerBasics:
    @pytest.mark.asyncio
    async def test_simple_text_response(self, tmp_path):
        model = FakeModelPort([_text("Hello!")])
        runner = _make_runner(model, _make_registry())
        session = Session(workspace=tmp_path)

        chunks: list[str] = []
        answer = await runner.run_turn(session, "Hi", on_text=chunks.append)

        assert answer == "Hello!"
        assert chunks == ["Hello!"]
        assert [m["role"] for m in session.messages] == ["user", "assistant"]
        assert session.messages[-1]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_tool_call_loop(self, tmp_path):
        model = FakeModelPort([
            _tool_call("call_1", "search_papers", {"query": "llm"}),
            _text("Found 3 papers."),
        ])
        registry = _make_registry({"search_papers": _noop_handler})
        runner = _make_runner(model, registry)
        session = Session(workspace=tmp_path)

        events: list[tuple] = []
        answer = await runner.run_turn(
            session, "search llm papers",
            on_tool=lambda phase, name, args, result, elapsed: events.append((phase, name)),
        )

        assert answer == "Found 3 papers."
        assert events == [("start", "search_papers"), ("end", "search_papers")]
        roles = [m["role"] for m in session.messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        tool_msg = session.messages[2]
        assert tool_msg["tool_call_id"] == "call_1"
        assert "ok" in tool_msg["content"]
        # 两次模型调用：第一次带用户消息，第二次带工具结果
        assert len(model.calls) == 2
        assert any(m.get("role") == "tool" for m in model.calls[1])

    @pytest.mark.asyncio
    async def test_max_iterations(self, tmp_path):
        endless = [_tool_call(f"call_{i}", "search_papers", {}) for i in range(10)]
        model = FakeModelPort(endless)
        runner = _make_runner(model, _make_registry({"search_papers": _noop_handler}),
                              max_iterations=3)
        session = Session(workspace=tmp_path)

        answer = await runner.run_turn(session, "loop forever")
        assert "最大迭代次数（3）" in answer
        assert len(model.responses) == 7  # 3 次调用后停止

    @pytest.mark.asyncio
    async def test_cancel_before_tools(self, tmp_path):
        model = FakeModelPort([_tool_call("call_1", "search_papers", {})])
        runner = _make_runner(model, _make_registry({"search_papers": _noop_handler}))

        async def cancelled() -> bool:
            return True

        session = Session(workspace=tmp_path)
        answer = await runner.run_turn(session, "hi", should_cancel=cancelled)
        assert answer == "任务已取消。"

    @pytest.mark.asyncio
    async def test_usage_accounting(self, tmp_path):
        model = FakeModelPort([ModelResult(
            content="ok", tool_calls=[], stop_reason="stop",
            usage={"prompt_tokens": 100, "completion_tokens": 20},
        )])
        runner = _make_runner(model, _make_registry())
        session = Session(workspace=tmp_path)
        await runner.run_turn(session, "hi")
        assert session.usage_tracker.cumulative_input == 100
        assert session.usage_tracker.cumulative_output == 20

    @pytest.mark.asyncio
    async def test_recovery_state_events(self, tmp_path):
        model = FakeModelPort([
            _tool_call("call_1", "search_papers", {}),
            _text("done"),
        ])
        runner = _make_runner(model, _make_registry({"search_papers": _noop_handler}))
        session = Session(workspace=tmp_path)

        snapshots: list[dict] = []
        await runner.run_turn(
            session, "hi",
            on_recovery_state=lambda snap: snapshots.append(snap),
        )
        assert snapshots, "应推送 RecoveryState 快照"
        final = snapshots[-1]
        assert final["run_status"] == "completed"
        assert any(rec["status"] == "completed" for rec in final["tool_calls"].values())


# ── RAG 校验路径 ──────────────────────────────────────────────


class TestVerificationPath:
    @pytest.mark.asyncio
    async def test_rag_used_triggers_verification(self, tmp_path):
        model = FakeModelPort([
            _tool_call("call_1", "rag_query", {"query": "q"}),
            _text("raw answer"),
        ])
        registry = _make_registry({"rag_query": _noop_handler})
        verifier = FakeVerifier(corrected="verified answer")
        runner = _make_runner(model, registry, hallucination_verifier=verifier)
        session = Session(workspace=tmp_path)

        streamed: list[str] = []
        reports: list[dict] = []
        answer = await runner.run_turn(
            session, "question",
            on_text=streamed.append,
            on_verification=lambda r: reports.append(r),
        )

        assert answer == "verified answer"
        assert len(verifier.calls) == 1
        # 缓冲模式：原始回答不流式，校验后补发修正回答
        assert streamed == ["verified answer"]
        assert reports and reports[0]["risk"] == "high"
        assert session.messages[-1]["content"] == "verified answer"
        assert session.messages[-1]["_verification"]["risk"] == "high"

    @pytest.mark.asyncio
    async def test_no_rag_no_verification(self, tmp_path):
        model = FakeModelPort([_text("direct answer")])
        verifier = FakeVerifier()
        runner = _make_runner(model, _make_registry(), hallucination_verifier=verifier)
        session = Session(workspace=tmp_path)

        streamed: list[str] = []
        answer = await runner.run_turn(session, "hi", on_text=streamed.append)
        assert answer == "direct answer"
        assert verifier.calls == []  # 未用 RAG 不触发校验
        assert streamed == ["direct answer"]


# ── registry 结构化错误 + timeout ─────────────────────────────


class TestRegistryStructuredErrors:
    @pytest.mark.asyncio
    async def test_exception_returns_structured_error(self):
        async def boom(args, **kwargs):
            raise ValueError("kaboom")

        registry = _make_registry({"flaky": boom})
        result = await registry.execute("flaky", {})
        parsed = json.loads(result)
        assert parsed["ok"] is False
        assert parsed["error_code"] == "UNKNOWN_ERROR"
        assert parsed["retryable"] is False
        assert "kaboom" in parsed["error"]

    @pytest.mark.asyncio
    async def test_unknown_tool_structured(self):
        registry = _make_registry()
        parsed = json.loads(await registry.execute("nope", {}))
        assert parsed["error_code"] == "UNKNOWN_TOOL"
        assert parsed["ok"] is False

    @pytest.mark.asyncio
    async def test_timeout_seconds_enforced(self):
        async def slow(args, **kwargs):
            await __import__("asyncio").sleep(5)
            return "late"

        registry = _make_registry({"slow": slow})
        tool = registry.get_tool("slow")
        tool.timeout_seconds = 0.05
        parsed = json.loads(await registry.execute("slow", {}))
        assert parsed["error_code"] == "TIMEOUT"
        assert parsed["retryable"] is True


# ── ModelPort ────────────────────────────────────────────────


class StreamingFakeChatModel:
    """支持 astream 的最小 BaseChatModel 替身：逐字符流式输出预置 AIMessage。"""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self._i = 0

    def bind_tools(self, tools, **kwargs):
        return self

    async def astream(self, messages, **kwargs):
        from langchain_core.messages import AIMessageChunk

        response = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        if getattr(response, "tool_calls", None):
            # 模拟真实模型：文本与 tool_call 同时流式输出
            text = response.text() if hasattr(response, "text") else str(response)
            for ch in text:
                yield AIMessageChunk(content=ch)
            for tc in response.tool_calls:
                args_json = json.dumps(tc["args"], ensure_ascii=False)
                # 模拟真实流式：id/name 首帧，args 分两帧
                yield AIMessageChunk(content="", tool_call_chunks=[{
                    "name": tc["name"], "args": args_json[:len(args_json) // 2],
                    "id": tc["id"], "index": 0, "type": "tool_call_chunk",
                }])
                yield AIMessageChunk(content="", tool_call_chunks=[{
                    "name": None, "args": args_json[len(args_json) // 2:],
                    "id": None, "index": 0, "type": "tool_call_chunk",
                }])
            yield AIMessageChunk(content="", response_metadata={"finish_reason": "tool_calls"})
        else:
            text = response.text() if hasattr(response, "text") else str(response)
            for ch in text:
                yield AIMessageChunk(content=ch)
            yield AIMessageChunk(content="", response_metadata={"finish_reason": "stop"})


class TestModelPorts:
    def test_legacy_port_wraps_llm_client(self):
        from unittest.mock import AsyncMock

        from novare.llm_client import LLMClient, LLMResponse

        client = LLMClient.__new__(LLMClient)  # 不建立真实连接
        client.collect_stream = AsyncMock(return_value=LLMResponse(
            content="hi", tool_calls=[], stop_reason="stop",
            usage={"prompt_tokens": 1}, reasoning_content="thinking",
        ))
        port = build_model_port(port_name="auto", llm_client=client)
        assert isinstance(port, LegacyModelPort)

        import asyncio

        result = asyncio.run(port.astream_collect([{"role": "user", "content": "x"}]))
        assert result.content == "hi"
        assert result.reasoning_content == "thinking"

    @pytest.mark.asyncio
    async def test_langchain_port_aggregates_tool_calls(self):
        from langchain_core.messages import AIMessage

        model = StreamingFakeChatModel(responses=[
            AIMessage(content="let me search", tool_calls=[
                {"name": "search_papers", "args": {"query": "llm"}, "id": "call_9"},
            ]),
            AIMessage(content="final answer"),
        ])
        port = LangChainModelPort(model)

        deltas: list[str] = []
        result = await port.astream_collect(
            [{"role": "user", "content": "q"}],
            tools=[{"type": "function", "function": {
                "name": "search_papers", "description": "", "parameters": {},
            }}],
            on_delta=lambda d: deltas.append(d.text),
        )
        # 单次 astream_collect 只消费第一个响应：文本与 tool_calls 并存
        assert result.content == "let me search"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search_papers"
        assert result.tool_calls[0].arguments == {"query": "llm"}
        assert result.stop_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_langchain_port_via_graph_runner(self, tmp_path):
        """LangChainModelPort 端到端：FakeChatModel 驱动 GraphRunner。"""
        from langchain_core.messages import AIMessage

        model = StreamingFakeChatModel(responses=[AIMessage(content="Hello from LC!")])
        port = LangChainModelPort(model)
        runner = GraphRunner(
            model=port,
            tool_registry=_make_registry(),
            system_prompt="test",
            auto_compact_threshold=0,
        )
        session = Session(workspace=tmp_path)
        answer = await runner.run_turn(session, "hi")
        assert answer == "Hello from LC!"


# ── state reducer ─────────────────────────────────────────────


class TestMessageReducer:
    def test_append(self):
        assert merge_messages([{"role": "user", "content": "a"}],
                              [{"role": "assistant", "content": "b"}]) == [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]

    def test_replace_sentinel(self):
        replaced = merge_messages(
            [{"role": "user", "content": "old"}] * 3,
            ReplaceMessages([{"role": "user", "content": "new"}]),
        )
        assert replaced == [{"role": "user", "content": "new"}]

    def test_none_new_keeps_old(self):
        old = [{"role": "user", "content": "a"}]
        assert merge_messages(old, None) == old
