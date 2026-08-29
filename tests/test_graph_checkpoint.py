"""tests/test_graph_checkpoint.py — 阶段 2：checkpoint 持久化与恢复语义

覆盖（MemorySaver 后端，与 PostgresSaver 仅存储层不同）：
- thread_id 与 RecoveryState 的映射（随快照持久化）
- 中途崩溃后从 checkpoint resume：已完成工具不重放副作用
- 恢复对账 _reconcile_with_ledger 的三路处置（skip / replay / synthesize）
- 正常完成后 checkpoint 清理
"""

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from novare.graph.checkpointer import build_thread_id
from novare.graph.nodes.tools import _reconcile_with_ledger
from novare.graph.adapters.model import ModelResult, ToolCallSpec
from novare.graph.context import RunContext
from novare.graph.runner import GraphRunner
from novare.recovery.state import RecoveryState, ToolCallStatus
from novare.session import Session
from novare.tools.registry import ToolDef, ToolRegistry

from tests.test_graph_runtime import (
    FakeModelPort,
    _make_registry,
    _text,
    _tool_call,
)


class _CrashOnNthModel(FakeModelPort):
    """第 n 次 astream_collect 抛异常（之前的响应正常消费）。"""

    def __init__(self, responses, crash_at: int):
        super().__init__(responses)
        self._crash_at = crash_at
        self._n = 0

    async def astream_collect(self, messages, tools=None, on_delta=None):
        self._n += 1
        if self._n == self._crash_at:
            raise RuntimeError("boom: transport crash mid-turn")
        return await super().astream_collect(messages, tools=tools, on_delta=on_delta)


def _make_runner(model, registry, **kwargs) -> GraphRunner:
    defaults = dict(
        system_prompt="test",
        max_iterations=5,
        turn_timeout=30,
        auto_compact_threshold=0,
        checkpointer=InMemorySaver(),
    )
    defaults.update(kwargs)
    return GraphRunner(model=model, tool_registry=registry, **defaults)


async def _counting_handler(counter: dict, key: str):
    async def _handler(args, **kwargs) -> str:
        counter[key] = counter.get(key, 0) + 1
        return json.dumps({"ok": True, "summary": f"{key} ran"})
    return _handler


# ── thread_id 映射 ────────────────────────────────────────────


class TestThreadIdMapping:
    @pytest.mark.asyncio
    async def test_thread_id_populated_and_persisted(self, tmp_path):
        """run_turn 后 RecoveryState.thread_id 被填充并随快照外发。"""
        model = FakeModelPort([_text("done")])
        runner = _make_runner(model, _make_registry())
        session = Session(session_id="sess-1", workspace=tmp_path)

        snapshots: list[dict] = []
        await runner.run_turn(
            session, "hi",
            tool_context={"user_id": "user-9"},
            on_recovery_state=lambda snap: snapshots.append(snap),
        )
        final = snapshots[-1]
        assert final["thread_id"]
        assert final["thread_id"].startswith("user-9:sess-1:")
        # run_id 与 thread_id 尾段一致（恢复映射的锚点）
        assert final["thread_id"].endswith(":" + final["run_id"])

    def test_build_thread_id_anonymous(self):
        assert build_thread_id(None, "s1", "r1") == "anonymous:s1:r1"


# ── checkpoint 恢复 ──────────────────────────────────────────


class TestCheckpointResume:
    @pytest.mark.asyncio
    async def test_resume_skips_completed_tools(self, tmp_path):
        """工具执行完成后模型调用崩溃 → resume 不重放已完成的工具。"""
        calls: dict = {}
        registry = _make_registry({"search_papers": await _counting_handler(calls, "search_papers")})

        model = _CrashOnNthModel([
            _tool_call("call_1", "search_papers", {"q": "llm"}),
            _text("unused: this call crashes"),
        ], crash_at=2)
        runner = _make_runner(model, registry)
        session = Session(session_id="sess-1", workspace=tmp_path)

        snapshots: list[dict] = []
        with pytest.raises(RuntimeError, match="boom"):
            await runner.run_turn(
                session, "search papers",
                tool_context={"user_id": "u1"},
                on_recovery_state=lambda snap: snapshots.append(snap),
            )

        assert calls.get("search_papers") == 1, "崩溃前工具已执行一次"
        snapshot = snapshots[-1]
        assert snapshot["thread_id"]
        assert snapshot["tool_calls"]["call_1"]["status"] == "completed"

        # ── resume：同一 runner（同一 checkpointer / graph），恢复账本 ──
        runner.model = FakeModelPort([_text("Found 3 papers.")])
        answer = await runner.run_turn(
            session, "search papers",
            tool_context={"user_id": "u1"},
            resume=True,
            thread_id=snapshot["thread_id"],
            recovery_state=RecoveryState.from_dict(snapshot),
        )

        assert answer == "Found 3 papers."
        assert calls.get("search_papers") == 1, "resume 后不得重放已完成的工具"
        # tool result 恰好一条（commit_tool_result_once 幂等）
        tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

    @pytest.mark.asyncio
    async def test_completed_turn_discards_checkpoint(self, tmp_path):
        """正常完成的 turn 清理 checkpoint，不可被 resume 复用。"""
        model = FakeModelPort([_text("done")])
        runner = _make_runner(model, _make_registry())
        session = Session(session_id="sess-2", workspace=tmp_path)

        snapshots: list[dict] = []
        await runner.run_turn(
            session, "hi",
            tool_context={"user_id": "u1"},
            on_recovery_state=lambda snap: snapshots.append(snap),
        )
        thread_id = snapshots[-1]["thread_id"]
        state = runner.get_thread_state(thread_id)
        assert not state.values, "completed 的 thread checkpoint 应被清理"


# ── 恢复对账单元测试 ─────────────────────────────────────────


class TestLedgerReconcile:
    def _make_ctx(self, recovery: RecoveryState, tmp_path) -> RunContext:
        return RunContext(
            model=None,
            tools=None,
            compactor=None,
            session=Session(workspace=tmp_path),
            recovery_state=recovery,
            resumed=True,
        )

    def test_completed_call_skipped(self, tmp_path):
        recovery = RecoveryState()
        recovery.register_tool_calls_batch(
            [{"id": "c1", "name": "search_papers", "arguments": {}}], _make_registry(),
        )
        recovery.mark_executing("c1")
        recovery.mark_completed("c1")

        batch, synthesized = _reconcile_with_ledger(
            [{"id": "c1", "name": "search_papers", "arguments": {}}],
            self._make_ctx(recovery, tmp_path),
        )
        assert batch == []
        assert synthesized == []

    def test_non_idempotent_pending_synthesized(self, tmp_path):
        recovery = RecoveryState()
        recovery.register_tool_calls_batch(
            [{"id": "c1", "name": "write_file", "arguments": {}}], _make_registry(),
        )
        recovery.mark_executing("c1")  # 崩溃点：执行中

        call = {"id": "c1", "name": "write_file", "arguments": {}}
        batch, synthesized = _reconcile_with_ledger(
            [call], self._make_ctx(recovery, tmp_path),
        )
        assert batch == []
        assert len(synthesized) == 1
        assert json.loads(synthesized[0][1])["error_code"] == "UNKNOWN_OUTCOME"

    def test_idempotent_pending_replayed(self, tmp_path):
        registry = _make_registry()
        # 把工具显式声明为幂等读
        registry.register_tool(ToolDef(
            name="rag_query", description="", parameters={}, handler=lambda: None,
            idempotency="read",
        ))
        recovery = RecoveryState()
        recovery.register_tool_calls_batch(
            [{"id": "c1", "name": "rag_query", "arguments": {}}], registry,
        )
        recovery.mark_executing("c1")

        call = {"id": "c1", "name": "rag_query", "arguments": {}}
        batch, synthesized = _reconcile_with_ledger(
            [call], self._make_ctx(recovery, tmp_path),
        )
        assert batch == [call]
        assert synthesized == []
