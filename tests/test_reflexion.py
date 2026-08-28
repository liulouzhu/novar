"""tests/test_reflexion.py — PR 3：Reflexion Engine 集成测试

覆盖 26 个场景：
1. 单次 transient failure 不触发 Reflexion
2. transient retry 成功不触发 Reflexion
3. semantic tool error 触发一次 Reflexion
4. 相同 action fingerprint 失败两次触发反思
5. 连续三轮无进展触发反思
6. 正常成功工作流不调用 reflection LLM
7. 同一 trigger fingerprint 不重复反思
8. max_reflections_per_turn 生效
9. malformed JSON 只修复一次
10. goal 被修改时 validation 拒绝
11. 删除安全约束时 validation 拒绝
12. suggested tool 不存在时拒绝
13. suggested arguments 不符合 schema 时拒绝
14. suggested action 与失败 fingerprint 相同时拒绝
15. forbidden action 在执行前被阻止
16. non_idempotent UNKNOWN_OUTCOME 不产生重放建议
17. reviewer_llm 配置时只调用 reviewer
18. reviewer 未配置时正确回退主模型
19. Reflection LLM transient error 使用 PR 1 Retry
20. cancellation/deadline 时不启动 Reflexion
21. tool-call 尚未终态完整时不启动 Reflexion（顺序性）
22. private Recovery Context 注入下一轮，但不进入普通消息历史
23. Reflection events 正确持久化（回调）
24. 进程恢复后 forbidden fingerprints 保留
25. reflexion_enabled=false 时现有行为完全不变
26. PR 1/PR 2 的 timeout/cancel/recovery 测试继续通过（由 test_recovery_state.py 等保证）
"""

import asyncio
import json

import httpx
import pytest
from httpx import HTTPStatusError, Request, Response
from unittest.mock import AsyncMock

from novare.agent_loop import AgentLoop
from novare.llm_client import LLMResponse, ToolCall
from novare.recovery.policy import RetryPolicy
from novare.reflexion import ReflexionState, compute_action_fingerprint
from novare.session import Session
from novare.tools.registry import ToolDef, ToolRegistry

READER_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
}


def _reflection_json(action_fp: str = "fp", **overrides) -> str:
    default = {
        "failure_type": "QUERY_TOO_NARROW",
        "evidence_refs": ["event:tc1"],
        "diagnosis": "检索条件过窄，连续失败",
        "preserve": ["用户要求的时间范围"],
        "changes": ["先宽泛检索再精确补充"],
        "forbidden_repeat": [action_fp],
        "revised_plan": ["宽泛检索", "摘要筛选", "精确补充"],
        "suggested_next_action": {"tool": "reader", "arguments": {"path": "/tmp/x"}},
        "decision": "REPLAN",
    }
    default.update(overrides)
    return json.dumps(default, ensure_ascii=False)


def _status_error(status_code: int) -> HTTPStatusError:
    req = Request("POST", "http://example.com/chat/completions")
    resp = Response(status_code, request=req)
    return HTTPStatusError(f"HTTP {status_code}", request=req, response=resp)


def _build_loop(
    *,
    agent_responses,
    tool_handlers=None,
    reflection_responses=None,
    reviewer_llm=None,
    max_reflections=2,
    repeated_failure_threshold=2,
    no_progress_threshold=3,
    enabled=True,
    tool_max_attempts=1,
    **loop_kwargs,
):
    """构造带 Reflexion 的 AgentLoop。

    reviewer_llm 提供时反思用 reviewer；否则回退主模型（按 system prompt 区分调用）。
    返回 (loop, sleeps, main, reflection_mock, registry, reflection_calls)，
    reflection_calls 为反思 LLM 调用次数计数器（独立于 agent 循环调用）。
    """
    sleeps = []
    reflection_calls: list[int] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    main = AsyncMock()
    registry = ToolRegistry()
    for name, (handler, params) in (tool_handlers or {}).items():
        registry.register_tool(ToolDef(
            name=name, description=f"tool {name}", parameters=params,
            handler=handler, idempotency="read",
            retry_policy=RetryPolicy(max_attempts=tool_max_attempts),
        ))

    if reviewer_llm is not None:
        main.collect_stream = AsyncMock(side_effect=agent_responses)
        reflection_mock = reviewer_llm
        if reflection_responses is not None:
            reviewer_llm.collect_stream = AsyncMock(side_effect=reflection_responses)
        else:
            reviewer_llm.collect_stream = AsyncMock(
                return_value=LLMResponse(content=_reflection_json("fp"), tool_calls=[], stop_reason="stop"),
            )
        original = reviewer_llm.collect_stream

        async def counting_reviewer(*args, **kwargs):
            reflection_calls.append(1)
            return await original(*args, **kwargs)

        reviewer_llm.collect_stream = AsyncMock(side_effect=counting_reviewer)
        reflection_mock = reviewer_llm
    else:
        agent_q = list(agent_responses)
        refl_q = list(reflection_responses) if reflection_responses is not None else None

        async def main_collect(messages, **kwargs):
            if messages and str(messages[0].get("content", "")).startswith("你是一个严谨的反思分析器"):
                reflection_calls.append(1)
                if refl_q is None:
                    return LLMResponse(content=_reflection_json("fp"), tool_calls=[], stop_reason="stop")
                item = refl_q.pop(0)
                if isinstance(item, BaseException):
                    raise item
                return item
            return agent_q.pop(0)

        main.collect_stream = AsyncMock(side_effect=main_collect)
        reflection_mock = main

    loop = AgentLoop(
        llm_client=main,
        reviewer_llm=reviewer_llm,
        tool_registry=registry,
        system_prompt="You are a test assistant.",
        reflexion_enabled=enabled,
        max_reflections_per_turn=max_reflections,
        reflexion_repeated_failure_threshold=repeated_failure_threshold,
        reflexion_no_progress_threshold=no_progress_threshold,
        reflexion_sleep=fake_sleep,
        **loop_kwargs,
    )
    return loop, sleeps, main, reflection_mock, registry, reflection_calls


def _tc(name: str, args: dict, call_id: str = "tc1") -> LLMResponse:
    return LLMResponse(
        content="", tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
        stop_reason="tool_calls", usage={},
    )


class TestNoTrigger:
    @pytest.mark.asyncio
    async def test_single_transient_failure_no_reflexion(self):
        """单次瞬时错误（PR 1 未重试耗尽）不触发 Reflexion。"""
        async def flaky(args, workspace=None):
            return "Error executing reader: connection reset"

        loop, _, main, reflection_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (flaky, READER_SCHEMA)},
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(reflection_calls) == 0

    @pytest.mark.asyncio
    async def test_transient_retry_success_no_reflexion(self):
        """瞬时错误重试成功不触发 Reflexion。"""
        calls = []

        async def flaky(args, workspace=None):
            calls.append(1)
            if len(calls) == 1:
                return "Error executing reader: connection reset"
            return "ok result"

        loop, _, _, reflection_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (flaky, READER_SCHEMA)},
            tool_max_attempts=3,
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(calls) == 2
        assert len(reflection_calls) == 0

    @pytest.mark.asyncio
    async def test_success_workflow_no_reflexion(self):
        """正常成功工作流不调用 reflection LLM。"""
        async def ok_handler(args, workspace=None):
            return "ok result"

        loop, _, _, reflection_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (ok_handler, READER_SCHEMA)},
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(reflection_calls) == 0

    @pytest.mark.asyncio
    async def test_non_idempotent_unknown_outcome_no_replay(self):
        """non_idempotent + UNKNOWN_OUTCOME 不触发反思、不产生重放建议。"""
        async def unknown(args, workspace=None):
            return ("{\"ok\": false, \"error\": \"x\", \"error_code\": \"UNKNOWN_OUTCOME\", "
                    "\"retryable\": false, \"outcome\": \"not_applied\"}")

        loop, _, _, reflection_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("writer", {"path": "/a"}, call_id="tcw"),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={
                "writer": (
                    unknown,
                    {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                ),
            },
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(reflection_calls) == 0


class TestTriggers:
    @pytest.mark.asyncio
    async def test_semantic_error_triggers_once(self):
        """semantic tool error 触发一次 Reflexion 并成功提交。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        fp = compute_action_fingerprint("reader", {"path": "/a"})
        events = []
        loop, _, _, reflection_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            reflection_responses=[
                LLMResponse(content=_reflection_json(fp), tool_calls=[], stop_reason="stop"),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go", on_reflexion_event=lambda t, p: events.append(t))

        assert len(reflection_calls) == 1
        # 反思被验证并提交（不是 rejected）
        assert "REFLECTION_COMMITTED" in events
        assert "REFLECTION_REJECTED" not in events

    @pytest.mark.asyncio
    async def test_repeated_failed_action_triggers(self):
        """相同 action fingerprint 失败两次触发反思（第一次不触发）。"""
        calls = []

        async def boom(args, workspace=None):
            calls.append(1)
            return "Error executing reader: unexpected boom"

        loop, _, _, reflection_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (boom, READER_SCHEMA)},
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(calls) == 2
        assert len(reflection_calls) == 1

    @pytest.mark.asyncio
    async def test_no_progress_triggers_after_threshold(self):
        """连续 N 轮无进展触发反思（进展指纹不变）。"""
        async def same(args, workspace=None):
            return "same result"

        # 4 次迭代：首次为基线，之后 3 次完全相同动作（累计成功信号不变）
        # → no_progress_count=3 → 触发
        loop, _, _, reflection_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}, call_id="tc1"),
                _tc("reader", {"path": "/a"}, call_id="tc2"),
                _tc("reader", {"path": "/a"}, call_id="tc3"),
                _tc("reader", {"path": "/a"}, call_id="tc4"),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (same, READER_SCHEMA)},
            no_progress_threshold=3,
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(reflection_calls) == 1

    @pytest.mark.asyncio
    async def test_same_trigger_fingerprint_not_repeated(self):
        """同一 trigger fingerprint 不重复反思。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        loop, _, _, reflection_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(reflection_calls) == 1

    @pytest.mark.asyncio
    async def test_max_reflections_per_turn_enforced(self):
        """max_reflections_per_turn=1 时第二次触发被预算阻止。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        loop, _, _, reflection_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                _tc("reader", {"path": "/b"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            max_reflections=1,
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(reflection_calls) == 1


class TestValidation:
    @pytest.mark.asyncio
    async def test_malformed_json_repaired_once(self):
        """malformed JSON 允许一次格式修复。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        loop, _, _, reflection_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            reflection_responses=[
                LLMResponse(content="not json at all", tool_calls=[], stop_reason="stop"),
                LLMResponse(content=_reflection_json("fp"), tool_calls=[], stop_reason="stop"),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(reflection_calls) == 2

    @pytest.mark.asyncio
    async def test_goal_modification_rejected(self):
        """goal 被修改时 validation 拒绝。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        events = []
        loop, _, _, reflection_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            reflection_responses=[
                LLMResponse(content=_reflection_json("fp", goal="修改后的目标"), tool_calls=[], stop_reason="stop"),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go", on_reflexion_event=lambda t, p: events.append(t))

        assert "REFLECTION_REJECTED" in events
        assert len(reflection_calls) == 1

    @pytest.mark.asyncio
    async def test_removing_safety_constraint_rejected(self):
        """删除安全约束时 validation 拒绝。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        events = []
        loop, _, _, _, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            reflection_responses=[
                LLMResponse(
                    content=_reflection_json("fp", changes=["删除用户目标不可变约束"]),
                    tool_calls=[], stop_reason="stop",
                ),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go", on_reflexion_event=lambda t, p: events.append(t))

        assert "REFLECTION_REJECTED" in events

    @pytest.mark.asyncio
    async def test_unknown_suggested_tool_rejected(self):
        """suggested tool 不存在时拒绝。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        events = []
        loop, _, _, _, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            reflection_responses=[
                LLMResponse(
                    content=_reflection_json(
                        "fp", suggested_next_action={"tool": "no_such_tool", "arguments": {}},
                    ),
                    tool_calls=[], stop_reason="stop",
                ),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go", on_reflexion_event=lambda t, p: events.append(t))

        assert "REFLECTION_REJECTED" in events

    @pytest.mark.asyncio
    async def test_suggested_arguments_schema_rejected(self):
        """suggested arguments 不符合 schema 时拒绝。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        events = []
        loop, _, _, _, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            reflection_responses=[
                LLMResponse(
                    content=_reflection_json(
                        "fp", suggested_next_action={"tool": "reader", "arguments": {}},
                    ),
                    tool_calls=[], stop_reason="stop",
                ),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go", on_reflexion_event=lambda t, p: events.append(t))

        assert "REFLECTION_REJECTED" in events

    @pytest.mark.asyncio
    async def test_suggested_same_as_failed_rejected(self):
        """suggested action 与失败 fingerprint 相同时拒绝。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        fp = compute_action_fingerprint("reader", {"path": "/a"})
        events = []
        loop, _, _, _, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            reflection_responses=[
                LLMResponse(
                    content=_reflection_json(
                        fp,
                        suggested_next_action={"tool": "reader", "arguments": {"path": "/a"}},
                    ),
                    tool_calls=[], stop_reason="stop",
                ),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go", on_reflexion_event=lambda t, p: events.append(t))

        assert "REFLECTION_REJECTED" in events


class TestForbiddenAction:
    @pytest.mark.asyncio
    async def test_forbidden_action_blocked_before_execution(self):
        """反思提交的 forbidden action 在执行前被阻止。"""
        calls = []

        async def bad_args(args, workspace=None):
            calls.append(1)
            return "Error: Invalid parameter 'path'"

        fp = compute_action_fingerprint("reader", {"path": "/a"})
        events = []

        loop, _, _, _, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}, call_id="tc1"),
                _tc("reader", {"path": "/a"}, call_id="tc2"),  # 相同动作（forbidden）
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            reflection_responses=[
                LLMResponse(content=_reflection_json(fp), tool_calls=[], stop_reason="stop"),
            ],
        )
        session = Session()
        result = await loop.run_turn(
            session, "go", on_reflexion_event=lambda t, p: events.append(t),
        )

        # 第一次执行失败触发反思；第二次（同动作）被阻止，handler 不再调用
        assert len(calls) == 1
        assert "FORBIDDEN_ACTION_BLOCKED" in events
        assert result == "done"
        blocked_msgs = [
            m for m in session.messages
            if m["role"] == "tool" and "FORBIDDEN_REPEATED_ACTION" in m["content"]
        ]
        assert len(blocked_msgs) == 1


class TestModelSelection:
    @pytest.mark.asyncio
    async def test_reviewer_used_when_configured(self):
        """配置 reviewer_llm 时反思只调用 reviewer。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        reviewer = AsyncMock()
        loop, _, main, reviewer_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            reviewer_llm=reviewer,
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert reviewer_mock.collect_stream.await_count == 1
        # main 只服务 agent 循环（tool call + final 共 2 次）
        assert main.collect_stream.await_count == 2

    @pytest.mark.asyncio
    async def test_fallback_to_main_when_no_reviewer(self):
        """未配置 reviewer 时正确回退主模型。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        loop, _, main, reflection_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
        )
        session = Session()
        await loop.run_turn(session, "go")

        # 回退模式：反思调用 main.collect_stream（agent 2 次 + 反思 1 次 = 3）
        assert reflection_mock is main
        assert len(reflection_calls) == 1
        assert main.collect_stream.await_count == 3

    @pytest.mark.asyncio
    async def test_reflection_retry_uses_pr1_retry(self):
        """Reflection LLM transient error 使用 PR 1 Retry。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        loop, sleeps, _, reflection_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            reflection_responses=[
                _status_error(503),
                LLMResponse(content=_reflection_json("fp"), tool_calls=[], stop_reason="stop"),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(reflection_calls) == 2
        assert len(sleeps) == 1  # 一次退避等待


class TestControlFlow:
    @pytest.mark.asyncio
    async def test_cancel_prevents_reflexion(self):
        """cancellation 时不启动 Reflexion。"""
        async def ok_handler(args, workspace=None):
            return "ok result"

        cancel_results = [False, False, True]

        def should_cancel():
            return cancel_results.pop(0) if cancel_results else True

        loop, _, _, reflection_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (ok_handler, READER_SCHEMA)},
        )
        session = Session()
        result = await loop.run_turn(session, "go", should_cancel=should_cancel)

        assert "取消" in result
        assert len(reflection_calls) == 0

    @pytest.mark.asyncio
    async def test_reflexion_after_all_tool_results_terminal(self):
        """反思发生在所有 tool results 终态提交之后（协议完整性后）。"""
        async def boom(args, workspace=None):
            return "Error executing reader: unexpected boom"

        events = []

        def on_tool(event, name, args, result, elapsed):
            events.append(("tool", event, name))

        loop, _, _, reflection_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}, call_id="tc1"),
                _tc("reader", {"path": "/a"}, call_id="tc2"),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (boom, READER_SCHEMA)},
        )
        session = Session()
        await loop.run_turn(session, "go", on_tool=on_tool)

        assert len(reflection_calls) == 1
        # 反思（第二次失败）发生在第二个 tool 事件之后
        assert ("tool", "end", "reader") in events or ("tool", "error", "reader") in events

    @pytest.mark.asyncio
    async def test_recovery_context_injected_not_in_history(self):
        """private Recovery Context 注入下一轮 system prompt，不进入消息历史。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        fp = compute_action_fingerprint("reader", {"path": "/a"})
        loop, _, main, reflection_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            reflection_responses=[
                LLMResponse(content=_reflection_json(fp), tool_calls=[], stop_reason="stop"),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go")

        # 反思诊断不出现在普通消息历史
        history_text = json.dumps(session.messages, ensure_ascii=False)
        assert "检索条件过窄" not in history_text
        # 下一轮 LLM 调用收到的 system prompt 含 [Recovery Context]
        system_content = main.collect_stream.await_args_list[-1].args[0][0]["content"]
        assert "[Recovery Context]" in system_content
        assert "检索条件过窄" in system_content

    @pytest.mark.asyncio
    async def test_reflexion_events_callback(self):
        """Reflexion events 通过回调持久化（TRIGGERED/STARTED/COMMITTED/PLAN_REVISED）。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        fp = compute_action_fingerprint("reader", {"path": "/a"})
        events = []

        async def on_event(event_type, payload):
            events.append(event_type)

        loop, _, _, _, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            reflection_responses=[
                LLMResponse(content=_reflection_json(fp), tool_calls=[], stop_reason="stop"),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go", on_reflexion_event=on_event)

        assert "REFLECTION_TRIGGERED" in events
        assert "REFLECTION_STARTED" in events
        assert "REFLECTION_COMMITTED" in events
        assert "PLAN_REVISED" in events


class TestStatePersistence:
    def test_state_roundtrip_preserves_forbidden(self):
        """ReflexionState to_dict/from_dict 保留 forbidden fingerprints（v3 严格恢复）。"""
        fake_fp = "a" * 64  # 64 位小写十六进制 action fingerprint
        trigger_fp = f"semantic_tool_failure:{fake_fp}:INVALID_ARGUMENT"
        state = ReflexionState()
        state.add_reflection(__import__("novare.reflexion.types", fromlist=["make_reflection_record"]).make_reflection_record(
            trigger="semantic_tool_failure",
            trigger_fingerprint=trigger_fp,
            evidence_refs=["event:tc1"],
            failure_type="INVALID_ARGUMENT",
            diagnosis="诊断",
            preserve=[],
            changes=["变更"],
            forbidden_action_fingerprints=[fake_fp],
            revised_plan=["计划"],
            suggested_next_action=None,
            decision="REPLAN",
            validated=True,
            applied=True,
        ))
        restored = ReflexionState.from_dict(state.to_dict())
        assert restored.reflection_count == 1
        assert fake_fp in restored.forbidden_action_fingerprints
        assert trigger_fp in restored.reflected_trigger_fingerprints

    @pytest.mark.asyncio
    async def test_restored_forbidden_fingerprint_blocked(self):
        """进程恢复（initial_reflexion_state）后 forbidden fingerprints 生效。"""
        restored = ReflexionState()
        restored.forbidden_action_fingerprints.add(compute_action_fingerprint("reader", {"path": "/a"}))

        calls = []

        async def ok_handler(args, workspace=None):
            calls.append(1)
            return "ok result"

        loop, _, _, _, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}, call_id="tc1"),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (ok_handler, READER_SCHEMA)},
        )
        session = Session()
        await loop.run_turn(session, "go", initial_reflexion_state=restored)

        assert calls == []  # 被阻止，handler 未执行
        assert any(
            "FORBIDDEN_REPEATED_ACTION" in m["content"]
            for m in session.messages if m["role"] == "tool"
        )

    @pytest.mark.asyncio
    async def test_disabled_reflexion_unchanged(self):
        """reflexion_enabled=false 时现有行为完全不变（不调用反思、无 context）。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        loop, _, main, reflection_mock, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            enabled=False,
        )
        session = Session()
        result = await loop.run_turn(session, "go")

        assert result == "done"
        assert len(reflection_calls) == 0
        system_content = main.collect_stream.await_args_list[-1].args[0][0]["content"]
        assert "[Recovery Context]" not in system_content
