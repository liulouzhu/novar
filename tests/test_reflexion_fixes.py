"""tests/test_reflexion_fixes.py — PR 3 code review 修复验证

覆盖 10 组场景：
1. Reflexion LLM 永久 503：重试耗尽 → REFLECTION_FAILED，run_turn 不抛，主模型继续
2. 反思 collect_stream 永不返回：reflexion_timeout 取消，无后台 task 泄漏
3. CancelledError 立即传播，不记录为普通 failure
4. 成功 conflict=true 可触发反思；成功动作不进 forbidden，后续可再次执行
5. NO_PROGRESS：失败 batch 不误判进展；同停滞阶段一次；真实进展后再停滞第二次
6. Validator：suggested 在历史/本次 forbidden 集合拒绝；缺触发指纹拒绝；conflict/no-progress 不强制
7. JSON Schema：array items 类型错误拒绝；嵌套错误拒绝；合法嵌套通过
8. Web 恢复：显式 recovery_run_id 恢复；forbidden 恢复后阻止；跨用户拒绝；malformed 降级
9. Recovery Context：suggested_next_action 注入 system，不进 history，不自动执行
10. 共享 RetryBudget：Reflexion transport retry 消耗同一预算；耗尽后安全降级
"""

import asyncio
import json
import uuid

import pytest
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


def _reflection_json(action_fp: str, **overrides) -> str:
    default = {
        "failure_type": "QUERY_TOO_NARROW",
        "evidence_refs": ["event:tc1"],
        "diagnosis": "诊断结论",
        "preserve": ["用户要求"],
        "changes": ["变更"],
        "forbidden_repeat": [action_fp],
        "revised_plan": ["计划"],
        "suggested_next_action": {"tool": "reader", "arguments": {"path": "/x"}},
        "decision": "REPLAN",
    }
    default.update(overrides)
    return json.dumps(default, ensure_ascii=False)


def _status_error(status_code: int):
    from httpx import HTTPStatusError, Request, Response

    req = Request("POST", "http://example.com/chat/completions")
    resp = Response(status_code, request=req)
    return HTTPStatusError(f"HTTP {status_code}", request=req, response=resp)


def _tc(name: str, args: dict, call_id: str = "tc1") -> LLMResponse:
    return LLMResponse(
        content="", tool_calls=[ToolCall(id=call_id, name=name, arguments=args)],
        stop_reason="tool_calls", usage={},
    )


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
    max_retries_per_turn=6,
    **loop_kwargs,
):
    """构造带 Reflexion 的 AgentLoop（reviewer 独立 / 回退主模型）。"""
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
            retry_policy=RetryPolicy(max_attempts=1),
        ))

    if reviewer_llm is not None:
        main.collect_stream = AsyncMock(side_effect=agent_responses)
        reflection_mock = reviewer_llm
        if reflection_responses is not None:
            reviewer_llm.collect_stream = AsyncMock(side_effect=reflection_responses)
        else:
            reviewer_llm.collect_stream = AsyncMock(
                return_value=LLMResponse(content=_reflection_json("a" * 64), tool_calls=[], stop_reason="stop"),
            )
        original = reviewer_llm.collect_stream

        async def counting_reviewer(*args, **kwargs):
            reflection_calls.append(1)
            return await original(*args, **kwargs)

        reviewer_llm.collect_stream = AsyncMock(side_effect=counting_reviewer)
    else:
        agent_q = list(agent_responses)
        refl_q = list(reflection_responses) if reflection_responses is not None else None

        async def main_collect(messages, **kwargs):
            if messages and str(messages[0].get("content", "")).startswith("你是一个严谨的反思分析器"):
                reflection_calls.append(1)
                if refl_q is None:
                    return LLMResponse(content=_reflection_json("a" * 64), tool_calls=[], stop_reason="stop")
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
        max_retries_per_turn=max_retries_per_turn,
        **loop_kwargs,
    )
    return loop, sleeps, main, reflection_mock, registry, reflection_calls


class TestModelFaultIsolation:
    @pytest.mark.asyncio
    async def test_permanent_503_does_not_fail_turn(self):
        """反思 LLM 永久 503：重试耗尽 → REFLECTION_FAILED，run_turn 不抛，主模型继续。"""
        calls = []

        async def bad_args(args, workspace=None):
            calls.append(1)
            return "Error: Invalid parameter 'path'"

        events = []
        loop, _, _, _, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            reflection_responses=[
                _status_error(503), _status_error(503), _status_error(503),
            ],
        )
        session = Session()
        result = await loop.run_turn(session, "go", on_reflexion_event=lambda t, p: events.append(t))

        assert result == "done"  # 主 Agent 继续并返回最终答案
        assert "REFLECTION_FAILED" in events
        assert "REFLECTION_STARTED" in events
        # 已提交 tool result 保留
        assert len([m for m in session.messages if m["role"] == "tool"]) == 1
        # 重试 3 次（max_attempts=3）后失败
        assert len(reflection_calls) == 3

    @pytest.mark.asyncio
    async def test_hanging_reflection_times_out(self):
        """反思 collect_stream 永不返回：reflexion_timeout 取消，不等待 turn_timeout。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        async def hang_stream(messages, **kwargs):
            await asyncio.sleep(3600)

        main = AsyncMock()
        main.collect_stream = AsyncMock(side_effect=[
            _tc("reader", {"path": "/a"}),
            LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
        ])
        reviewer = AsyncMock()
        reviewer.collect_stream = AsyncMock(side_effect=hang_stream)
        registry = ToolRegistry()
        registry.register_tool(ToolDef(
            name="reader", description="r", parameters=READER_SCHEMA,
            handler=bad_args, idempotency="read",
            retry_policy=RetryPolicy(max_attempts=1),
        ))
        events = []
        loop = AgentLoop(
            llm_client=main,
            reviewer_llm=reviewer,
            tool_registry=registry,
            system_prompt="You are a test assistant.",
            reflexion_enabled=True,
            reflexion_timeout=0.1,
            reflexion_sleep=lambda d: asyncio.sleep(0),
            max_reflections_per_turn=2,
        )
        session = Session()
        before = set(asyncio.all_tasks())
        result = await loop.run_turn(session, "go", on_reflexion_event=lambda t, p: events.append(t))
        after = set(asyncio.all_tasks())

        assert result == "done"
        assert "REFLECTION_FAILED" in events
        assert after == before  # 无后台 task 泄漏

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_not_recorded(self):
        """反思模型抛 CancelledError：立即传播，不记录为普通 failure。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        events = []
        reviewer = AsyncMock()
        loop, _, _, _, _, _ = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            reviewer_llm=reviewer,
            reflection_responses=[asyncio.CancelledError()],
        )
        session = Session()
        with pytest.raises(asyncio.CancelledError):
            await loop.run_turn(session, "go", on_reflexion_event=lambda t, p: events.append(t))
        assert "REFLECTION_FAILED" not in events
        assert "REFLECTION_REJECTED" not in events

    @pytest.mark.asyncio
    async def test_reflexion_retry_consumes_shared_budget(self):
        """Reflexion transport retry 消耗共享 turn RetryBudget；耗尽后安全降级。"""
        calls = []

        async def bad_args(args, workspace=None):
            calls.append(1)
            return "Error: Invalid parameter 'path'"

        fp_a = compute_action_fingerprint("reader", {"path": "/a"})
        events = []
        loop, _, _, _, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}, call_id="tc1"),
                _tc("reader", {"path": "/b"}, call_id="tc2"),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            reflection_responses=[
                _status_error(503),
                LLMResponse(content=_reflection_json(fp_a), tool_calls=[], stop_reason="stop"),
                _status_error(503),
            ],
            max_retries_per_turn=1,
        )
        session = Session()
        result = await loop.run_turn(session, "go", on_reflexion_event=lambda t, p: events.append(t))

        assert result == "done"
        # 反思1：503+valid（2 次调用，1 次 transport retry 消耗唯一预算）
        # 反思2：503 后预算耗尽不再重试（1 次调用）→ REFLECTION_FAILED
        assert len(reflection_calls) == 3
        assert "REFLECTION_FAILED" in events


class TestConflictSuccess:
    @pytest.mark.asyncio
    async def test_conflict_success_triggers_without_forbidden(self):
        """成功 conflict=true 可触发反思；成功动作不进 forbidden，后续可再次执行。"""
        handler_calls = []

        async def conflicting(args, workspace=None):
            handler_calls.append(1)
            return json.dumps({
                "ok": True,
                "summary": "found",
                "data": {"conflict": True, "conflict_detail": "两篇论文结论矛盾"},
            })

        events = []
        loop, _, _, _, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}, call_id="tc1"),
                _tc("reader", {"path": "/a"}, call_id="tc2"),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (conflicting, READER_SCHEMA)},
            reflection_responses=[
                LLMResponse(
                    content=_reflection_json(
                        "a" * 64,
                        forbidden_repeat=[],
                        suggested_next_action={"tool": "reader", "arguments": {"path": "/a"}},
                    ),
                    tool_calls=[], stop_reason="stop",
                ),
            ],
        )
        session = Session()
        result = await loop.run_turn(session, "go", on_reflexion_event=lambda t, p: events.append(t))

        assert result == "done"
        assert "REFLECTION_COMMITTED" in events
        # 成功动作未被禁止：第二次执行未被 FORBIDDEN_ACTION_BLOCKED 阻止
        assert "FORBIDDEN_ACTION_BLOCKED" not in events
        assert len(handler_calls) == 2
        assert len(reflection_calls) == 2


class TestNoProgressFixes:
    @pytest.mark.asyncio
    async def test_failed_batch_not_misjudged_as_progress(self):
        """成功 batch 后的失败 batch 不被摘要消失误判为进展。"""
        async def flaky(args, workspace=None):
            if args.get("path") == "/a":
                return "first ok"
            return "Error executing reader: boom"

        loop, _, _, _, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}, call_id="tc1"),
                _tc("reader", {"path": "/b"}, call_id="tc2"),
                _tc("reader", {"path": "/c"}, call_id="tc3"),
                _tc("reader", {"path": "/d"}, call_id="tc4"),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (flaky, READER_SCHEMA)},
            no_progress_threshold=3,
        )
        session = Session()
        await loop.run_turn(session, "go")

        # 失败 batch 不算进展 → 第 4 次迭代触发 NO_PROGRESS 反思
        assert len(reflection_calls) == 1

    @pytest.mark.asyncio
    async def test_progress_after_stall_allows_second_reflection(self):
        """真实进展后再停滞，允许触发第二次 NO_PROGRESS 反思。"""
        async def flaky(args, workspace=None):
            path = args.get("path")
            if path == "/a":
                return "first ok"
            if path == "/b":
                return "stall b"
            return "stall c"

        # 停滞1：t1 /a 基线 → t2 /b 新信号（进展）→ t3-t5 /b 相同重复 → 触发1
        # 进展：t6 /c 新信号 → t7-t9 /c 相同重复 → 触发2
        loop, _, _, _, _, reflection_calls = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}, call_id="t1"),
                _tc("reader", {"path": "/b"}, call_id="t2"),
                _tc("reader", {"path": "/b"}, call_id="t3"),
                _tc("reader", {"path": "/b"}, call_id="t4"),
                _tc("reader", {"path": "/b"}, call_id="t5"),
                _tc("reader", {"path": "/c"}, call_id="t6"),
                _tc("reader", {"path": "/c"}, call_id="t7"),
                _tc("reader", {"path": "/c"}, call_id="t8"),
                _tc("reader", {"path": "/c"}, call_id="t9"),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (flaky, READER_SCHEMA)},
            no_progress_threshold=3,
            max_iterations=12,
        )
        session = Session()
        await loop.run_turn(session, "go")

        assert len(reflection_calls) == 2

    def test_pending_changes_not_progress(self):
        """pending 文本变化不算真实进展；重复成功结果不重置。"""
        from novare.reflexion.progress import ProgressTracker, progress_signal_digest

        tracker = ProgressTracker()
        d1 = progress_signal_digest(kind="tool_success", tool="r", action_fingerprint="fp", summary_digest="a" * 64)
        d2 = progress_signal_digest(kind="tool_success", tool="r", action_fingerprint="fp2", summary_digest="a" * 64)
        tracker.update(completed=["c1"], key_findings=["f1"], success_signal_digests=[d1])
        made = tracker.update(completed=["c1"], key_findings=["f1"], success_signal_digests=[d1])
        assert made is False
        made = tracker.update(
            completed=["c1"], key_findings=["f1"],
            success_signal_digests=[d1, d1],
        )
        assert made is False
        made = tracker.update(
            completed=["c1"], key_findings=["f1"],
            success_signal_digests=[d1, d2],
        )
        assert made is True


class TestValidatorForbiddenChecks:
    def _validator(self, **kwargs):
        from novare.reflexion.validator import ReflectionValidator

        registry = ToolRegistry()
        registry.register_tool(ToolDef(
            name="reader", description="r", parameters=READER_SCHEMA,
            handler=None, idempotency="read",
            retry_policy=RetryPolicy(max_attempts=1),
        ))
        defaults = dict(
            tool_registry=registry,
            available_tool_names={"reader"},
            user_goal="goal",
            safety_constraints=[],
            real_event_ids=["tc1"],
            triggering_action_fingerprint=None,
            existing_forbidden_action_fingerprints=set(),
            failed_tool=None,
            failed_arguments=None,
            idempotency="read",
        )
        defaults.update(kwargs)
        return ReflectionValidator(**defaults)

    def test_suggested_in_existing_forbidden_rejected(self):
        fp = compute_action_fingerprint("reader", {"path": "/a"})
        v = self._validator(existing_forbidden_action_fingerprints={fp})
        ok, reason = v.validate({
            "decision": "REPLAN", "diagnosis": "d",
            "failure_type": "QUERY_TOO_NARROW",
            "changes": ["c"], "revised_plan": ["p"],
            "suggested_next_action": {"tool": "reader", "arguments": {"path": "/a"}},
        })
        assert not ok
        assert "existing forbidden" in reason

    def test_suggested_in_forbidden_repeat_rejected(self):
        fp = compute_action_fingerprint("reader", {"path": "/a"})
        v = self._validator()
        ok, reason = v.validate({
            "decision": "REPLAN", "diagnosis": "d",
            "failure_type": "QUERY_TOO_NARROW",
            "changes": ["c"], "revised_plan": ["p"],
            "forbidden_repeat": [fp],
            "suggested_next_action": {"tool": "reader", "arguments": {"path": "/a"}},
        })
        assert not ok
        assert "forbidden_repeat" in reason

    def test_failed_action_trigger_missing_fingerprint_rejected(self):
        fp = compute_action_fingerprint("reader", {"path": "/a"})
        v = self._validator(triggering_action_fingerprint=fp)
        ok, _ = v.validate({
            "decision": "REPLAN", "diagnosis": "d",
            "failure_type": "QUERY_TOO_NARROW",
            "changes": ["c"], "revised_plan": ["p"],
            "forbidden_repeat": [],
        })
        assert not ok

    def test_conflict_no_progress_no_fake_forbidden_required(self):
        v = self._validator(triggering_action_fingerprint=None)
        ok, reason = v.validate({
            "decision": "REPLAN", "diagnosis": "d",
            "failure_type": "QUERY_TOO_NARROW",
            "changes": ["c"], "revised_plan": ["p"],
            "forbidden_repeat": [],
        })
        assert ok, reason

    def test_invalid_fingerprint_format_rejected(self):
        v = self._validator(triggering_action_fingerprint=None)
        ok, reason = v.validate({
            "decision": "REPLAN", "diagnosis": "d",
            "failure_type": "QUERY_TOO_NARROW",
            "changes": ["c"], "revised_plan": ["p"],
            "forbidden_repeat": ["not-a-fingerprint"],
        })
        assert not ok
        assert "invalid action fingerprint" in reason


class TestJsonSchemaItems:
    def test_array_items_type_mismatch_rejected(self):
        from novare.reflexion.validator import validate_arguments_against_schema

        schema = {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
        ok, _ = validate_arguments_against_schema({"tags": [1, 2]}, schema)
        assert not ok
        ok, _ = validate_arguments_against_schema({"tags": ["a", "b"]}, schema)
        assert ok

    def test_nested_array_object_rejected(self):
        from novare.reflexion.validator import validate_arguments_against_schema

        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                        "required": ["x"],
                    },
                },
            },
        }
        ok, _ = validate_arguments_against_schema({"items": [{"x": "not-int"}]}, schema)
        assert not ok
        ok, _ = validate_arguments_against_schema({"items": [{"x": 1}, {"x": 2}]}, schema)
        assert ok
        ok, _ = validate_arguments_against_schema({"items": [{"x": 1}, {"x": "bad"}]}, schema)
        assert not ok

    def test_nested_array_in_array_rejected(self):
        from novare.reflexion.validator import validate_arguments_against_schema

        schema = {
            "type": "object",
            "properties": {
                "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
            },
        }
        ok, _ = validate_arguments_against_schema({"matrix": [[1, 2], ["a"]]}, schema)
        assert not ok
        ok, _ = validate_arguments_against_schema({"matrix": [[1, 2], [3]]}, schema)
        assert ok

    def test_bool_not_integer(self):
        from novare.reflexion.validator import validate_arguments_against_schema

        schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
        ok, _ = validate_arguments_against_schema({"count": True}, schema)
        assert not ok


class TestSuggestedInjection:
    @pytest.mark.asyncio
    async def test_suggested_next_action_injected_not_in_history(self):
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        fp = compute_action_fingerprint("reader", {"path": "/a"})
        loop, _, main, _, _, _ = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            reflection_responses=[
                LLMResponse(
                    content=_reflection_json(
                        fp, suggested_next_action={"tool": "reader", "arguments": {"path": "/x"}},
                    ),
                    tool_calls=[], stop_reason="stop",
                ),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go")

        system_content = main.collect_stream.await_args_list[-1].args[0][0]["content"]
        assert "[Recovery Context]" in system_content
        assert "建议下一步" in system_content
        assert '"tool": "reader"' in system_content
        history_text = json.dumps(session.messages, ensure_ascii=False)
        assert "建议下一步" not in history_text

    @pytest.mark.asyncio
    async def test_suggested_next_action_not_auto_executed(self):
        """suggested_next_action 只是建议，不会被直接执行。"""
        async def bad_args(args, workspace=None):
            return "Error: Invalid parameter 'path'"

        fp = compute_action_fingerprint("reader", {"path": "/a"})
        tool_events = []
        loop, _, _, _, _, _ = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (bad_args, READER_SCHEMA)},
            reflection_responses=[
                LLMResponse(
                    content=_reflection_json(
                        fp, suggested_next_action={"tool": "reader", "arguments": {"path": "/x"}},
                    ),
                    tool_calls=[], stop_reason="stop",
                ),
            ],
        )
        session = Session()
        await loop.run_turn(
            session, "go",
            on_tool=lambda event, name, args, result, elapsed: tool_events.append((event, name)),
        )
        assert tool_events.count(("start", "reader")) == 1


class TestWebRestore:
    async def _seed_recovery_state(self, db_session_factory, user_uuid, session_id, run_id, recovery_data):
        from web.backend.db.models import User
        from web.backend.auth.service import hash_password
        from web.backend.repositories import RecoveryStateRepository, SessionRepository

        async with db_session_factory() as db:
            user = User(
                id=user_uuid,
                username=f"test_{user_uuid.hex[:8]}",
                email=f"test_{user_uuid.hex[:8]}@test.com",
                password_hash=hash_password("pass"),
            )
            db.add(user)
            await db.flush()
            session_repo = SessionRepository(db, user_uuid)
            await session_repo.create(session_id, title="Test")
            await db.commit()

            repo = RecoveryStateRepository(db, user_uuid)
            await repo.upsert(
                session_id=session_id, run_id=run_id, turn_id="t1",
                recovery_data=recovery_data,
                run_status="failed", iteration=2, retry_count=0, schema_version=2,
            )
            await db.commit()

    @pytest.mark.asyncio
    async def test_restore_from_db_and_block_forbidden(self, db_session_factory, monkeypatch):
        import web.backend.agent_service as agent_mod
        from web.backend.agent_service import AgentService

        user_uuid = uuid.uuid4()
        session_id = "restore-session"
        run_id = "restore-run"
        fp = compute_action_fingerprint("reader", {"path": "/a"})

        state = ReflexionState()
        state.forbidden_action_fingerprints.add(fp)
        await self._seed_recovery_state(
            db_session_factory, user_uuid, session_id, run_id,
            {"reflexion_state": state.to_dict()},
        )

        monkeypatch.setattr(agent_mod, "get_session_factory", lambda: db_session_factory)
        svc = AgentService()
        restored = await svc._restore_reflexion_state(session_id, run_id, str(user_uuid))
        assert restored is not None
        assert fp in restored.forbidden_action_fingerprints

        # 恢复的 forbidden 在 AgentLoop 执行前阻止工具调用
        handler_calls = []

        async def ok_handler(args, workspace=None):
            handler_calls.append(1)
            return "ok result"

        loop, _, _, _, _, _ = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}, call_id="tc1"),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (ok_handler, READER_SCHEMA)},
        )
        session = Session()
        await loop.run_turn(session, "go", initial_reflexion_state=restored)
        assert handler_calls == []

    @pytest.mark.asyncio
    async def test_cross_user_restore_rejected(self, db_session_factory, monkeypatch):
        import web.backend.agent_service as agent_mod
        from web.backend.agent_service import AgentService, RecoveryResumeError

        owner_uuid = uuid.uuid4()
        other_uuid = uuid.uuid4()
        state = ReflexionState()
        state.forbidden_action_fingerprints.add("a" * 64)
        await self._seed_recovery_state(
            db_session_factory, owner_uuid, "s1", "r1",
            {"reflexion_state": state.to_dict()},
        )

        monkeypatch.setattr(agent_mod, "get_session_factory", lambda: db_session_factory)
        svc = AgentService()
        # 跨用户恢复必须 fail closed（抛异常，不静默降级）
        with pytest.raises(RecoveryResumeError):
            await svc._restore_reflexion_state("s1", "r1", str(other_uuid))

    @pytest.mark.asyncio
    async def test_malformed_state_fails_closed(self, db_session_factory, monkeypatch):
        import web.backend.agent_service as agent_mod
        from web.backend.agent_service import AgentService, RecoveryResumeError

        user_uuid = uuid.uuid4()
        await self._seed_recovery_state(
            db_session_factory, user_uuid, "s1", "r1",
            {"reflexion_state": {"schema_version": 99, "records": "not-a-list"}},
        )

        monkeypatch.setattr(agent_mod, "get_session_factory", lambda: db_session_factory)
        svc = AgentService()
        # 损坏状态必须 fail closed（抛异常，不返回部分污染状态）
        with pytest.raises(RecoveryResumeError):
            await svc._restore_reflexion_state("s1", "r1", str(user_uuid))
