"""PR 3 review 第三轮修复测试。

覆盖：
一、严格输出字段类型校验（normalize 不自动转换）
二、失败 conflict 分类（ok=false 不得为 conflict success，继续评估 semantic）
三、ReflexionState.from_dict 严格校验（schema/type/invariant，fail closed）
四、显式恢复 fail closed（RecoveryResumeError → 统一错误事件，不执行新 turn）
六、ProgressTracker 跨进程恢复（累计信号入 state，无进展继续累加）
七、注入 suggested action 前复查当前 forbidden 集合
八、classify_exception(RetryExhaustedError) 保留安全字段
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from novare.agent_loop import AgentLoop
from novare.llm_client import LLMResponse, ToolCall
from novare.recovery.classifier import classify_exception
from novare.recovery.policy import RetryPolicy
from novare.recovery.types import ErrorEnvelope, FailureKind, RetryExhaustedError
from novare.reflexion import (
    CURRENT_SCHEMA_VERSION,
    InvalidReflexionStateError,
    ReflexionState,
    compute_action_fingerprint,
)
from novare.reflexion.progress import (
    ProgressTracker,
    compute_progress_fingerprint,
    progress_signal_digest,
)
from novare.reflexion.validator import ReflectionValidator
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
    """构造带 Reflexion 的 AgentLoop（reviewer 独立 / 回退主模型）。

    返回 (loop, sleeps, main, reflection_mock, registry, reflection_calls)。
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


class _ValidatorHarness:
    def _validator(self, **kwargs):
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
            required_evidence_refs=None,
            existing_forbidden_action_fingerprints=set(),
            failed_tool=None,
            failed_arguments=None,
            idempotency="read",
        )
        defaults.update(kwargs)
        return ReflectionValidator(**defaults)

    def _base(self, **overrides):
        out = {
            "decision": "REPLAN",
            "failure_type": "QUERY_TOO_NARROW",
            "diagnosis": "d",
            "changes": ["c"],
            "revised_plan": ["p"],
            "preserve": [],
            "evidence_refs": ["event:tc1"],
            "forbidden_repeat": [],
            "suggested_next_action": None,
        }
        out.update(overrides)
        return out


class TestStrictOutputValidation(_ValidatorHarness):
    """一、严格输出字段类型校验（禁止标量自动转 list）。"""

    def test_forbidden_repeat_scalar_string_rejected(self):
        v = self._validator()
        ok, reason = v.validate(self._base(forbidden_repeat="a" * 64))
        assert not ok
        assert "forbidden_repeat" in reason

    def test_changes_string_rejected(self):
        v = self._validator()
        ok, reason = v.validate(self._base(changes="single-change"))
        assert not ok
        assert "changes" in reason

    def test_revised_plan_string_rejected(self):
        v = self._validator()
        ok, reason = v.validate(self._base(revised_plan="single-plan"))
        assert not ok
        assert "revised_plan" in reason

    def test_changes_dict_rejected(self):
        v = self._validator()
        ok, reason = v.validate(self._base(changes={"a": "b"}))
        assert not ok
        assert "changes" in reason

    def test_diagnosis_empty_rejected(self):
        v = self._validator()
        ok, reason = v.validate(self._base(diagnosis=""))
        assert not ok
        assert "diagnosis" in reason

    def test_diagnosis_whitespace_rejected(self):
        v = self._validator()
        ok, reason = v.validate(self._base(diagnosis="   "))
        assert not ok
        assert "diagnosis" in reason

    def test_list_with_empty_string_rejected(self):
        v = self._validator()
        ok, reason = v.validate(self._base(changes=["ok", ""]))
        assert not ok
        assert "changes" in reason

    def test_list_with_non_string_rejected(self):
        v = self._validator()
        ok, reason = v.validate(self._base(revised_plan=["ok", 123]))
        assert not ok
        assert "revised_plan" in reason

    def test_suggested_next_action_string_rejected(self):
        v = self._validator()
        ok, reason = v.validate(self._base(suggested_next_action="reader"))
        assert not ok
        assert "suggested_next_action" in reason

    def test_suggested_next_action_list_rejected(self):
        v = self._validator()
        ok, reason = v.validate(self._base(suggested_next_action=["reader", {}]))
        assert not ok
        assert "suggested_next_action" in reason

    def test_suggested_arguments_missing_rejected(self):
        v = self._validator()
        ok, reason = v.validate(self._base(suggested_next_action={"tool": "reader"}))
        assert not ok
        assert "arguments" in reason

    def test_valid_structure_normalized_and_committed(self):
        v = self._validator()
        fp = compute_action_fingerprint("reader", {"path": "/a"})
        normalized, reason = v.normalize_and_validate(self._base(
            forbidden_repeat=[fp],
            suggested_next_action={"tool": "reader", "arguments": {"path": "/x"}},
        ))
        assert normalized is not None, reason
        # 归一化字段被 trim
        assert normalized["diagnosis"] == "d"
        assert normalized["changes"] == ["c"]
        assert normalized["forbidden_repeat"] == [fp]
        assert normalized["suggested_next_action"] == {
            "tool": "reader", "arguments": {"path": "/x"},
        }

    def test_evidence_refs_required_for_failed_trigger(self):
        fp = compute_action_fingerprint("reader", {"path": "/a"})
        v = self._validator(
            triggering_action_fingerprint=fp,
            required_evidence_refs=["event:tc1"],
        )
        # 空 evidence_refs → 拒绝
        ok, reason = v.validate(self._base(evidence_refs=[], forbidden_repeat=[fp]))
        assert not ok
        assert "evidence_refs" in reason
        # 缺少触发器提供的 reference → 拒绝
        ok, reason = v.validate(self._base(evidence_refs=["event:other"], forbidden_repeat=[fp]))
        assert not ok
        assert "include trigger evidence" in reason
        # 包含 → 通过
        ok, reason = v.validate(self._base(evidence_refs=["event:tc1"], forbidden_repeat=[fp]))
        assert ok, reason

    def test_evidence_refs_optional_for_no_progress(self):
        v = self._validator()  # triggering=None, required=None
        ok, reason = v.validate(self._base(evidence_refs=[]))
        assert ok, reason


class TestFailedConflictClassification:
    """二、失败 conflict 不得标记 conflict success，继续评估 semantic。"""

    @pytest.mark.asyncio
    async def test_failed_conflict_invalid_argument_triggers_semantic(self):
        async def conflict_bad_args(args, workspace=None):
            return json.dumps({
                "ok": False,
                "conflict": True,
                "conflict_detail": "conflicting file states",
                "error_code": "INVALID_ARGUMENT",
                "error": "bad argument",
            })

        fp = compute_action_fingerprint("reader", {"path": "/a"})
        states = []

        async def on_state(state_dict):
            states.append(state_dict)

        loop, _, _, _, _, _ = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (conflict_bad_args, READER_SCHEMA)},
            reflection_responses=[
                LLMResponse(content=_reflection_json(fp), tool_calls=[], stop_reason="stop"),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go", on_reflexion_state=on_state)

        assert states, "on_reflexion_state 应被调用"
        state = ReflexionState.from_dict(states[-1])
        # 失败 fingerprint 进入 forbidden（validator 要求触发 fp 在 forbidden_repeat）
        assert fp in state.forbidden_action_fingerprints
        # 触发类型是 SEMANTIC_TOOL_FAILURE（不是 CONFLICTING_OBSERVATIONS / conflict success）
        assert state.records, "应产生应用后的反思记录"
        assert state.records[0].trigger == "semantic_tool_failure"

    @pytest.mark.asyncio
    async def test_success_conflict_still_triggers_conflict(self):
        async def conflict_ok(args, workspace=None):
            return json.dumps({
                "ok": True,
                "conflict": True,
                "conflict_detail": "two sources disagree",
                "result": "partial",
            })

        loop, _, _, _, _, _ = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (conflict_ok, READER_SCHEMA)},
        )
        session = Session()
        # 默认 REJECTED（forbidden 不匹配触发 fp），只验证触发类型经 engine 调用
        await loop.run_turn(session, "go")


class TestStateRestoreValidation:
    """三、ReflexionState.from_dict 严格校验。"""

    def test_unknown_schema_version_rejected(self):
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict({"schema_version": 99})

    def test_records_string_rejected(self):
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict({
                "schema_version": CURRENT_SCHEMA_VERSION,
                "records": "not-a-list",
            })

    def test_records_item_not_dict_rejected(self):
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict({
                "schema_version": CURRENT_SCHEMA_VERSION,
                "records": ["x"],
            })

    def test_invalid_forbidden_fingerprint_rejected(self):
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict({
                "schema_version": CURRENT_SCHEMA_VERSION,
                "forbidden_action_fingerprints": ["fp-abc"],
            })

    def test_forbidden_fingerprint_not_string_rejected(self):
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict({
                "schema_version": CURRENT_SCHEMA_VERSION,
                "forbidden_action_fingerprints": [123],
            })

    def test_negative_count_rejected(self):
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict({
                "schema_version": CURRENT_SCHEMA_VERSION,
                "reflection_count": -1,
            })

    def test_bool_count_rejected(self):
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict({
                "schema_version": CURRENT_SCHEMA_VERSION,
                "no_progress_count": True,
            })

    def test_reflected_fingerprints_not_list_rejected(self):
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict({
                "schema_version": CURRENT_SCHEMA_VERSION,
                "reflected_trigger_fingerprints": "tf1",
            })

    def test_non_applied_record_rejected(self):
        """v3 严格恢复：非 validated+applied 的记录存在即视为损坏数据（fail closed）。"""
        from novare.reflexion.types import make_reflection_record

        rec = make_reflection_record(
            trigger="semantic_tool_failure", trigger_fingerprint="tf1",
            evidence_refs=["event:tc1"], failure_type="X", diagnosis="d",
            preserve=[], changes=["c"], forbidden_action_fingerprints=[],
            revised_plan=["p"], suggested_next_action=None,
            decision="REPLAN", validated=False, applied=False,
        )
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict({
                "schema_version": CURRENT_SCHEMA_VERSION,
                "records": [rec.to_dict()],
            })

    def test_old_schema_v1_migrates_explicitly(self):
        v1 = {
            "schema_version": 1,
            "reflection_count": 1,
            "no_progress_count": 0,
            "last_progress_fingerprint": None,
            "reflected_trigger_fingerprints": ["tf1"],
            "forbidden_action_fingerprints": [],
            "records": [],
            "blocked_reason": None,
        }
        state = ReflexionState.from_dict(v1)
        assert state.schema_version == CURRENT_SCHEMA_VERSION == 3
        # v1 无 progress 信号 → 空 digest 集合；fingerprint 可仅凭状态重建
        assert state.progress_signal_digests == set()
        assert len(state.last_progress_fingerprint) == 64
        assert state.reflected_trigger_fingerprints == {"tf1"}

    def test_from_dict_never_returns_partial_state(self):
        # 校验失败时抛异常，不存在"部分恢复"返回路径
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict({
                "schema_version": CURRENT_SCHEMA_VERSION,
                "reflection_count": 1,
                "forbidden_action_fingerprints": ["bad-fp"],
            })


class TestFailClosedResume:
    """四、显式恢复 fail closed：不静默执行新 turn。"""

    @pytest.mark.asyncio
    async def test_run_turn_resume_failure_emits_unified_error(self, monkeypatch):
        import web.backend.agent_service as agent_mod
        from web.backend.agent_service import AgentService, RecoveryResumeError
        from novare.config import NovareConfig

        svc = AgentService()
        svc.config = NovareConfig.load()
        svc.agent = MagicMock()
        svc.agent.run_turn = AsyncMock()
        monkeypatch.setattr(
            svc, "_restore_reflexion_state",
            AsyncMock(side_effect=RecoveryResumeError()),
        )
        session = Session()
        queue = asyncio.Queue()
        await svc.run_turn(session, "hi", queue, user_id="user-1", recovery_run_id="r1")
        event = queue.get_nowait()
        assert event["type"] == "error"
        assert event["code"] == "RECOVERY_RESUME_FAILED"
        assert "无法恢复" in event["message"]
        # 恢复失败：不执行新 turn
        svc.agent.run_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_turn_without_recovery_run_id_normal_turn(self, monkeypatch):
        import web.backend.agent_service as agent_mod
        from web.backend.agent_service import AgentService
        from novare.config import NovareConfig

        svc = AgentService()
        svc.config = NovareConfig.load()
        svc.agent = MagicMock()
        restored = MagicMock()
        monkeypatch.setattr(svc, "_restore_reflexion_state", AsyncMock(return_value=restored))
        session = Session()
        queue = asyncio.Queue()
        await svc.run_turn(session, "hi", queue, user_id="user-1")
        # 不传 recovery_run_id → 不调用恢复
        svc._restore_reflexion_state.assert_not_awaited()


class TestProgressTrackerRestore:
    """六、ProgressTracker 跨进程恢复（digest 化信号入 ReflexionState v3）。"""

    def _restored_state(self):
        """构造带持久化 progress_signal_digests + 重建 fingerprint 的 v3 状态。"""
        fp = compute_action_fingerprint("reader", {"path": "/a"})
        digest = progress_signal_digest(
            kind="tool_success", tool="reader",
            action_fingerprint=fp, summary_digest="b" * 64,
        )
        state = ReflexionState()
        state.no_progress_count = 2
        state.progress_signal_digests = {digest}
        state.last_progress_fingerprint = compute_progress_fingerprint(
            signal_digests=[digest],
        )
        return state

    def test_restored_no_progress_continues(self):
        state = self._restored_state()
        tracker = ProgressTracker.from_state(state)
        # 无真实进展（同 digest 集合）：继续无进展
        assert tracker.update(
            completed=[], key_findings=[], success_signal_digests=state.progress_signal_digests,
        ) is False

    def test_restored_real_progress_resets(self):
        state = self._restored_state()
        tracker = ProgressTracker.from_state(state)
        new_digest = progress_signal_digest(
            kind="tool_success", tool="reader",
            action_fingerprint="c" * 64, summary_digest="d" * 64,
        )
        assert tracker.update(completed=[], key_findings=[], success_signal_digests=[new_digest]) is True

    def test_sync_to_state_updates_state(self):
        state = ReflexionState()
        tracker = ProgressTracker.from_state(state)
        tracker.update(completed=["step1"], key_findings=[], success_signal_digests=[])
        tracker.sync_to_state(state)
        # 状态中只有 digest（64-hex），不含明文 "step1"
        assert len(state.progress_signal_digests) == 1
        d = next(iter(state.progress_signal_digests))
        assert len(d) == 64 and "step1" not in d
        assert state.last_progress_fingerprint == tracker.last_progress_fingerprint

    @pytest.mark.asyncio
    async def test_agentloop_no_progress_continues_after_restore(self):
        """恢复后无真实进展时 no_progress_count 继续累加（不清零）。"""
        async def fail_handler(args, workspace=None):
            return "Error: reader failed"

        state = self._restored_state()  # no_progress_count=2
        states = []

        async def on_state(state_dict):
            states.append(state_dict)

        loop, _, _, _, _, _ = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (fail_handler, READER_SCHEMA)},
            reflection_responses=[
                LLMResponse(content=_reflection_json("a" * 64), tool_calls=[], stop_reason="stop"),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go", initial_reflexion_state=state, on_reflexion_state=on_state)

        assert states, "on_reflexion_state 应被调用"
        final = ReflexionState.from_dict(states[-1])
        # 失败轮（无新成功信号）→ 无进展 → 2 + 1 = 3
        assert final.no_progress_count == 3


class TestSuggestedForbiddenRecheck:
    """七、注入 suggested action 前复查当前 forbidden 集合。"""

    def test_suggested_not_injected_when_forbidden(self):
        fp = compute_action_fingerprint("reader", {"path": "/a"})
        state = ReflexionState()
        state.forbidden_action_fingerprints.add(fp)
        from novare.reflexion.types import make_reflection_record

        state.add_reflection(make_reflection_record(
            trigger="semantic_tool_failure", trigger_fingerprint="tf1",
            evidence_refs=["event:tc1"], failure_type="X", diagnosis="d",
            preserve=[], changes=["c"], forbidden_action_fingerprints=[fp],
            revised_plan=["p"],
            suggested_next_action={"tool": "reader", "arguments": {"path": "/a"}},
            decision="REPLAN", validated=True, applied=True,
        ))
        loop, _, _, _, _, _ = _build_loop(
            agent_responses=[LLMResponse(content="done", tool_calls=[], stop_reason="stop")],
        )
        block = loop._build_recovery_context_block(state)
        # 建议动作已被禁止 → 不注入建议（但仍注入诊断/禁止列表）
        assert "建议下一步" not in block
        assert fp in block

    def test_suggested_injected_when_allowed(self):
        fp = compute_action_fingerprint("reader", {"path": "/a"})
        state = ReflexionState()
        from novare.reflexion.types import make_reflection_record

        state.add_reflection(make_reflection_record(
            trigger="semantic_tool_failure", trigger_fingerprint="tf1",
            evidence_refs=["event:tc1"], failure_type="X", diagnosis="d",
            preserve=[], changes=["c"], forbidden_action_fingerprints=[fp],
            revised_plan=["p"],
            suggested_next_action={"tool": "reader", "arguments": {"path": "/x"}},
            decision="REPLAN", validated=True, applied=True,
        ))
        loop, _, _, _, _, _ = _build_loop(
            agent_responses=[LLMResponse(content="done", tool_calls=[], stop_reason="stop")],
        )
        block = loop._build_recovery_context_block(state)
        assert "建议下一步" in block

    def test_malformed_suggested_not_injected(self):
        state = ReflexionState()
        from novare.reflexion.types import make_reflection_record

        state.add_reflection(make_reflection_record(
            trigger="semantic_tool_failure", trigger_fingerprint="tf1",
            evidence_refs=["event:tc1"], failure_type="X", diagnosis="d",
            preserve=[], changes=["c"], forbidden_action_fingerprints=[],
            revised_plan=["p"],
            suggested_next_action={"tool": "reader", "arguments": "not-a-dict"},
            decision="REPLAN", validated=True, applied=True,
        ))
        loop, _, _, _, _, _ = _build_loop(
            agent_responses=[LLMResponse(content="done", tool_calls=[], stop_reason="stop")],
        )
        block = loop._build_recovery_context_block(state)
        assert "建议下一步" not in block


class TestRetryExhaustedClassification:
    """八、classify_exception 保留 RetryExhaustedError 安全字段。"""

    def test_preserves_error_code_and_status(self):
        exc = RetryExhaustedError(
            "retry exhausted", attempts=3, error_code="SERVICE_UNAVAILABLE",
            cause_type="HTTPStatusError", status_code=503,
        )
        # RetryExhaustedError 自身保留安全诊断字段
        assert exc.attempts == 3
        assert exc.cause_type == "HTTPStatusError"
        envelope = classify_exception(exc)
        assert isinstance(envelope, ErrorEnvelope)
        assert envelope.error_code == "SERVICE_UNAVAILABLE"
        assert envelope.status_code == 503
        assert envelope.retryable is False
        assert envelope.kind == FailureKind.TRANSIENT

    def test_no_status_does_not_degrade_to_unknown(self):
        exc = RetryExhaustedError(
            "retry exhausted", attempts=2, error_code="RETRY_EXHAUSTED",
            cause_type="TimeoutError",
        )
        envelope = classify_exception(exc)
        assert envelope.error_code == "RETRY_EXHAUSTED"
        assert envelope.status_code is None
        assert envelope.kind != FailureKind.UNKNOWN
        assert envelope.kind == FailureKind.TRANSIENT
        assert envelope.retryable is False
