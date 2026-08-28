"""PR 3 review 第四轮修复测试。

一、禁止持久化/发送原始工具 summary（digest 化 + 全路径脱敏）
二、v3 progress_signal_digests（可重建、有界、不误判）
三、v1/v2 → v3 migration（丢弃明文）
四、严格恢复 ReflectionRecord + 状态 invariant
五、resume fail-closed 的 Redis 任务状态清理
六、WebSocket 真实恢复失败（route 级）
"""

import asyncio
import hashlib
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from novare.agent_loop import AgentLoop
from novare.llm_client import LLMResponse, ToolCall
from novare.recovery.policy import RetryPolicy
from novare.reflexion import (
    CURRENT_SCHEMA_VERSION,
    InvalidReflexionStateError,
    ReflexionState,
    compute_action_fingerprint,
)
from novare.reflexion.progress import (
    MAX_PROGRESS_SIGNAL_DIGESTS,
    ProgressTracker,
    compute_progress_fingerprint,
    progress_signal_digest,
)
from novare.session import Session
from novare.tools.registry import ToolDef, ToolRegistry

SECRET_TEXT = (
    "Authorization: Bearer sk-secret-value "
    "api_key=secret url_token=abc123token"
)

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

    返回 (loop, sleeps, main, reflection_mock, registry, reflection_calls, seen_messages)。
    seen_messages 记录反思 LLM 收到的原始 messages（供 secret 泄漏断言）。
    """
    sleeps = []
    reflection_calls: list[int] = []
    seen_messages: list = []

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
            seen_messages.append(args[0] if args else kwargs.get("messages", []))
            return await original(*args, **kwargs)

        reviewer_llm.collect_stream = AsyncMock(side_effect=counting_reviewer)
    else:
        agent_q = list(agent_responses)
        refl_q = list(reflection_responses) if reflection_responses is not None else None

        async def main_collect(messages, **kwargs):
            if messages and str(messages[0].get("content", "")).startswith("你是一个严谨的反思分析器"):
                reflection_calls.append(1)
                seen_messages.append(messages)
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
    return loop, sleeps, main, reflection_mock, registry, reflection_calls, seen_messages


class TestSummarySecrets:
    """一、summary secret 不进入 state / reviewer / Recovery Context / 事件。"""

    @pytest.mark.asyncio
    async def test_success_summary_secret_absent_from_all_paths(self):
        """工具成功结果含 secret：不出现在 to_dict / on_reflexion_state /
        recovery_data / recovery events / reviewer messages / Recovery Context。"""
        async def leaky_handler(args, workspace=None):
            return json.dumps({"ok": True, "result": {"data": SECRET_TEXT}})

        async def fail_handler(args, workspace=None):
            return json.dumps({
                "ok": False, "error_code": "INVALID_ARGUMENT", "error": "bad arg",
            })

        state_dicts = []
        recovery_state_dicts = []
        reflexion_events = []

        async def on_state(state_dict):
            state_dicts.append(state_dict)

        async def on_recovery(state_dict):
            recovery_state_dicts.append(state_dict)

        async def on_event(event_type, payload):
            reflexion_events.append({"event_type": event_type, **payload})

        loop, _, _, _, _, _, seen_messages = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/ok"}),
                _tc("reader", {"path": "/fail"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={
                "reader": (None, READER_SCHEMA),
            },
            reflection_responses=[
                LLMResponse(content=_reflection_json("a" * 64), tool_calls=[], stop_reason="stop"),
            ],
        )
        # 替换 handler：第一轮泄漏 secret，第二轮失败
        calls = {"n": 0}

        async def handler(args, workspace=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return json.dumps({"ok": True, "result": {"data": SECRET_TEXT}})
            return json.dumps({
                "ok": False, "error_code": "INVALID_ARGUMENT", "error": "bad arg",
            })

        registry = loop.tool_registry
        registry.register_tool(ToolDef(
            name="reader", description="r", parameters=READER_SCHEMA,
            handler=handler, idempotency="read",
            retry_policy=RetryPolicy(max_attempts=1),
        ))

        session = Session()
        await loop.run_turn(
            session, "go",
            on_reflexion_state=on_state,
            on_recovery_state=on_recovery,
            on_reflexion_event=on_event,
        )

        assert state_dicts, "on_reflexion_state 应被调用"
        assert recovery_state_dicts, "on_recovery_state 应被调用"
        assert reflexion_events, "on_reflexion_event 应被调用"

        # 所有传播路径序列化后都不含 secret 明文
        serialized = []
        for d in state_dicts:
            serialized.append(json.dumps(d, ensure_ascii=False))
        for d in recovery_state_dicts:
            serialized.append(json.dumps(d, ensure_ascii=False))
        for d in reflexion_events:
            serialized.append(json.dumps(d, ensure_ascii=False))
        for msgs in seen_messages:
            serialized.append(json.dumps(msgs, ensure_ascii=False))
        joined = "\n".join(serialized)
        assert "sk-secret-value" not in joined
        assert "api_key=secret" not in joined
        assert "abc123token" not in joined

        # ReflexionState.to_dict() 不含明文
        state = ReflexionState.from_dict(state_dicts[-1])
        assert SECRET_TEXT not in json.dumps(state.to_dict(), ensure_ascii=False)
        # progress_signal_digests 全部为 64-hex digest（无明文）
        for d in state.progress_signal_digests:
            assert len(d) == 64
            assert d.isalnum()

        # Recovery Context（注入 system prompt 的私有块）不含 secret
        block = loop._build_recovery_context_block(state)
        assert "sk-secret-value" not in block
        assert "api_key=secret" not in block

    @pytest.mark.asyncio
    async def test_reflection_output_with_secret_sanitized_into_state(self):
        """反思输出含 secret：engine 在入 record 前 sanitize，state.to_dict 不含明文。"""
        fp = compute_action_fingerprint("reader", {"path": "/a"})
        state_dicts = []

        async def on_state(state_dict):
            state_dicts.append(state_dict)

        async def fail_reader(args, workspace=None):
            return json.dumps({
                "ok": False, "error_code": "INVALID_ARGUMENT", "error": "bad arg",
            })

        loop, _, _, _, _, _, _ = _build_loop(
            agent_responses=[
                _tc("reader", {"path": "/a"}),
                LLMResponse(content="done", tool_calls=[], stop_reason="stop"),
            ],
            tool_handlers={"reader": (fail_reader, READER_SCHEMA)},
            reflection_responses=[
                LLMResponse(content=_reflection_json(
                    fp,
                    diagnosis=f"原因: {SECRET_TEXT}",
                    changes=[f"建议: {SECRET_TEXT}"],
                ), tool_calls=[], stop_reason="stop"),
            ],
        )
        session = Session()
        await loop.run_turn(session, "go", on_reflexion_state=on_state)

        state = ReflexionState.from_dict(state_dicts[-1])
        raw = json.dumps(state.to_dict(), ensure_ascii=False)
        assert "sk-secret-value" not in raw
        assert "api_key=secret" not in raw
        assert "abc123token" not in raw
        # 含 secret 的反思输出被 validator 拒绝（安全），不进入 record/状态
        assert state.records == []
        # 反思被触发一次但被拒绝（reflection_count 消耗预算，不产生记录）
        assert state.reflection_count >= 1


class TestProgressDigest:
    """二、v3 progress_signal_digests：不透明、可重建、有界、不误判。"""

    def test_digest_hides_tool_and_summary_plaintext(self):
        summary_digest = hashlib.sha256(SECRET_TEXT.encode("utf-8")).hexdigest()
        digest = progress_signal_digest(
            kind="tool_success", tool="reader",
            action_fingerprint="a" * 64, summary_digest=summary_digest,
        )
        assert len(digest) == 64
        assert "reader" not in digest
        assert "secret" not in digest
        assert SECRET_TEXT not in digest

    def test_fingerprint_rebuildable_from_state(self):
        tracker = ProgressTracker()
        tracker.update(
            completed=["step A"], key_findings=["finding 1"],
            success_signal_digests=[],
        )
        state = ReflexionState()
        tracker.sync_to_state(state)
        # 仅凭持久化状态重建 fingerprint
        rebuilt = compute_progress_fingerprint(signal_digests=state.progress_signal_digests)
        assert rebuilt == state.last_progress_fingerprint
        # 状态中无明文
        raw = json.dumps(state.to_dict(), ensure_ascii=False)
        assert "step A" not in raw and "finding 1" not in raw

    def test_completed_findings_restore_no_false_progress(self):
        """completed/key_findings 非空恢复：无新信号时继续累计，不误判进展。"""
        tracker = ProgressTracker()
        tracker.update(completed=["c1"], key_findings=["f1"], success_signal_digests=[])
        state = ReflexionState()
        state.no_progress_count = 2
        tracker.sync_to_state(state)

        restored = ReflexionState.from_dict(state.to_dict())
        assert restored.progress_signal_digests == state.progress_signal_digests
        assert restored.last_progress_fingerprint == state.last_progress_fingerprint

        tracker2 = ProgressTracker.from_state(restored)
        made = tracker2.update(completed=[], key_findings=[], success_signal_digests=[])
        assert made is False  # 无新信号 → 不误判进展
        restored.no_progress_count += 1  # AgentLoop 中 record_no_progress
        assert restored.no_progress_count == 3

    def test_signal_digests_capped_deterministically(self):
        """超上限时使用确定性策略（字典序移除），不无界增长。"""
        tracker = ProgressTracker()
        for i in range(MAX_PROGRESS_SIGNAL_DIGESTS + 50):
            tracker.update(completed=[f"step-{i}"], key_findings=[], success_signal_digests=[])
        assert len(tracker.signal_digests) == MAX_PROGRESS_SIGNAL_DIGESTS
        # 确定性：再次构造相同集合结果一致
        tracker2 = ProgressTracker()
        for i in range(MAX_PROGRESS_SIGNAL_DIGESTS + 50):
            tracker2.update(completed=[f"step-{i}"], key_findings=[], success_signal_digests=[])
        assert tracker2.signal_digests == tracker.signal_digests
        # 全部为 64-hex
        assert all(len(d) == 64 for d in tracker.signal_digests)


class TestMigration:
    """三、v1/v2 → v3 migration：丢弃明文、重建 baseline。"""

    def test_v2_migration_drops_plaintext_signals(self):
        v2 = {
            "schema_version": 2,
            "reflection_count": 1,
            "no_progress_count": 4,
            "last_progress_fingerprint": "x" * 64,
            "reflected_trigger_fingerprints": ["tf1"],
            "forbidden_action_fingerprints": [],
            "records": [],
            "blocked_reason": None,
            "cumulative_success_signals": [
                f"reader:abc:{SECRET_TEXT}",
                "other:def:plain-ok",
            ],
        }
        state = ReflexionState.from_dict(v2)
        assert state.schema_version == CURRENT_SCHEMA_VERSION == 3
        raw = json.dumps(state.to_dict(), ensure_ascii=False)
        assert "sk-secret-value" not in raw
        assert "api_key=secret" not in raw
        assert "abc123token" not in raw
        assert "plain-ok" not in raw
        assert "cumulative_success_signals" not in raw
        # 旧明文信号被哈希为 digest（64-hex），明文丢弃
        assert len(state.progress_signal_digests) == 2
        assert all(len(d) == 64 for d in state.progress_signal_digests)
        # baseline 可仅凭持久化状态重建
        assert state.last_progress_fingerprint == compute_progress_fingerprint(
            signal_digests=state.progress_signal_digests,
        )
        # no_progress_count 保留（安全计数）
        assert state.no_progress_count == 4

    def test_v1_migration_no_signals(self):
        state = ReflexionState.from_dict({
            "schema_version": 1,
            "reflection_count": 1,
            "no_progress_count": 0,
            "last_progress_fingerprint": None,
            "reflected_trigger_fingerprints": ["tf1"],
            "forbidden_action_fingerprints": [],
            "records": [],
            "blocked_reason": None,
        })
        assert state.schema_version == 3
        assert state.progress_signal_digests == set()
        assert len(state.last_progress_fingerprint) == 64

    def test_unknown_version_still_fails_closed(self):
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict({"schema_version": 99})

    def test_v3_serialized_never_contains_plaintext_field(self):
        state = ReflexionState()
        state.progress_signal_digests.add("a" * 64)
        raw = json.dumps(state.to_dict(), ensure_ascii=False)
        assert "cumulative_success_signals" not in raw
        assert "progress_signal_digests" in raw


class TestStrictRecordAndInvariants:
    """四、严格恢复 ReflectionRecord + 状态 invariant。"""

    def _valid_record_dict(self):
        fp = "a" * 64
        return {
            "reflection_id": "refl_1",
            "trigger": "semantic_tool_failure",
            "trigger_fingerprint": f"semantic_tool_failure:{fp}:INVALID_ARGUMENT",
            "evidence_refs": ["event:tc1"],
            "failure_type": "INVALID_ARGUMENT",
            "diagnosis": "诊断",
            "preserve": [],
            "changes": ["变更"],
            "forbidden_action_fingerprints": [fp],
            "revised_plan": ["计划"],
            "suggested_next_action": None,
            "decision": "REPLAN",
            "validated": True,
            "applied": True,
            "created_at": "2026-01-01T00:00:00+00:00",
        }

    def _valid_state_dict(self, records):
        fp = "a" * 64
        return {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "reflection_count": len(records),
            "no_progress_count": 0,
            "last_progress_fingerprint": "b" * 64,
            "reflected_trigger_fingerprints": [r["trigger_fingerprint"] for r in records],
            "forbidden_action_fingerprints": [fp],
            "records": records,
            "blocked_reason": None,
            "progress_signal_digests": [],
        }

    def test_valid_record_accepted(self):
        from novare.reflexion.types import ReflectionRecord

        rec = ReflectionRecord.from_dict_strict(self._valid_record_dict())
        assert rec.reflection_id == "refl_1"
        assert rec.validated is True

    def test_scalar_evidence_refs_rejected(self):
        from novare.reflexion.types import ReflectionRecord

        d = self._valid_record_dict()
        d["evidence_refs"] = "event:tc1"  # 标量 → 拒绝（禁止自动转 list）
        with pytest.raises(InvalidReflexionStateError):
            ReflectionRecord.from_dict_strict(d)

    def test_scalar_changes_rejected(self):
        from novare.reflexion.types import ReflectionRecord

        d = self._valid_record_dict()
        d["changes"] = "single-change"
        with pytest.raises(InvalidReflexionStateError):
            ReflectionRecord.from_dict_strict(d)

    def test_wrong_trigger_fingerprint_format_rejected(self):
        from novare.reflexion.types import ReflectionRecord

        d = self._valid_record_dict()
        d["trigger_fingerprint"] = "tf1"
        with pytest.raises(InvalidReflexionStateError):
            ReflectionRecord.from_dict_strict(d)

    def test_decision_invalid_rejected(self):
        from novare.reflexion.types import ReflectionRecord

        d = self._valid_record_dict()
        d["decision"] = "AUTO_RETRY"
        with pytest.raises(InvalidReflexionStateError):
            ReflectionRecord.from_dict_strict(d)

    def test_validated_not_strict_true_rejected(self):
        from novare.reflexion.types import ReflectionRecord

        d = self._valid_record_dict()
        d["validated"] = 1  # 非严格 True
        with pytest.raises(InvalidReflexionStateError):
            ReflectionRecord.from_dict_strict(d)

    def test_records_over_limit_rejected(self):
        rec = self._valid_record_dict()
        state_dict = self._valid_state_dict([rec] * (20 + 1))
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict(state_dict)

    def test_forbidden_set_over_limit_rejected(self):
        rec = self._valid_record_dict()
        state_dict = self._valid_state_dict([rec])
        state_dict["forbidden_action_fingerprints"] = [f"{i:064x}" for i in range(513)]
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict(state_dict)

    def test_signal_digests_over_limit_rejected(self):
        rec = self._valid_record_dict()
        state_dict = self._valid_state_dict([rec])
        state_dict["progress_signal_digests"] = [f"{i:064x}" for i in range(MAX_PROGRESS_SIGNAL_DIGESTS + 1)]
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict(state_dict)

    def test_invariant_trigger_fp_missing_from_reflected_rejected(self):
        rec = self._valid_record_dict()
        state_dict = self._valid_state_dict([rec])
        state_dict["reflected_trigger_fingerprints"] = []
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict(state_dict)

    def test_invariant_forbidden_missing_rejected(self):
        rec = self._valid_record_dict()
        state_dict = self._valid_state_dict([rec])
        state_dict["forbidden_action_fingerprints"] = []
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict(state_dict)

    def test_invariant_reflection_count_lt_records_rejected(self):
        rec = self._valid_record_dict()
        state_dict = self._valid_state_dict([rec])
        state_dict["reflection_count"] = 0
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict(state_dict)

    def test_last_progress_fingerprint_non_hex_rejected(self):
        state_dict = self._valid_state_dict([])
        state_dict["last_progress_fingerprint"] = "not-a-fingerprint"
        with pytest.raises(InvalidReflexionStateError):
            ReflexionState.from_dict(state_dict)


class _FakeRedis:
    """内存版 RedisService（记录 set_json/删除操作）。"""

    def __init__(self):
        self.is_available = True
        self.store: dict = {}
        self.set_nx = AsyncMock(return_value=True)
        self.delete_if_value = AsyncMock(return_value=True)
        self.delete = AsyncMock()
        self.set = AsyncMock()
        self.get = AsyncMock(return_value=None)
        self.set_json = AsyncMock(side_effect=self._set_json)
        self.get_json = AsyncMock(side_effect=self._get_json)

    def _set_json(self, key, value, ttl=None):
        self.store[key] = json.loads(json.dumps(value))

    def _get_json(self, key):
        return self.store.get(key)


class TestResumeFailClosedRedis:
    """五、resume fail-closed 的 Redis 任务状态清理。"""

    @pytest.mark.asyncio
    async def test_resume_failure_cleans_task_status_and_lock(self, monkeypatch):
        import web.backend.agent_service as agent_mod
        from web.backend.agent_service import AgentService, RecoveryResumeError
        from novare.config import NovareConfig

        fake_redis = _FakeRedis()
        monkeypatch.setattr(agent_mod, "redis_service", fake_redis)

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
        user_id = str(uuid.uuid4())
        await svc.run_turn(session, "hi", queue, user_id=user_id, recovery_run_id="r1")

        task_key = f"task:user:{user_id}:session:{session.session_id}"
        lock_key = f"lock:user:{user_id}:session:{session.session_id}"
        # 1. task status → error（统一安全消息），updated_at 更新
        task = fake_redis.store[task_key]
        assert task["status"] == "error"
        assert task["error"] == "无法恢复指定任务，请重新开始或选择有效的运行记录。"
        assert task["updated_at"]
        # 2. lock 被释放
        fake_redis.delete_if_value.assert_awaited()
        # 3. AgentLoop 未调用
        svc.agent.run_turn.assert_not_awaited()
        # 4. WebSocket 收到一次终态 error
        event = queue.get_nowait()
        assert event["type"] == "error"
        assert event["code"] == "RECOVERY_RESUME_FAILED"
        with pytest.raises(asyncio.QueueEmpty):
            queue.get_nowait()

    @pytest.mark.asyncio
    async def test_resume_failure_task_status_not_running(self, monkeypatch):
        """/api/chat/{session_id}/task 不返回 running（恢复失败后为 error/idle）。"""
        import web.backend.agent_service as agent_mod
        from web.backend.agent_service import AgentService, RecoveryResumeError
        from novare.config import NovareConfig

        fake_redis = _FakeRedis()
        monkeypatch.setattr(agent_mod, "redis_service", fake_redis)

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
        user_id = str(uuid.uuid4())
        await svc.run_turn(session, "hi", queue, user_id=user_id, recovery_run_id="r1")

        # 恢复失败前状态是 running → 失败后必须不是 running
        task = await fake_redis.get_json(f"task:user:{user_id}:session:{session.session_id}")
        assert task is not None
        assert task["status"] != "running"

    @pytest.mark.asyncio
    async def test_redis_unavailable_resume_failure_still_errors(self, monkeypatch):
        """Redis 不可用（降级态）时恢复失败仍发终态 error，不残留 running。"""
        import web.backend.agent_service as agent_mod
        from web.backend.agent_service import AgentService, RecoveryResumeError
        from novare.config import NovareConfig

        fake_redis = _FakeRedis()
        fake_redis.is_available = False
        monkeypatch.setattr(agent_mod, "redis_service", fake_redis)

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
        user_id = str(uuid.uuid4())
        await svc.run_turn(session, "hi", queue, user_id=user_id, recovery_run_id="r1")
        event = queue.get_nowait()
        assert event["code"] == "RECOVERY_RESUME_FAILED"
        svc.agent.run_turn.assert_not_awaited()


class TestRouteRealResumeFailure:
    """六、WebSocket 真实恢复失败（route 级）。"""

    def _patch(self, monkeypatch, fake_redis, svc):
        import web.backend.routes.chat as chat_mod
        import web.backend.agent_service as agent_mod
        import web.backend.app as app_mod

        monkeypatch.setattr(app_mod, "agent_service", svc)
        monkeypatch.setattr(agent_mod, "redis_service", fake_redis)
        monkeypatch.setattr(chat_mod, "redis_service", fake_redis)
        monkeypatch.setattr(chat_mod, "SessionRepository", _FakeSessionRepo)
        monkeypatch.setattr(chat_mod, "get_session_factory", lambda: lambda: _FakeDB())
        fake_user_id = str(uuid.uuid4())
        monkeypatch.setattr(chat_mod, "decode_access_token", lambda token: fake_user_id)
        return fake_user_id

    def test_ws_nonexistent_recovery_run_id_fails_closed(self, monkeypatch):
        """合法格式但不存在的 recovery_run_id → 只收到 RECOVERY_RESUME_FAILED，
        无执行事件；AgentLoop 未启动；task 状态不残留 running。"""
        import web.backend.agent_service as agent_mod
        from web.backend.agent_service import AgentService, RecoveryResumeError
        from web.backend.app import app
        from novare.config import NovareConfig

        fake_redis = _FakeRedis()
        svc = AgentService()
        svc.config = NovareConfig.load()
        svc.agent = MagicMock()
        svc.agent.run_turn = AsyncMock()
        monkeypatch.setattr(
            svc, "_restore_reflexion_state",
            AsyncMock(side_effect=RecoveryResumeError()),
        )
        fake_user_id = self._patch(monkeypatch, fake_redis, svc)

        session_id = f"s-{uuid.uuid4().hex[:8]}"
        client = TestClient(app)
        with client.websocket_connect(f"/ws/chat/{session_id}?token=dummy") as ws:
            ws.send_json({"type": "send", "content": "resume", "recovery_run_id": "nonexistent-run"})
            event = ws.receive_json()
            # 客户端只收到一次终态 error（RECOVERY_RESUME_FAILED），无执行事件
            assert event["type"] == "error"
            assert event["code"] == "RECOVERY_RESUME_FAILED"
            assert "text_delta" not in event
            assert "tool" not in event
            assert "recovery_state" not in event

        # AgentLoop 未启动
        svc.agent.run_turn.assert_not_awaited()
        # task 状态不残留 running
        task = fake_redis.store.get(f"task:user:{fake_user_id}:session:{session_id}")
        assert task is not None and task["status"] != "running"


class _FakeDB:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def close(self):
        pass


class _FakeSessionRepo:
    def __init__(self, db, user_uuid):
        pass

    async def get_by_id(self, session_id):
        return object()
