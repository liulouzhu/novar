"""novare/agent_loop.py — 核心 agent 循环

借鉴 claw-code 的 ConversationRuntime.run_turn() 模式，
支持主智能体和子智能体共用同一循环（通过 duck typing 接受 ToolRegistry 或 SubagentToolExecutor）。

PR 2: 协议完整性、执行恢复状态与持久化
- 每个 tool_call_id 最终恰好有一个 tool result
- commit_tool_result_once 统一提交入口
- Batch 注册在工具执行前完成
- timeout/cancel/exception 终态化
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random as _random
import time
import uuid
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol, runtime_checkable

from novare.context_manager import (
    TokenUsage,
    estimate_messages_tokens,
    estimate_tools_tokens,
)
from novare.context_compactor import HybridContextCompactor
from novare.recovery.classifier import classify_exception, classify_tool_result, sanitize_error
from novare.recovery.executor import RetryExecutor, retry_tool_call
from novare.recovery.policy import RetryBudget, RetryPolicy
from novare.recovery.state import (
    RecoveryState,
    RunStatus,
    ToolCallStatus,
    _make_synthetic_result,
    commit_tool_result_once,  # re-export：函数本体已移至 recovery.state（协议层归属）
)
from novare.recovery.terminalize import (
    terminalize_on_cancel,
    terminalize_on_exception,
    terminalize_on_max_iterations,
    terminalize_on_timeout,
)
from novare.recovery.types import Outcome
from novare.reflexion.engine import ReflectionContext, ReflexionEngine
from novare.reflexion.progress import ProgressTracker, progress_signal_digest
from novare.reflexion.triggers import (
    ToolEventSummary,
    compute_action_fingerprint,
    evaluate_triggers,
    is_semantic_error_code,
    is_terminal_error_code,
    is_transient_error_code,
)
from novare.reflexion.types import ReflexionState
from novare.task_state import TaskState, TaskStateManager
from novare.tool_result import parse_tool_result

if TYPE_CHECKING:
    from novare.hallucination_verifier import HallucinationVerifier
    from novare.llm_client import LLMClient
    from novare.session import Session


@runtime_checkable
class ToolExecutor(Protocol):
    """工具执行器协议 — ToolRegistry 和 SubagentToolExecutor 都满足此接口"""

    def to_openai_tools(self) -> list[dict]: ...

    async def execute(self, name: str, arguments: dict, tool_context: dict | None = None) -> str: ...

logger = logging.getLogger("novare.loop")


def _extract_conflict(parsed_result) -> tuple[bool, str | None]:
    """从结构化工具结果中检测显式冲突（不凭模型感觉判断）。

    识别：data.conflict=true / data.conflicting_observations=true /
    data.conflicts 非空。
    """
    if not parsed_result.is_json or parsed_result.data is None:
        return False, None
    data = parsed_result.data
    if isinstance(data, dict):
        if data.get("conflict") is True or data.get("conflicting_observations") is True:
            detail = data.get("conflict_detail") or "conflicting observations reported"
            return True, str(detail)[:200]
        if data.get("conflicts"):
            return True, str(data.get("conflicts"))[:200]
    return False, None


class AgentLoop:
    """等价于 Claw Code 的 ConversationRuntime.run_turn()

    tool_registry 参数接受任何满足 ToolExecutor 协议的对象：
    - ToolRegistry（主智能体，完整工具集）
    - SubagentToolExecutor（子智能体，白名单受限工具集）
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolExecutor,
        system_prompt: str = "",
        max_iterations: int = 20,
        reviewer_llm: LLMClient | None = None,
        auto_compact_threshold: int = 100_000,
        turn_timeout: int = 300,
        context_max_turns: int = 3,
        context_token_budget: int = 12_000,
        context_summary_max_tokens: int = 2_500,
        context_tool_result_max_tokens: int = 1_200,
        context_llm_timeout: float = 30.0,
        context_llm_enabled: bool = True,
        context_compactor: HybridContextCompactor | None = None,
        hallucination_verifier: HallucinationVerifier | None = None,
        # ── PR 1：重试配置（默认值来自 novare/config.py）──
        llm_retry_attempts: int = 3,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 8.0,
        max_retries_per_turn: int = 6,
        retry_after_max_delay: float = 30.0,
        retry_sleep: Callable[[float], Awaitable[None]] | None = None,
        retry_random: Callable[[float, float], float] | None = None,
        # ── PR 3：Reflexion（默认关闭，关闭时行为与 PR 1/PR 2 完全一致）──
        reflexion_enabled: bool = False,
        max_reflections_per_turn: int = 2,
        reflexion_no_progress_threshold: int = 3,
        reflexion_repeated_failure_threshold: int = 2,
        reflexion_timeout: float = 30.0,
        reflexion_max_tokens: int = 1200,
        reflexion_max_recent_events: int = 8,
        reflexion_sleep: Callable[[float], Awaitable[None]] | None = None,
    ):
        self.llm_client = llm_client
        self.tool_registry: ToolExecutor = tool_registry
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.reviewer_llm = reviewer_llm
        self.auto_compact_threshold = auto_compact_threshold
        self.context_compactor = context_compactor or HybridContextCompactor(
            llm_client,
            max_turns=context_max_turns,
            token_budget=context_token_budget,
            summary_max_tokens=context_summary_max_tokens,
            tool_result_max_tokens=context_tool_result_max_tokens,
            llm_timeout=context_llm_timeout,
            llm_enabled=context_llm_enabled,
        )
        self.hallucination_verifier = hallucination_verifier
        self.turn_timeout = turn_timeout
        # ── PR 1：重试状态 ──
        self.llm_retry_attempts = llm_retry_attempts
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.max_retries_per_turn = max_retries_per_turn
        self.retry_after_max_delay = retry_after_max_delay
        self._retry_sleep = retry_sleep or asyncio.sleep
        self._retry_random = retry_random or _random.uniform
        # ── PR 3：Reflexion 状态 ──
        self.reflexion_enabled = reflexion_enabled
        self.max_reflections_per_turn = max_reflections_per_turn
        self.reflexion_no_progress_threshold = reflexion_no_progress_threshold
        self.reflexion_repeated_failure_threshold = reflexion_repeated_failure_threshold
        self.reflexion_timeout = reflexion_timeout
        self.reflexion_max_tokens = reflexion_max_tokens
        self.reflexion_max_recent_events = reflexion_max_recent_events
        self._reflexion_sleep = reflexion_sleep or asyncio.sleep

    async def run_turn(
        self,
        session: Session,
        user_input: str,
        on_text: Callable[[str], None] | None = None,
        on_tool: Callable[[str, str, dict, str | None, float | None], None] | None = None,
        tool_context: dict | None = None,
        on_task_state: Callable[[dict], None] | None = None,
        system_prompt: str | None = None,
        autosave: bool = True,
        on_compact: Callable[[object], Awaitable[None] | None] | None = None,
        should_cancel: Callable[[], Awaitable[bool] | bool] | None = None,
        on_message: Callable[[dict], Awaitable[None] | None] | None = None,
        on_verification: Callable[[dict], Awaitable[None] | None] | None = None,
        on_recovery_state: Callable[[dict], Awaitable[None] | None] | None = None,
        on_reflexion_event: Callable[[str, dict], Awaitable[None] | None] | None = None,
        on_reflexion_state: Callable[[dict], Awaitable[None] | None] | None = None,
        initial_reflexion_state: ReflexionState | None = None,
    ) -> str:
        """执行一轮对话，带 per-turn 超时保护。

        system_prompt: 可选的 per-turn 覆盖，不传则使用 self.system_prompt。
        autosave: compact 后是否自动 session.save() 写 JSONL。CLI 默认 True，Web 应传 False。
        on_compact: compact 发生后的回调，上层可用于持久化 compact 后的消息（如写 DB）。
        should_cancel: 可选的协作式取消回调，每轮迭代和工具调用前检查。返回 True 时优雅停止。
        on_recovery_state: 可选回调，RecoveryState 变更时推送快照。
        on_reflexion_event: 可选回调，Reflexion 事件（REFLECTION_*/PLAN_REVISED/...）推送。
        initial_reflexion_state: 可选的进程恢复状态（forbidden fingerprints 等）。
        超时时返回友好提示，已执行的工具调用和消息保留在 session 中。
        """
        session.verification = None

        # ── PR 2：在 run_turn 外层创建 RecoveryState ──
        # 这样 timeout handler 可以访问 state、session 和 callbacks
        recovery_state = RecoveryState()

        # 初始化局部变量，避免 early exception 时 UnboundLocalError
        _recovery_state_data: dict | None = None
        budget = RetryBudget(max_retries=self.max_retries_per_turn)
        deadline = time.monotonic() + self.turn_timeout

        # ── PR 3：turn-scoped Reflexion 状态与引擎 ──
        reflexion_state = initial_reflexion_state or ReflexionState()
        reflexion_engine = self._make_reflexion_engine(
            on_reflexion_event,
            budget=budget,
            turn_deadline=deadline,
        )
        # 从恢复状态初始化进展跟踪（累计信号 + 上次进展指纹）
        progress_tracker = ProgressTracker.from_state(reflexion_state)
        # turn 级连续失败计数（fp → count），不持久化
        reflexion_failure_counts: dict[str, int] = {}
        # PR 3：turn 级 forbidden action 阻止计数（跨迭代累计）
        forbidden_blocked_count: list[int] = [0]

        try:
            return await asyncio.wait_for(
                self._run_turn_core(
                    session, user_input, on_text, on_tool, tool_context,
                    on_task_state, system_prompt, autosave, on_compact,
                    should_cancel, on_message, on_verification,
                    on_recovery_state,
                    recovery_state,
                    budget=budget, deadline=deadline,
                    reflexion_state=reflexion_state,
                    reflexion_engine=reflexion_engine,
                    progress_tracker=progress_tracker,
                    reflexion_failure_counts=reflexion_failure_counts,
                    on_reflexion_event=on_reflexion_event,
                    on_reflexion_state=on_reflexion_state,
                    forbidden_blocked_count=forbidden_blocked_count,
                ),
                timeout=self.turn_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Turn timed out after %ds (user_input=%s)", self.turn_timeout, user_input[:80])
            # ── PR 2：终态化所有 pending calls ──
            await terminalize_on_timeout(recovery_state, session)
            await self._emit_recovery_state(on_recovery_state, recovery_state)
            return f"本轮任务超时（超过 {self.turn_timeout} 秒），请简化问题或拆分为更小的子任务后重试。"
        except asyncio.CancelledError:
            # ── PR 2：CancelledError 传播前终态化 ──
            try:
                await asyncio.wait_for(
                    terminalize_on_cancel(recovery_state, session),
                    timeout=3.0,
                )
            except Exception:
                logger.warning("Failed to terminalize on CancelledError")
            raise
        except Exception as e:
            # ── PR 2：普通异常终态化 ──
            await terminalize_on_exception(recovery_state, session, e)
            await self._emit_recovery_state(on_recovery_state, recovery_state)
            raise

    async def _run_turn_core(
        self,
        session: Session,
        user_input: str,
        on_text: Callable[[str], None] | None = None,
        on_tool: Callable[[str, str, dict, str | None, float | None], None] | None = None,
        tool_context: dict | None = None,
        on_task_state: Callable[[dict], None] | None = None,
        system_prompt: str | None = None,
        autosave: bool = True,
        on_compact: Callable[[object], Awaitable[None] | None] | None = None,
        should_cancel: Callable[[], Awaitable[bool] | bool] | None = None,
        on_message: Callable[[dict], Awaitable[None] | None] | None = None,
        on_verification: Callable[[dict], Awaitable[None] | None] | None = None,
        on_recovery_state: Callable[[dict], Awaitable[None] | None] | None = None,
        recovery_state: RecoveryState | None = None,
        budget: RetryBudget | None = None,
        deadline: float | None = None,
        reflexion_state: ReflexionState | None = None,
        reflexion_engine: ReflexionEngine | None = None,
        progress_tracker: ProgressTracker | None = None,
        reflexion_failure_counts: dict[str, int] | None = None,
        on_reflexion_event: Callable[[str, dict], Awaitable[None] | None] | None = None,
        on_reflexion_state: Callable[[dict], Awaitable[None] | None] | None = None,
        forbidden_blocked_count: list[int] | None = None,
    ) -> str:
        """执行一轮对话的核心逻辑：用户输入 → LLM（流式） → 工具循环 → 最终回答"""
        # 当前 turn 使用的 system_prompt
        effective_prompt = system_prompt if system_prompt is not None else self.system_prompt

        # Turn-scoped TaskState
        task_mgr = TaskStateManager()
        task_mgr.init_turn(user_input)
        rag_used = False

        # RecoveryState 由外层传入
        if recovery_state is None:
            recovery_state = RecoveryState()

        # PR 3：Reflexion 状态由外层传入（关闭时保持 None 语义）
        reflexion_state = reflexion_state if self.reflexion_enabled else None
        reflexion_engine = reflexion_engine if self.reflexion_enabled else None
        progress_tracker = progress_tracker if self.reflexion_enabled else None
        reflexion_failure_counts = reflexion_failure_counts if self.reflexion_enabled else None

        try:
            # 注入 reviewer_llm
            if tool_context is None:
                tool_context = {}
            if self.reviewer_llm:
                tool_context["reviewer_llm"] = self.reviewer_llm

            session.add_user_message(user_input)
            await self._emit_message(on_message, session.messages[-1])

            for iteration in range(self.max_iterations):
                recovery_state.increment_iteration()

                # 协作式取消检查
                if should_cancel:
                    _cancel_result = should_cancel()
                    if asyncio.iscoroutine(_cancel_result):
                        _cancel_result = await _cancel_result
                    if _cancel_result:
                        # ── PR 2：终态化所有 pending calls ──
                        await terminalize_on_cancel(recovery_state, session)
                        await self._emit_recovery_state(on_recovery_state, recovery_state)
                        return "任务已取消。"

                # 构建消息
                messages = self._build_messages(
                    session, task_state=task_mgr.state, system_prompt=effective_prompt,
                    reflexion_state=reflexion_state,
                )

                # Preflight compact
                if await self._preflight_compact(session, messages, task_mgr.state, system_prompt=effective_prompt, autosave=autosave, on_compact=on_compact):
                    messages = self._build_messages(
                        session, task_state=task_mgr.state, system_prompt=effective_prompt,
                        reflexion_state=reflexion_state,
                    )

                # RAG buffering
                tools = self.tool_registry.to_openai_tools()
                should_buffer = bool(
                    rag_used
                    and self.hallucination_verifier
                    and self.hallucination_verifier.enabled
                )
                response = await self._collect_stream_with_retry(
                    self.llm_client, messages, tools=tools,
                    on_text=None if should_buffer else on_text,
                    budget=budget, deadline=deadline,
                )

                # Usage tracking
                if response.usage:
                    session.usage_tracker.add(TokenUsage(
                        input_tokens=response.usage.get("prompt_tokens", 0)
                            or response.usage.get("input_tokens", 0),
                        output_tokens=response.usage.get("completion_tokens", 0)
                            or response.usage.get("output_tokens", 0),
                    ))

                # 无工具调用 → 返回最终回答
                if not response.tool_calls:
                    final_content = response.content
                    verification_report = None
                    if should_buffer and self.hallucination_verifier:
                        verification = await self.hallucination_verifier.verify(
                            answer=response.content,
                            user_question=user_input,
                            tool_context=tool_context,
                        )
                        final_content = verification.corrected_answer
                        verification_report = verification.to_dict()
                        await self._emit_verification(on_verification, verification_report)
                        if on_text and final_content:
                            on_text(final_content)

                    session.add_assistant_message(final_content)
                    if verification_report is not None:
                        session.verification = verification_report
                    await self._emit_message(on_message, session.messages[-1])

                    recovery_state.set_run_status(RunStatus.COMPLETED)
                    await self._emit_recovery_state(on_recovery_state, recovery_state)
                    await self._maybe_auto_compact(session, system_prompt=effective_prompt, autosave=autosave, on_compact=on_compact)
                    return final_content

                # ── PR 2：有工具调用 → 批量注册 ──
                # Step 1: 验证 tool_call_id 唯一性
                tc_ids = [tc.id for tc in response.tool_calls]
                if len(tc_ids) != len(set(tc_ids)):
                    # 重复 ID → 生成稳定的内部唯一 ID
                    seen = set()
                    for tc in response.tool_calls:
                        if tc.id in seen:
                            tc.id = f"{tc.id}_dedup_{uuid.uuid4().hex[:8]}"
                        seen.add(tc.id)

                # Step 2: 记录 assistant 消息（含 tool_calls）
                tool_calls_dicts = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                    for tc in response.tool_calls
                ]
                session.add_assistant_message(response.content or "", tool_calls=tool_calls_dicts)
                recovery_state.assistant_message_committed = True
                await self._emit_message(on_message, session.messages[-1])

                # Step 3: 一次性注册整个 batch 到 RecoveryState
                tc_dicts = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls]
                recovery_state.register_tool_calls_batch(tc_dicts, self.tool_registry)

                # Step 4: 持久化 TOOL_CALLS_REGISTERED 事件（通过回调）
                await self._emit_recovery_state(on_recovery_state, recovery_state)

                # PR 3：本 batch 的工具事件摘要（终态后用于触发评估）
                batch_events: list[ToolEventSummary] = []

                # Step 5: 执行每个工具调用
                for tc in response.tool_calls:
                    # PR 3：forbidden action 阻止（在执行前）
                    if reflexion_state is not None and reflexion_state.forbidden_action_fingerprints:
                        action_fp = compute_action_fingerprint(tc.name, tc.arguments)
                        if reflexion_state.is_forbidden(action_fp):
                            forbidden_blocked_count[0] += 1
                            blocked_result = json.dumps({
                                "ok": False,
                                "error": (
                                    f"FORBIDDEN_REPEATED_ACTION: {tc.name} 已被反思标记为禁止重复执行。"
                                    "请改变策略或使用其他工具。"
                                ),
                                "error_code": "FORBIDDEN_REPEATED_ACTION",
                                "retryable": False,
                                "outcome": "not_applied",
                                "attempts": 0,
                            }, ensure_ascii=False)
                            committed = await commit_tool_result_once(
                                session, recovery_state, tc.id, blocked_result, on_message,
                            )
                            if committed:
                                recovery_state.mark_failed(tc.id, "FORBIDDEN_REPEATED_ACTION")
                                if on_tool:
                                    on_tool("error", tc.name, tc.arguments, blocked_result, None)
                            await self._emit_reflexion_event(on_reflexion_event, "FORBIDDEN_ACTION_BLOCKED", {
                                "tool_call_id": tc.id,
                                "tool": tc.name,
                                "action_fingerprint": action_fp,
                                "blocked_count": forbidden_blocked_count[0],
                            })
                            await self._emit_recovery_state(on_recovery_state, recovery_state)
                            # 连续两次生成 forbidden action → 停止并返回明确阻塞原因
                            if forbidden_blocked_count[0] >= 2:
                                reflexion_state.block(
                                    "连续两次生成被禁止重复的动作，无法产生不同方案，已停止。"
                                )
                                blocked_message = "任务因重复生成被禁止的动作而停止：{}".format(
                                    reflexion_state.blocked_reason
                                )
                                # 让 Web/CLI 客户端可见（文本 + assistant 消息）
                                if on_text:
                                    on_text(blocked_message)
                                session.add_assistant_message(blocked_message)
                                await self._emit_message(on_message, session.messages[-1])
                                await terminalize_on_max_iterations(recovery_state, session)
                                await self._emit_recovery_state(on_recovery_state, recovery_state)
                                return blocked_message
                            continue

                    # 协作式取消检查
                    if should_cancel:
                        _cancel_result = should_cancel()
                        if asyncio.iscoroutine(_cancel_result):
                            _cancel_result = await _cancel_result
                        if _cancel_result:
                            # 终态化剩余未执行的 calls
                            await terminalize_on_cancel(recovery_state, session)
                            await self._emit_recovery_state(on_recovery_state, recovery_state)
                            return "任务已取消。"

                    record = recovery_state.get_record(tc.id)
                    if record:
                        # 注入 idempotency_key 到 tool_context
                        tool_context["_idempotency_key"] = record.idempotency_key
                        tool_context["_action_fingerprint"] = record.action_fingerprint

                    logger.info("Tool call: %s(%s)", tc.name, tc.arguments)
                    if on_tool:
                        on_tool("start", tc.name, tc.arguments, None, None)
                    recovery_state.mark_executing(tc.id)

                    # TOOL_STARTED 事件
                    await self._emit_recovery_state(on_recovery_state, recovery_state)

                    t0 = time.monotonic()
                    # 工具重试
                    policy = self._tool_retry_policy(tc.name)
                    tool_outcome = await retry_tool_call(
                        lambda: self.tool_registry.execute(tc.name, tc.arguments, tool_context=tool_context),
                        policy=policy,
                        budget=budget,
                        classify_result=lambda r: classify_tool_result(r, tool_name=tc.name),
                        is_ok=lambda r: parse_tool_result(r).ok,
                        sleep=self._retry_sleep,
                        random_fn=self._retry_random,
                        deadline=deadline,
                        on_retry=self._tool_on_retry(on_tool, tc.name, tc.arguments),
                    )
                    elapsed = time.monotonic() - t0
                    result = tool_outcome.result

                    # 终态失败 → 结构化错误
                    if tool_outcome.outcome != Outcome.SUCCESS:
                        envelope = tool_outcome.envelope or classify_tool_result(result, tool_name=tc.name)
                        result = envelope.to_tool_failure(tool_outcome.outcome, tool_outcome.attempts)

                    # 结构化错误检测
                    parsed_result = parse_tool_result(result)
                    is_error = not parsed_result.ok
                    if tc.name == "rag_query" and parsed_result.ok:
                        rag_used = True

                    # ── PR 2：通过 commit_tool_result_once 提交 ──
                    committed = await commit_tool_result_once(
                        session, recovery_state, tc.id, result, on_message,
                    )

                    if committed:
                        if is_error:
                            recovery_state.mark_failed(tc.id, parsed_result.error)
                            if on_tool:
                                on_tool("error", tc.name, tc.arguments, result, elapsed)
                        else:
                            recovery_state.mark_completed(tc.id)
                            if on_tool:
                                on_tool("end", tc.name, tc.arguments, result, elapsed)

                    # TOOL_COMPLETED 事件
                    await self._emit_recovery_state(on_recovery_state, recovery_state)

                    # PR 3：收集 batch 事件摘要（用于触发评估）
                    if reflexion_state is not None:
                        record = recovery_state.get_record(tc.id)
                        envelope = None
                        if is_error:
                            envelope = classify_tool_result(result, tool_name=tc.name)
                        conflict, conflict_detail = _extract_conflict(parsed_result)
                        batch_events.append(ToolEventSummary(
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            action_fingerprint=compute_action_fingerprint(tc.name, tc.arguments),
                            ok=not is_error,
                            error_code=envelope.error_code if envelope else None,
                            retryable=envelope.retryable if envelope else None,
                            outcome=envelope.outcome if envelope else None,
                            attempts=record.attempts if record else 1,
                            idempotency=record.idempotency if record else "non_idempotent",
                            has_conflict=conflict,
                            conflict_detail=conflict_detail,
                            # 只放脱敏、限长后的摘要（不进入状态/反思 prompt 原文）
                            summary=sanitize_error(parsed_result.summary)[:200] if not is_error else "",
                        ))

                    # 更新 task state
                    task_mgr.update_from_tool(tc.name, tc.arguments, result)

                    # 清理 tool_context
                    tool_context.pop("_idempotency_key", None)
                    tool_context.pop("_action_fingerprint", None)

                # 工具循环结束后推送 task state
                if on_task_state and task_mgr.state:
                    on_task_state(task_mgr.state.to_dict())

                # 协议完整性检查（不再是 warning，而是终态化）
                incomplete = recovery_state.check_completeness()
                if incomplete:
                    logger.error(
                        "Protocol incompleteness: %d tool_call_id(s) missing result: %s",
                        len(incomplete), incomplete,
                    )
                    # 终态化缺失的 calls
                    for tc_id in incomplete:
                        record = recovery_state.get_record(tc_id)
                        if record and record.status in (ToolCallStatus.PENDING, ToolCallStatus.EXECUTING):
                            if record.idempotency == "non_idempotent":
                                status = ToolCallStatus.UNKNOWN_OUTCOME
                            else:
                                status = ToolCallStatus.INTERRUPTED
                            synthetic = _make_synthetic_result(tc_id, record.tool_name, status, "Incompleteness detected")
                            await commit_tool_result_once(session, recovery_state, tc_id, synthetic, on_message)
                            recovery_state.mark_tool_call_terminal(tc_id, status, "Incompleteness detected")

                # ── PR 3：每个工具 batch 最多触发一次 Reflexion ──
                # 前置条件：所有 tool results 已终态提交（协议完整性已保证）
                if reflexion_state is not None and batch_events:
                    await self._maybe_reflect(
                        session=session,
                        task_mgr=task_mgr,
                        reflexion_state=reflexion_state,
                        reflexion_engine=reflexion_engine,
                        progress_tracker=progress_tracker,
                        reflexion_failure_counts=reflexion_failure_counts,
                        batch_events=batch_events,
                        on_reflexion_event=on_reflexion_event,
                        should_cancel=should_cancel,
                        deadline=deadline,
                        iteration=iteration,
                        available_tool_names={
                            t["function"]["name"] for t in tools
                        },
                    )
                    # ReflexionState 变化后推送快照（供上层持久化 / 进程恢复）
                    await self._emit_reflexion_state(on_reflexion_state, reflexion_state)

                # 压缩检查
                await self._maybe_auto_compact(session, system_prompt=effective_prompt, autosave=autosave, on_compact=on_compact)

            # ── 达到最大迭代次数 ──
            await terminalize_on_max_iterations(recovery_state, session)
            await self._emit_recovery_state(on_recovery_state, recovery_state)
            return "达到最大迭代次数（{}），请简化问题后重试。".format(self.max_iterations)
        finally:
            task_mgr.clear()

    @staticmethod
    async def _emit_message(
        callback: Callable[[dict], Awaitable[None] | None] | None,
        message: dict,
    ) -> None:
        if callback is None:
            return
        result = callback(dict(message))
        if asyncio.iscoroutine(result):
            await result

    @staticmethod
    async def _emit_verification(
        callback: Callable[[dict], Awaitable[None] | None] | None,
        report: dict,
    ) -> None:
        if callback is None:
            return
        result = callback(dict(report))
        if asyncio.iscoroutine(result):
            await result

    @staticmethod
    async def _emit_recovery_state(
        callback: Callable[[dict], Awaitable[None] | None] | None,
        state: RecoveryState,
    ) -> None:
        """推送 RecoveryState 快照"""
        if callback is None:
            return
        result = callback(state.to_dict())
        if asyncio.iscoroutine(result):
            await result

    async def run_reviewer(
        self,
        prompt: str,
        on_text: Callable[[str], None] | None = None,
    ) -> str:
        """用评审模型独立评估。"""
        if not self.reviewer_llm:
            return "Error: 评审模型未配置。请设置 NOVARE_REVIEWER_API_KEY 等环境变量。"

        messages = [
            {"role": "system", "content": "你是一个独立的研究评审专家。请根据提供的候选创新点和相关论文，给出客观的评审意见。输出 JSON 格式。"},
            {"role": "user", "content": prompt},
        ]

        budget = RetryBudget(max_retries=self.max_retries_per_turn)
        deadline = time.monotonic() + self.turn_timeout
        response = await self._collect_stream_with_retry(
            self.reviewer_llm, messages, tools=None, on_text=on_text,
            budget=budget, deadline=deadline,
        )
        return response.content or ""

    # ── PR 1：LLM / 工具重试 ──────────────────────────────────────

    def _llm_retry_policy(self) -> RetryPolicy:
        """LLM 重试策略"""
        return RetryPolicy(
            max_attempts=self.llm_retry_attempts,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
            retry_after_max_delay=self.retry_after_max_delay,
        )

    def _tool_retry_policy(self, name: str) -> RetryPolicy:
        """查询工具的重试策略并强制执行幂等保护。"""
        declared: RetryPolicy | None = None
        retry_getter = getattr(self.tool_registry, "retry_policy_for", None)
        if callable(retry_getter):
            try:
                candidate = retry_getter(name)
                if isinstance(candidate, RetryPolicy):
                    declared = candidate
            except Exception:
                declared = None

        idempotency = "non_idempotent"
        idem_getter = getattr(self.tool_registry, "idempotency_for", None)
        if callable(idem_getter):
            try:
                candidate = idem_getter(name)
                if candidate in ("read", "idempotent_write", "non_idempotent"):
                    idempotency = candidate
            except Exception:
                idempotency = "non_idempotent"

        if idempotency == "non_idempotent":
            max_attempts = 1
        else:
            max_attempts = declared.max_attempts if declared else 1

        return RetryPolicy(
            max_attempts=max_attempts,
            base_delay=self.retry_base_delay,
            max_delay=self.retry_max_delay,
            retry_after_max_delay=self.retry_after_max_delay,
            backoff_factor=declared.backoff_factor if declared else 2.0,
            jitter=declared.jitter if declared else True,
        )

    async def _collect_stream_with_retry(
        self,
        llm_client,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_text: Callable[[str], None] | None = None,
        budget: RetryBudget | None = None,
        deadline: float | None = None,
    ):
        """带传输层重试的 collect_stream。"""
        emitted = False

        if on_text is None:
            stream_on_text = None
        else:
            def _on_text(chunk: str):
                nonlocal emitted
                emitted = True
                on_text(chunk)
            stream_on_text = _on_text

        def _abort() -> bool:
            return emitted

        def _on_retry(attempt: int, max_attempts: int, delay: float, error_code: str):
            logger.warning(
                "LLM stream retry %d/%d after %.2fs (error_code=%s, emitted=%s)",
                attempt, max_attempts, delay, error_code, emitted,
            )

        executor = RetryExecutor(
            self._llm_retry_policy(),
            budget,
            sleep=self._retry_sleep,
            random_fn=self._retry_random,
            deadline=deadline,
            on_retry=_on_retry,
        )
        outcome = await executor.run(
            lambda: llm_client.collect_stream(
                messages, tools=tools, on_text=stream_on_text,
            ),
            classify_exception,
            abort_retry=_abort,
        )
        return outcome.result

    def _tool_on_retry(
        self,
        on_tool: Callable[[str, str, dict, str | None, float | None], None] | None,
        name: str,
        arguments: dict,
    ) -> Callable[[int, int, float, str], None]:
        """构造 on_tool 的 retry 事件回调。"""
        def _on_retry(attempt: int, max_attempts: int, delay: float, error_code: str):
            if on_tool:
                info = json.dumps({
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "delay": round(delay, 3),
                    "error_code": error_code,
                }, ensure_ascii=False)
                on_tool("retry", name, arguments, info, None)
        return _on_retry

    # ── PR 3：Reflexion 集成 ──────────────────────────────────────

    def _make_reflexion_engine(
        self,
        on_reflexion_event: Callable[[str, dict], Awaitable[None] | None] | None,
        *,
        budget: RetryBudget | None = None,
        turn_deadline: float | None = None,
    ) -> ReflexionEngine | None:
        """创建 ReflexionEngine。

        模型选择：配置了 reviewer_llm 时默认使用 reviewer_llm（独立 client），
        否则回退主 llm_client。必须显式传递 client，不允许复用错误模型。
        共享 turn RetryBudget 与 turn deadline 一并传入：
        Reflexion 的额外 transport retry 消耗同一 RetryBudget，
        整次反思受 min(turn_deadline, start + reflexion_timeout) 约束。
        """
        if not self.reflexion_enabled:
            return None
        reflexion_llm = self.reviewer_llm or self.llm_client
        return ReflexionEngine(
            reflexion_llm,
            self.tool_registry,
            max_reflections_per_turn=self.max_reflections_per_turn,
            timeout=self.reflexion_timeout,
            max_tokens=self.reflexion_max_tokens,
            retry_policy=self._llm_retry_policy(),
            budget=budget,
            turn_deadline=turn_deadline,
            sleep=self._reflexion_sleep,
            random_fn=self._retry_random,
            event_callback=(
                (lambda event_type, payload: self._emit_reflexion_event(
                    on_reflexion_event, event_type, payload,
                ))
                if on_reflexion_event
                else None
            ),
        )

    @staticmethod
    async def _emit_reflexion_event(
        callback: Callable[[str, dict], Awaitable[None] | None] | None,
        event_type: str,
        payload: dict,
    ) -> None:
        if callback is None:
            return
        result = callback(event_type, payload)
        if asyncio.iscoroutine(result):
            await result

    @staticmethod
    async def _emit_reflexion_state(
        callback: Callable[[dict], Awaitable[None] | None] | None,
        state: ReflexionState,
    ) -> None:
        """推送 ReflexionState 快照（供持久化 / 进程恢复）。"""
        if callback is None:
            return
        result = callback(state.to_dict())
        if asyncio.iscoroutine(result):
            await result

    async def _maybe_reflect(
        self,
        *,
        session,
        task_mgr: TaskStateManager,
        reflexion_state: ReflexionState,
        reflexion_engine: ReflexionEngine | None,
        progress_tracker: ProgressTracker | None,
        reflexion_failure_counts: dict[str, int],
        batch_events: list,
        on_reflexion_event: Callable[[str, dict], Awaitable[None] | None] | None,
        should_cancel,
        deadline: float | None,
        iteration: int,
        available_tool_names: set[str],
    ) -> None:
        """每个工具 batch 结束后评估并执行一次 Reflexion（最多一次）。

        - 取消与 deadline 优先于 Reflexion。
        - 先更新 progress（no_progress_count），再评估触发器。
        - 不写 tool result；不破坏 PR 2 的 pending tool-call 完整性
          （调用方保证所有结果已终态提交）。
        """
        # 取消 / deadline 优先
        if should_cancel:
            _cancel_result = should_cancel()
            if asyncio.iscoroutine(_cancel_result):
                _cancel_result = await _cancel_result
            if _cancel_result:
                return
        if deadline is not None and time.monotonic() >= deadline:
            return

        # ── 更新进展指纹（确定性、digest 化信号）──
        # 成功信号转为 64-hex digest（tool + action_fingerprint + SHA-256(summary)），
        # 状态中只保存 digest，不保存明文 summary / 文本；
        # pending 文本变化不算真实进展
        if progress_tracker is not None and task_mgr.state is not None:
            success_signal_digests: list[str] = []
            for ev in batch_events:
                if not ev.ok:
                    continue
                success_signal_digests.append(
                    progress_signal_digest(
                        kind="tool_success",
                        tool=ev.tool_name,
                        action_fingerprint=ev.action_fingerprint,
                        summary_digest=hashlib.sha256(
                            ev.summary.encode("utf-8")
                        ).hexdigest(),
                    )
                )
            made_progress = progress_tracker.update(
                completed=task_mgr.state.completed,
                key_findings=task_mgr.state.key_findings,
                success_signal_digests=success_signal_digests,
            )
            if made_progress:
                reflexion_state.record_progress()
            else:
                reflexion_state.record_no_progress()
            reflexion_state.last_progress_fingerprint = progress_tracker.last_progress_fingerprint
            # 同步累计信号回 ReflexionState（供快照持久化 / 跨进程恢复）
            progress_tracker.sync_to_state(reflexion_state)

        # ── 更新连续失败计数（成功清零）──
        self._update_failure_counts(
            batch_events, reflexion_failure_counts, reflexion_state,
        )

        # ── 确定性触发评估 ──
        evaluation = evaluate_triggers(
            events=batch_events,
            reflexion_state=reflexion_state,
            recent_failure_counts=reflexion_failure_counts,
            no_progress_count=reflexion_state.no_progress_count,
            max_reflections_per_turn=self.max_reflections_per_turn,
            repeated_failure_threshold=self.reflexion_repeated_failure_threshold,
            no_progress_threshold=self.reflexion_no_progress_threshold,
            available_tool_names=available_tool_names,
            last_progress_fingerprint=(
                progress_tracker.last_progress_fingerprint if progress_tracker else None
            ),
        )
        if not evaluation.triggered:
            return

        await self._emit_reflexion_event(on_reflexion_event, "REFLECTION_TRIGGERED", {
            "trigger": evaluation.trigger.value,
            "trigger_fingerprint": evaluation.trigger_fingerprint,
            "evidence_refs": evaluation.evidence_refs,
            "iteration": iteration,
        })

        # ── 触发指纹去重（引擎内部也会再次检查）──
        if reflexion_state.already_reflected(evaluation.trigger_fingerprint):
            return

        # ── 构建最小化反思上下文 ──
        # 触发事件从 evaluation.evidence_refs 推导。
        # CONFLICTING_OBSERVATIONS 是"成功调用触发"的唯一结构化例外：
        # 该动作执行成功，不是 failed action，不得要求禁止重复执行
        failed_event = None
        if not evaluation.is_conflict_success and evaluation.evidence_refs:
            event_id = evaluation.evidence_refs[0].removeprefix("event:")
            for ev in batch_events:
                if ev.tool_call_id == event_id:
                    failed_event = ev
                    break
        context = ReflectionContext(
            user_goal=task_mgr.state.goal if task_mgr.state else "",
            current_plan=list(task_mgr.state.completed) if task_mgr.state else [],
            pending_steps=list(task_mgr.state.pending) if task_mgr.state else [],
            event_summaries=[
                {
                    "event_id": ev.tool_call_id,
                    "tool_name": ev.tool_name,
                    "ok": ev.ok,
                    "error_code": ev.error_code,
                    "attempts": ev.attempts,
                    "outcome": ev.outcome,
                    "summary": ev.summary,
                }
                for ev in batch_events[-self.reflexion_max_recent_events:]
            ],
            failure_classification=evaluation.trigger.value,
            error_code=failed_event.error_code if failed_event else None,
            action_fingerprint=failed_event.action_fingerprint if failed_event else None,
            available_tool_names=sorted(available_tool_names),
            remaining_iterations=max(0, self.max_iterations - iteration - 1),
            remaining_time_seconds=max(0.0, (deadline or time.monotonic()) - time.monotonic()),
            safety_constraints=["用户目标不可变", "不得扩大工具权限"],
            real_event_ids=[ev.tool_call_id for ev in batch_events],
            trigger_evidence_refs=list(evaluation.evidence_refs),
            triggering_action_fingerprint=failed_event.action_fingerprint if failed_event else None,
            failed_tool=failed_event.tool_name if failed_event else None,
            failed_arguments=None,
            idempotency=failed_event.idempotency if failed_event else "non_idempotent",
        )

        if reflexion_engine is None:
            return
        record = await reflexion_engine.reflect(
            context,
            reflexion_state,
            trigger=evaluation.trigger.value,
            trigger_fingerprint=evaluation.trigger_fingerprint,
            evidence_refs=evaluation.evidence_refs,
        )

        # 反思成功应用（validated+applied）→ 计划已修订
        if record is not None:
            await self._emit_reflexion_event(on_reflexion_event, "PLAN_REVISED", {
                "reflection_id": record.reflection_id,
                "trigger": record.trigger,
                "decision": record.decision,
            })
            # 反思本身不是进展，不重置 no_progress_count

    def _update_failure_counts(
        self,
        batch_events: list,
        reflexion_failure_counts: dict[str, int],
        reflexion_state: ReflexionState,
    ) -> None:
        """维护 turn 级连续失败计数：成功清零，可计失败 +1。

        触发 REPEATED_FAILED_ACTION 后对应 fingerprint 清零，避免每轮重复触发。
        """
        for ev in batch_events:
            if ev.ok:
                reflexion_failure_counts.pop(ev.action_fingerprint, None)
                continue
            code = (ev.error_code or "").upper()
            if (
                is_transient_error_code(code)
                or is_terminal_error_code(code)
                or is_semantic_error_code(code)
            ):
                continue
            if code == "UNKNOWN_OUTCOME" and ev.idempotency == "non_idempotent":
                continue
            reflexion_failure_counts[ev.action_fingerprint] = (
                reflexion_failure_counts.get(ev.action_fingerprint, 0) + 1
            )
        # 触发过 repeated 的指纹清零（避免循环触发）
        triggered_fp = {
            ev.action_fingerprint
            for ev in batch_events
            if not ev.ok
            and reflexion_state.already_reflected(f"repeated_failed_action:{ev.action_fingerprint}")
        }
        for fp in triggered_fp:
            reflexion_failure_counts.pop(fp, None)

    def _build_recovery_context_block(self, reflexion_state: ReflexionState) -> str:
        """构建注入 system prompt 的私有 [Recovery Context] 块。

        只包含 diagnosis / changes / revised plan / 禁止重复动作 / 建议下一步；
        不写 chain-of-thought；不进入普通消息历史。
        suggested_next_action 仅为建议（脱敏、限长），不自动执行。
        """
        lines = ["[Recovery Context]"]
        for record in reflexion_state.records:
            lines.append(f"- 反思（{record.trigger}）: {record.diagnosis}")
            if record.changes:
                lines.append("  变更: " + "; ".join(record.changes))
            if record.revised_plan:
                lines.append("  修订计划: " + "; ".join(record.revised_plan))
            suggested = record.suggested_next_action
            if isinstance(suggested, dict):
                suggested_tool = suggested.get("tool")
                suggested_args = suggested.get("arguments")
                # malformed 建议不注入
                if isinstance(suggested_tool, str) and suggested_tool and isinstance(suggested_args, dict):
                    suggested_fp = compute_action_fingerprint(suggested_tool, suggested_args)
                    # 重新检查当前 forbidden 集合：已被反思禁止的动作不注入
                    if suggested_fp in reflexion_state.forbidden_action_fingerprints:
                        logger.debug(
                            "Skipping suggested_next_action for %s: fingerprint now forbidden",
                            suggested_tool,
                        )
                    else:
                        # 脱敏、限长的建议（仅作参考，不自动执行）
                        try:
                            suggested_raw = json.dumps(suggested, ensure_ascii=False)[:200]
                        except (TypeError, ValueError):
                            suggested_raw = str(suggested)[:200]
                        lines.append("  建议下一步: " + sanitize_error(suggested_raw))
        if reflexion_state.forbidden_action_fingerprints:
            lines.append(
                "禁止重复动作: " + ", ".join(sorted(reflexion_state.forbidden_action_fingerprints))
            )
        return "\n".join(lines)

    def _build_messages(self, session, task_state: TaskState | None = None, system_prompt: str | None = None, reflexion_state: ReflexionState | None = None) -> list[dict]:
        """构建发送给 LLM 的消息列表"""
        effective = system_prompt if system_prompt is not None else self.system_prompt
        messages = []
        if effective:
            system_content = effective
            if task_state:
                system_content += "\n\n" + task_state.to_prompt_block()
            if reflexion_state is not None and reflexion_state.records:
                system_content += "\n\n" + self._build_recovery_context_block(reflexion_state)
            messages.append({"role": "system", "content": system_content})
        allowed_keys = ("role", "content", "name", "tool_calls", "tool_call_id")
        messages.extend([
            {key: value for key, value in message.items() if key in allowed_keys}
            for message in session.messages
        ])
        return messages

    async def _preflight_compact(self, session, messages: list[dict], task_state: TaskState | None = None, system_prompt: str | None = None, autosave: bool = True, on_compact: Callable[[object], Awaitable[None] | None] | None = None) -> bool:
        """LLM 调用前的预检：估算总量，超过阈值则先压缩。"""
        if self.auto_compact_threshold <= 0:
            return False

        preflight_threshold = int(self.auto_compact_threshold * 0.8)
        tools = self.tool_registry.to_openai_tools()
        estimated = estimate_messages_tokens(messages) + estimate_tools_tokens(tools)

        if estimated < preflight_threshold and not self.context_compactor.needs_compaction(session.messages):
            return False

        return await self._compact_session(session, autosave=autosave, on_compact=on_compact, reason="preflight")

    async def _maybe_auto_compact(self, session, system_prompt: str | None = None, autosave: bool = True, on_compact: Callable[[object], Awaitable[None] | None] | None = None) -> bool:
        """检查是否需要自动压缩"""
        if self.auto_compact_threshold <= 0:
            return False

        if (
            not session.usage_tracker.should_compact(self.auto_compact_threshold)
            and not self.context_compactor.needs_compaction(session.messages)
        ):
            return False

        return await self._compact_session(session, autosave=autosave, on_compact=on_compact, reason="post_turn")

    async def _compact_session(self, session, *, autosave: bool, on_compact: Callable[[object], Awaitable[None] | None] | None, reason: str) -> bool:
        old_count = len(session.messages)
        old_tokens = estimate_messages_tokens(session.messages)
        result = await self.context_compactor.compact(session.messages)
        if not result.did_compact:
            if result.budget_overflow:
                logger.warning("Context budget overflow cannot be reduced safely: tokens=%d budget=%d", result.estimated_tokens, self.context_compactor.token_budget)
            return False

        session.messages = result.messages
        session.usage_tracker.reset_after_compact()
        if autosave:
            session.save()
        if on_compact:
            callback_result = on_compact(session)
            if asyncio.iscoroutine(callback_result):
                await callback_result
        logger.info(
            "Context compaction complete: reason=%s strategy=%s messages=%d->%d tokens=%d->%d turns=%d overflow=%s llm_calls=%d",
            reason, result.strategy, old_count, len(result.messages), old_tokens, result.estimated_tokens, result.selected_turns, result.budget_overflow, result.llm_calls,
        )
        return True
