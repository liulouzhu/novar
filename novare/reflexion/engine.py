"""novare/reflexion/engine.py — ReflexionEngine

流程：
1. 检查 budget / 触发去重（由调用方传入 trigger fingerprint）
2. 构建最小结构化 prompt
3. 调用反思模型（复用 PR 1 传输层 Retry，消耗共享 turn RetryBudget；
   asyncio.timeout 包住真实 await，截止 = min(turn_deadline, start + timeout)）
4. 解析 JSON；malformed 允许一次格式修复，之后安全失败
5. ReflectionValidator 验证
6. 通过 → 生成 ReflectionRecord 并应用（更新 ReflexionState）
   失败 → 记录 REFLECTION_REJECTED，不应用计划修改

故障隔离：
- 反思模型故障（认证错误、永久 503、RetryExhaustedError、超时、其他异常）
  不终止主 Agent turn：记录 REFLECTION_FAILED，消耗一次 budget，
  记录 trigger fingerprint 防止无限调用，返回 None。
- asyncio.CancelledError 必须立即传播（外部 turn 取消），不得降级。
- 不保存原始模型输出 / chain-of-thought；只保存 validated+applied 的简洁记录。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from novare.recovery.executor import RetryExecutor
from novare.recovery.policy import RetryBudget, RetryPolicy
from novare.recovery.classifier import classify_exception, sanitize_error
from novare.reflexion.prompts import REFLECTION_SYSTEM_PROMPT, build_reflection_user_prompt
from novare.reflexion.types import (
    ReflectionDecision,
    ReflectionRecord,
    ReflexionState,
    make_reflection_record,
)
from novare.reflexion.validator import ReflectionValidator, parse_model_json

logger = logging.getLogger("novare.reflexion")

# 事件回调：event_type(str) + payload(dict)
EventCallback = Callable[[str, dict], Awaitable[None] | None]

# 反思模型可能输出的"重试相同动作"决策（禁止）
_FORBIDDEN_DECISIONS = {"RETRY", "RETRY_SAME", "REPEAT", "RETRY_ACTION"}


@dataclass
class ReflectionContext:
    """反思模型的输入上下文（全部脱敏、最小化）。"""

    user_goal: str
    current_plan: list[str] = field(default_factory=list)
    pending_steps: list[str] = field(default_factory=list)
    event_summaries: list[dict] = field(default_factory=list)
    failure_classification: str = ""
    error_code: str | None = None
    action_fingerprint: str | None = None
    available_tool_names: list[str] = field(default_factory=list)
    remaining_iterations: int = 0
    remaining_time_seconds: float = 0.0
    safety_constraints: list[str] = field(default_factory=list)
    real_event_ids: list[str] = field(default_factory=list)
    trigger_evidence_refs: list[str] = field(default_factory=list)
    triggering_action_fingerprint: str | None = None
    failed_tool: str | None = None
    failed_arguments: dict | None = None
    idempotency: str = "non_idempotent"


class ReflexionEngine:
    """反思引擎。llm_client 必须由调用方显式传入（reviewer 优先，主模型回退）。"""

    def __init__(
        self,
        llm_client,
        tool_registry=None,
        *,
        max_reflections_per_turn: int = 2,
        timeout: float = 30.0,
        max_tokens: int = 1200,
        retry_policy: RetryPolicy | None = None,
        budget: RetryBudget | None = None,
        turn_deadline: float | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random_fn=None,
        event_callback: EventCallback | None = None,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.max_reflections_per_turn = max_reflections_per_turn
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.retry_policy = retry_policy or RetryPolicy(
            max_attempts=3,
            base_delay=0.5,
            max_delay=8.0,
            retry_after_max_delay=30.0,
        )
        # 共享 turn RetryBudget：Reflexion 的额外 transport retry 消耗同一预算
        self.budget = budget
        # turn absolute deadline（time.monotonic()）
        self.turn_deadline = turn_deadline
        self._sleep = sleep
        self._random_fn = random_fn
        self.event_callback = event_callback
        # 本次 reflection 的起始时间（用于整次 wall-clock 限制）
        self._reflection_start: float | None = None

    async def _emit(self, event_type: str, payload: dict) -> None:
        if not self.event_callback:
            return
        result = self.event_callback(event_type, payload)
        if asyncio.iscoroutine(result):
            await result

    async def reflect(
        self,
        context: ReflectionContext,
        reflexion_state: ReflexionState,
        trigger: str,
        trigger_fingerprint: str,
        evidence_refs: list[str],
    ) -> ReflectionRecord | None:
        """执行一次反思。返回已应用（validated+applied）的记录，失败/拒绝返回 None。

        模型故障（认证错误、永久 503、重试耗尽、超时等）不冒泡到主 Agent turn；
        asyncio.CancelledError 立即传播。
        """
        if reflexion_state.budget_exhausted(self.max_reflections_per_turn):
            await self._emit("REFLECTION_BUDGET_EXHAUSTED", {
                "trigger": trigger,
                "reflection_count": reflexion_state.reflection_count,
                "max_reflections": self.max_reflections_per_turn,
            })
            return None
        if reflexion_state.already_reflected(trigger_fingerprint):
            return None

        self._reflection_start = time.monotonic()
        try:
            return await self._reflect_inner(
                context, reflexion_state, trigger, trigger_fingerprint, evidence_refs,
            )
        except asyncio.CancelledError:
            # 外部 turn 取消必须立即传播，不得降级为反思失败
            raise
        except Exception as exc:
            await self._fail_model(reflexion_state, trigger, trigger_fingerprint, exc)
            return None

    async def _reflect_inner(
        self,
        context: ReflectionContext,
        reflexion_state: ReflexionState,
        trigger: str,
        trigger_fingerprint: str,
        evidence_refs: list[str],
    ) -> ReflectionRecord | None:
        await self._emit("REFLECTION_STARTED", {
            "trigger": trigger,
            "trigger_fingerprint": trigger_fingerprint,
        })

        user_prompt = build_reflection_user_prompt(
            user_goal=context.user_goal,
            current_plan=context.current_plan,
            pending_steps=context.pending_steps,
            event_summaries=context.event_summaries,
            failure_classification=context.failure_classification,
            error_code=context.error_code,
            action_fingerprint=context.action_fingerprint,
            forbidden_action_fingerprints=sorted(reflexion_state.forbidden_action_fingerprints),
            remaining_iterations=context.remaining_iterations,
            remaining_reflections=max(0, self.max_reflections_per_turn - reflexion_state.reflection_count),
            remaining_time_seconds=context.remaining_time_seconds,
            available_tools=context.available_tool_names,
            safety_constraints=context.safety_constraints,
        )

        raw_output = await self._call_model(user_prompt)
        output = parse_model_json(raw_output)

        # ── 一次格式修复（受控修复调用，不算 transport retry）──
        if output is None:
            logger.info("Reflexion output malformed; attempting one format repair")
            await self._emit("REFLECTION_FORMAT_REPAIR", {
                "trigger": trigger,
                "trigger_fingerprint": trigger_fingerprint,
            })
            repair_prompt = (
                user_prompt
                + "\n\n注意：你上一次的输出不是合法 JSON。请只输出一个合法 JSON 对象，不要任何其他文字。"
            )
            raw_output = await self._call_model(repair_prompt)
            output = parse_model_json(raw_output)

        if output is None:
            await self._reject(reflexion_state, trigger, trigger_fingerprint, "malformed json after repair")
            return None

        # 禁止"自动重试相同动作"决策
        decision = output.get("decision")
        if isinstance(decision, str) and decision.upper() in _FORBIDDEN_DECISIONS:
            await self._reject(reflexion_state, trigger, trigger_fingerprint, "forbidden retry-same-action decision")
            return None

        validator = ReflectionValidator(
            tool_registry=self.tool_registry,
            available_tool_names=context.available_tool_names,
            user_goal=context.user_goal,
            safety_constraints=context.safety_constraints,
            real_event_ids=context.real_event_ids,
            triggering_action_fingerprint=context.triggering_action_fingerprint,
            required_evidence_refs=context.trigger_evidence_refs,
            existing_forbidden_action_fingerprints=reflexion_state.forbidden_action_fingerprints,
            failed_tool=context.failed_tool,
            failed_arguments=context.failed_arguments,
            idempotency=context.idempotency,
        )
        normalized, reason = validator.normalize_and_validate(output)

        if normalized is None:
            await self._reject(reflexion_state, trigger, trigger_fingerprint, reason)
            return None

        # 只使用 Validator 归一化后的结构（不直接迭代未归一化模型输出）；
        # 反思输出文本脱敏后入 record（防御：模型可能把工具摘要中的敏感信息抄进来）
        forbidden = normalized["forbidden_repeat"]
        record = make_reflection_record(
            trigger=trigger,
            trigger_fingerprint=trigger_fingerprint,
            evidence_refs=list(normalized["evidence_refs"]),
            failure_type=sanitize_error(normalized["failure_type"])[:80],
            diagnosis=sanitize_error(normalized["diagnosis"]),
            preserve=[sanitize_error(x) for x in normalized["preserve"]],
            changes=[sanitize_error(x) for x in normalized["changes"]],
            forbidden_action_fingerprints=list(forbidden),
            revised_plan=[sanitize_error(x) for x in normalized["revised_plan"]],
            suggested_next_action=normalized["suggested_next_action"],
            decision=normalized["decision"],
            validated=True,
            applied=True,
        )
        reflexion_state.add_reflection(record)

        await self._emit("REFLECTION_COMMITTED", {
            "reflection_id": record.reflection_id,
            "trigger": record.trigger,
            "trigger_fingerprint": record.trigger_fingerprint,
            "decision": record.decision,
            "failure_type": record.failure_type,
        })
        logger.info(
            "Reflection committed: trigger=%s decision=%s diagnosis=%s",
            record.trigger, record.decision, record.diagnosis[:60],
        )
        return record

    async def _reject(
        self,
        reflexion_state: ReflexionState,
        trigger: str,
        trigger_fingerprint: str,
        reason: str,
        detail: dict | None = None,
    ) -> None:
        # 拒绝也计入 reflected_trigger_fingerprints（避免同一触发反复调用模型）
        # 且消耗反思预算（reflection_count 计为一次尝试，防止无限调用）
        reflexion_state.reflected_trigger_fingerprints.add(trigger_fingerprint)
        reflexion_state.reflection_count += 1
        # 不持久化原始模型输出：只记录脱敏、限长的拒绝原因
        payload: dict = {
            "trigger": trigger,
            "trigger_fingerprint": trigger_fingerprint,
            "reason": sanitize_error(reason)[:300],
        }
        await self._emit("REFLECTION_REJECTED", payload)
        logger.warning("Reflection rejected: trigger=%s reason=%s", trigger, reason)

    async def _fail_model(
        self,
        reflexion_state: ReflexionState,
        trigger: str,
        trigger_fingerprint: str,
        exc: BaseException,
    ) -> None:
        """反思模型调用失败：消耗 budget、记录 fingerprint、发出 REFLECTION_FAILED。

        不生成 ReflectionRecord、不写 tool result、不影响已提交的工具结果。
        """
        reflexion_state.reflected_trigger_fingerprints.add(trigger_fingerprint)
        reflexion_state.reflection_count += 1
        envelope = classify_exception(exc)
        await self._emit("REFLECTION_FAILED", {
            "trigger": trigger,
            "trigger_fingerprint": trigger_fingerprint,
            "error_code": envelope.error_code,
            "reason": f"model_call_failed:{envelope.error_code}",
        })
        logger.warning(
            "Reflection model call failed: trigger=%s error_code=%s (%s)",
            trigger, envelope.error_code, type(exc).__name__,
        )

    async def _call_model(self, user_prompt: str) -> str:
        """调用反思模型（复用 PR 1 传输层 Retry + 共享 RetryBudget）。

        - 真实 LLM await 被 asyncio.timeout 包住：截止时间 =
          min(turn_deadline, reflection_start + reflexion_timeout)
        - 超时会取消正在运行的模型调用，不留下后台 task
        - 不递归触发 Reflexion（本方法不评估触发器）
        - 外部 CancelledError 立即传播
        """
        messages = [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        # 整次 reflection 的 wall-clock 截止时间（含 repair 调用）
        start = self._reflection_start or time.monotonic()
        deadline = min(
            self.turn_deadline if self.turn_deadline is not None else float("inf"),
            start + self.timeout,
        )
        remaining = max(0.0, deadline - time.monotonic())

        executor = RetryExecutor(
            self.retry_policy,
            self.budget,  # 共享 turn RetryBudget：额外 transport retry 消耗同一预算
            sleep=self._sleep,
            random_fn=self._random_fn,
            deadline=deadline,
            on_retry=lambda attempt, max_attempts, delay, error_code: logger.warning(
                "Reflection LLM retry %d/%d after %.2fs (%s)",
                attempt, max_attempts, delay, error_code,
            ),
        )
        # asyncio.wait_for 取消真实 LLM await（含重试与退避），兼容 Python 3.10+
        outcome = await asyncio.wait_for(
            executor.run(
                lambda: self.llm_client.collect_stream(messages, on_text=None, max_tokens=self.max_tokens),
                classify_exception,
            ),
            timeout=remaining,
        )
        return outcome.result.content or ""
