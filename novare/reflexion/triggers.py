"""novare/reflexion/triggers.py — 确定性触发规则

Reflexion 只能由确定性规则触发，禁止"每轮调用"或凭模型感觉判断。

支持触发器：
1. SEMANTIC_TOOL_FAILURE        — INVALID_ARGUMENT/BAD_REQUEST/UNKNOWN_TOOL/NO_HANDLER 等
2. REPEATED_FAILED_ACTION       — 相同 action fingerprint 连续失败达到阈值
3. NO_PROGRESS                  — 连续 N 个 iteration 无进展
4. CONFLICTING_OBSERVATIONS     — 结构化结果明确报告冲突
5. RETRY_EXHAUSTED_WITH_ALTERNATIVE — 瞬时错误重试耗尽且存在替代空间

禁止触发：
- 单次瞬时错误（PR 1 重试未耗尽 / 策略禁用）
- 成功工具调用
- TERMINAL 错误（401/403/缺 key，需用户处理）
- non_idempotent + UNKNOWN_OUTCOME
- 同一 trigger fingerprint 已反思且无新证据
- reflection budget 已耗尽
- 取消或 deadline 已到（由 AgentLoop 在调用前检查）
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from novare.reflexion.types import ReflexionState, ReflectionTrigger

# 语义失败（SEMANTIC）错误码
_SEMANTIC_ERROR_CODES = {
    "BAD_REQUEST",
    "INVALID_REQUEST",
    "INVALID_ARGUMENT",
    "UNKNOWN_TOOL",
    "NO_HANDLER",
    "NOT_ALLOWED",
    "NOT_FOUND",
    "SCHEMA_ERROR",
}

# TERMINAL 错误码（需要用户处理，禁止反思）
_TERMINAL_ERROR_CODES = {
    "UNAUTHORIZED",
    "FORBIDDEN",
    "PERMISSION_DENIED",
    "AUTHENTICATION_ERROR",
    "API_KEY_INVALID",
}

# TRANSIENT 错误码（属于 PR 1 重试范畴）
_TRANSIENT_ERROR_CODES = {
    "TIMEOUT",
    "UPSTREAM_TIMEOUT",
    "REQUEST_TIMEOUT",
    "GATEWAY_TIMEOUT",
    "CONNECTION_ERROR",
    "CONNECTION_RESET",
    "NETWORK_ERROR",
    "UPSTREAM_ERROR",
    "RATE_LIMITED",
    "TOO_EARLY",
    "INTERNAL_ERROR",
    "BAD_GATEWAY",
    "SERVICE_UNAVAILABLE",
    "TRANSIENT_ERROR",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ToolEventSummary:
    """一个工具调用的终态事件摘要（脱敏、结构化）。"""

    tool_call_id: str
    tool_name: str
    action_fingerprint: str
    ok: bool
    error_code: str | None = None
    retryable: bool | None = None
    outcome: str | None = None
    attempts: int = 1
    idempotency: str = "non_idempotent"
    has_conflict: bool = False
    conflict_detail: str | None = None
    summary: str = ""


@dataclass
class TriggerEvaluation:
    """一次评估的结果。"""

    triggered: bool = False
    trigger: ReflectionTrigger | None = None
    trigger_fingerprint: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    failure_type: str | None = None
    reason: str | None = None
    # CONFLICTING_OBSERVATIONS 是唯一"成功调用触发"的例外：
    # 该动作执行成功（ok=True），不是失败动作，不得被当作 failed action
    is_conflict_success: bool = False


def _is_transient(error_code: str | None) -> bool:
    return (error_code or "").upper() in _TRANSIENT_ERROR_CODES


def _is_terminal(error_code: str | None) -> bool:
    return (error_code or "").upper() in _TERMINAL_ERROR_CODES


def _is_semantic(error_code: str | None) -> bool:
    return (error_code or "").upper() in _SEMANTIC_ERROR_CODES


def is_transient_error_code(error_code: str | None) -> bool:
    """判断 error_code 是否属于 TRANSIENT（PR 1 重试范畴）。"""
    return _is_transient(error_code)


def is_terminal_error_code(error_code: str | None) -> bool:
    """判断 error_code 是否属于 TERMINAL（需用户处理，禁止反思）。"""
    return _is_terminal(error_code)


def is_semantic_error_code(error_code: str | None) -> bool:
    """判断 error_code 是否属于 SEMANTIC（有专属触发器，不计入 REPEATED）。"""
    return _is_semantic(error_code)


def _countable_failure(event: ToolEventSummary) -> bool:
    """是否属于可计入 REPEATED_FAILED_ACTION 的失败。

    排除：
    - 瞬时错误（属 PR 1 重试范畴，即使策略禁用也不反思）
    - TERMINAL（需用户处理）
    - SEMANTIC（有专属触发器 SEMANTIC_TOOL_FAILURE，避免叠加触发）
    - non_idempotent + UNKNOWN_OUTCOME（结果未知，禁止重放与反思）
    - 成功
    """
    if event.ok:
        return False
    code = (event.error_code or "").upper()
    if code in _TRANSIENT_ERROR_CODES:
        return False
    if code in _TERMINAL_ERROR_CODES:
        return False
    if code in _SEMANTIC_ERROR_CODES:
        return False
    if code == "UNKNOWN_OUTCOME" and event.idempotency == "non_idempotent":
        return False
    return True


def evaluate_triggers(
    *,
    events: list[ToolEventSummary],
    reflexion_state: ReflexionState,
    recent_failure_counts: dict[str, int],
    no_progress_count: int,
    max_reflections_per_turn: int,
    repeated_failure_threshold: int,
    no_progress_threshold: int,
    available_tool_names: set[str] | None = None,
    last_progress_fingerprint: str | None = None,
) -> TriggerEvaluation:
    """评估是否触发 Reflexion（一次评估最多返回一个触发器）。

    Args:
        events: 本 batch 已终态工具事件摘要
        reflexion_state: 当前 ReflexionState
        recent_failure_counts: turn 级连续失败计数（AgentLoop 维护，fp → count）
        no_progress_count: 当前无进展计数
        max_reflections_per_turn / repeated_failure_threshold / no_progress_threshold: 阈值
        available_tool_names: 当前可用工具名集合（判断替代空间）
    """
    if reflexion_state.budget_exhausted(max_reflections_per_turn):
        return TriggerEvaluation(
            triggered=False,
            reason=f"reflection budget exhausted ({reflexion_state.reflection_count}/{max_reflections_per_turn})",
        )

    if not events:
        return TriggerEvaluation(triggered=False)

    # ── 1. CONFLICTING_OBSERVATIONS（唯一"成功调用触发"的例外）──
    # 只允许 ok=True 且结构化冲突的结果触发；失败 conflict（ok=False）不是
    # conflict success，继续评估 semantic / repeated / retry-exhausted
    for ev in events:
        if ev.ok and ev.has_conflict:
            fp = f"conflicting_observations:{ev.tool_call_id}"
            if not reflexion_state.already_reflected(fp):
                return TriggerEvaluation(
                    triggered=True,
                    trigger=ReflectionTrigger.CONFLICTING_OBSERVATIONS,
                    trigger_fingerprint=fp,
                    evidence_refs=[f"event:{ev.tool_call_id}"],
                    failure_type="CONFLICTING_OBSERVATIONS",
                    reason=ev.conflict_detail or "structured conflict reported",
                    is_conflict_success=True,
                )

    # ── 2. REPEATED_FAILED_ACTION ──
    for ev in events:
        if not _countable_failure(ev):
            continue
        count = recent_failure_counts.get(ev.action_fingerprint, 0)
        if count >= repeated_failure_threshold:
            fp = f"repeated_failed_action:{ev.action_fingerprint}"
            if not reflexion_state.already_reflected(fp):
                return TriggerEvaluation(
                    triggered=True,
                    trigger=ReflectionTrigger.REPEATED_FAILED_ACTION,
                    trigger_fingerprint=fp,
                    evidence_refs=[f"event:{ev.tool_call_id}"],
                    failure_type=f"REPEATED_FAILED_ACTION_{ev.error_code or 'ERROR'}",
                    reason=f"action failed {count} consecutive times",
                )

    # ── 3. RETRY_EXHAUSTED_WITH_ALTERNATIVE ──
    for ev in events:
        if ev.ok or (ev.outcome or "") != "retry_exhausted":
            continue
        code = (ev.error_code or "").upper()
        if code not in _TRANSIENT_ERROR_CODES:
            continue
        # 替代空间：还有其他工具可用，或模型可换参数/换 provider（保守：当前工具仍可用即视为有空间）
        has_alternative = available_tool_names is None or len(available_tool_names) > 1
        if not has_alternative:
            continue
        fp = f"retry_exhausted:{ev.action_fingerprint}"
        if not reflexion_state.already_reflected(fp):
            return TriggerEvaluation(
                triggered=True,
                trigger=ReflectionTrigger.RETRY_EXHAUSTED_WITH_ALTERNATIVE,
                trigger_fingerprint=fp,
                evidence_refs=[f"event:{ev.tool_call_id}"],
                failure_type=f"RETRY_EXHAUSTED_{code}",
                reason=f"retry exhausted after {ev.attempts} attempts for {ev.tool_name}",
            )

    # ── 4. SEMANTIC_TOOL_FAILURE ──
    for ev in events:
        if ev.ok or not _is_semantic(ev.error_code):
            continue
        fp = f"semantic_tool_failure:{ev.action_fingerprint}:{(ev.error_code or 'ERROR').upper()}"
        if not reflexion_state.already_reflected(fp):
            return TriggerEvaluation(
                triggered=True,
                trigger=ReflectionTrigger.SEMANTIC_TOOL_FAILURE,
                trigger_fingerprint=fp,
                evidence_refs=[f"event:{ev.tool_call_id}"],
                failure_type=(ev.error_code or "SEMANTIC_ERROR").upper(),
                reason=f"semantic failure {ev.error_code} for {ev.tool_name}",
            )

    # ── 5. NO_PROGRESS ──
    # fingerprint 标识具体停滞阶段：no_progress:<最后一次真实进展指纹>
    # 同一停滞阶段只反思一次；出现真实进展后再次停滞可触发新的反思
    if no_progress_count >= no_progress_threshold:
        fp = f"no_progress:{last_progress_fingerprint or 'initial'}"
        if not reflexion_state.already_reflected(fp):
            return TriggerEvaluation(
                triggered=True,
                trigger=ReflectionTrigger.NO_PROGRESS,
                trigger_fingerprint=fp,
                evidence_refs=[],
                failure_type="NO_PROGRESS",
                reason=f"no progress for {no_progress_count} iterations",
            )

    return TriggerEvaluation(triggered=False)


def compute_action_fingerprint(tool_name: str, arguments: dict) -> str:
    """计算动作指纹（与 RecoveryState 语义一致，脱敏后 canonical JSON + SHA-256）。"""
    from novare.recovery.state import _compute_action_fingerprint

    return _compute_action_fingerprint(tool_name, arguments)
