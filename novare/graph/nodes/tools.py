"""novare/graph/nodes/tools.py — execute_tools 节点

保留 legacy AgentLoop 的全部工具执行语义：
- RecoveryState 批量注册 / commit_tool_result_once 幂等提交
- retry_tool_call（策略 + RetryBudget + turn deadline，非幂等工具不重试）
- 幂等键注入 tool_context（执行前写入、执行后清理）
- 结构化错误包装 + 完整性检查终态化
- TaskState 启发式更新、rag_used 检测、post-turn 自动压缩

与 legacy 的差异：
- 不再通过 getattr duck-typing 探测 retry/idempotency（ToolExecutor 协议
  的两个实现 ToolRegistry / SubagentToolExecutor 都显式提供该方法）
- rag 工具名集合提为模块常量，替代散落的字符串比较
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from novare.graph.context import RunContext, check_cancel
from novare.graph.state import GraphState
from novare.recovery.classifier import classify_tool_result, sanitize_error
from novare.recovery.executor import retry_tool_call
from novare.recovery.policy import RetryPolicy
from novare.recovery.state import (
    RecoveryState,
    ToolCallStatus,
    _make_synthetic_result,
    query_tool_retry_semantics,
)
from novare.recovery.terminalize import terminalize_on_cancel
from novare.recovery.types import Outcome
from novare.task_state import TaskState
from novare.tool_result import parse_tool_result

logger = logging.getLogger("novare.graph.nodes.tools")

# 触发 RAG 校验子图的工具名（未来可由 ToolDef 显式声明替代）
RAG_TOOL_NAMES = frozenset({"rag_query"})


async def _emit_recovery(ctx: RunContext) -> None:
    if ctx.on_recovery_state is None:
        return
    result = ctx.on_recovery_state(ctx.recovery_state.to_dict())
    if asyncio.iscoroutine(result):
        await result


def _tool_retry_policy(ctx: RunContext, name: str) -> RetryPolicy:
    """查询工具重试策略并强制幂等保护（非幂等工具 max_attempts=1）。

    探测逻辑统一在 recovery.state.query_tool_retry_semantics。
    """
    opts = ctx.options
    declared, idempotency = query_tool_retry_semantics(ctx.tools, name)
    max_attempts = 1 if idempotency == "non_idempotent" else (
        declared.max_attempts if declared else 1
    )
    return RetryPolicy(
        max_attempts=max_attempts,
        base_delay=opts.retry_base_delay,
        max_delay=opts.retry_max_delay,
        retry_after_max_delay=opts.retry_after_max_delay,
        backoff_factor=declared.backoff_factor if declared else 2.0,
        jitter=declared.jitter if declared else True,
    )


def _extract_conflict(parsed_result) -> tuple[bool, str | None]:
    """从结构化工具结果中检测显式冲突。"""
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


def _reconcile_with_ledger(
    batch: list[dict], ctx: RunContext,
) -> tuple[list[dict], list[tuple[dict, str]]]:
    """checkpoint 恢复后的工具重放对账（RecoveryState 是副作用账本）。

    checkpoint 只记录"执行到哪"，不保证外部副作用 exactly-once；
    从 checkpoint 恢复可能重入 execute_tools 节点，必须依据账本决定
    每个 call 的处置：

    - COMPLETED / FAILED / 已终态（UNKNOWN_OUTCOME / INTERRUPTED）：
      结果已提交 session → 跳过，不重放副作用
    - PENDING / EXECUTING + 非幂等：执行结果未知，不可自动重放
      → 合成 UNKNOWN_OUTCOME 错误结果提交，交由模型决定下一步
    - PENDING / EXECUTING + 幂等（read / idempotent_write）：允许重放

    返回 (replay_batch, synthesized)：replay_batch 为允许执行的 call 列表，
    synthesized 为 (call, 合成结果) 需要提交的项。
    """
    replay: list[dict] = []
    synthesized: list[tuple[dict, str]] = []
    recovery = ctx.recovery_state
    session = ctx.session

    for call in batch:
        tc_id = call["id"]
        record = recovery.get_record(tc_id)

        if record is None:
            # 无账本记录（理论上不会发生：batch 由 register_tool_calls_batch 注册）
            replay.append(call)
            continue

        if record.status not in (ToolCallStatus.PENDING, ToolCallStatus.EXECUTING):
            # 结果已提交（commit_tool_result_once 保证恰好一次）
            logger.info("Ledger reconcile: skip completed call %s (%s)", tc_id, record.status.value)
            continue

        # 未终态 → 崩溃点在执行中。非幂等不可重放，合成结果交模型处理；
        # 若 session.messages 里其实已有结果（终态化遗漏），commit 的幂等性兜底
        if record.idempotency == "non_idempotent" and not (
            session and any(
                m.get("role") == "tool" and m.get("tool_call_id") == tc_id
                for m in session.messages
            )
        ):
            synthetic = _make_synthetic_result(
                tc_id, record.tool_name, ToolCallStatus.UNKNOWN_OUTCOME,
                "Recovery: outcome unknown for non-idempotent call; not replayed",
            )
            synthesized.append((call, synthetic))
            recovery.mark_tool_call_terminal(
                tc_id, ToolCallStatus.UNKNOWN_OUTCOME, "reconciled: unknown outcome",
            )
            continue

        replay.append(call)

    return replay, synthesized


async def execute_tools(state: GraphState, ctx: RunContext) -> dict:
    """顺序执行当前批次的工具调用，返回状态增量。

    批次来自 call_model 节点写入 state 的 pending_tool_calls；
    执行完成后该字段清空（缺失 result 会被完整性检查终态化）。
    """
    batch = state.get("pending_tool_calls") or []
    updates: dict = {"pending_tool_calls": []}
    if not batch:
        return updates

    session = ctx.session
    assert session is not None, "GraphRunner 保证 session 存在"
    recovery: RecoveryState = ctx.recovery_state

    # 阶段 2：仅 checkpoint 恢复的重入 turn 需要对账 —— 新 turn 的
    # PENDING 记录是正常待执行状态，不得误判为崩溃现场
    if ctx.resumed:
        batch, synthesized = _reconcile_with_ledger(batch, ctx)
        for call, synthetic in synthesized:
            await commit_tool_result(ctx, call["id"], synthetic)
            if ctx.on_tool:
                ctx.on_tool("error", call["name"], call.get("arguments") or {}, synthetic, None)
            if ctx.task_mgr.state is not None:
                ctx.task_mgr.update_from_tool(call["name"], call.get("arguments") or {}, synthetic)
        if synthesized:
            await _emit_recovery(ctx)

    for call in batch:
        tc_id: str = call["id"]
        name: str = call["name"]
        arguments: dict = call.get("arguments") or {}

        # 协作式取消：终态化剩余未执行 calls 后停止
        if await check_cancel(ctx.should_cancel):
            await terminalize_on_cancel(recovery, session)
            await _emit_recovery(ctx)
            updates["run_status"] = "cancelled"
            return updates

        record = recovery.get_record(tc_id)
        if record:
            ctx.tool_context["_idempotency_key"] = record.idempotency_key
            ctx.tool_context["_action_fingerprint"] = record.action_fingerprint

        logger.info("Tool call: %s(%s)", name, arguments)
        if ctx.on_tool:
            ctx.on_tool("start", name, arguments, None, None)
        recovery.mark_executing(tc_id)
        await _emit_recovery(ctx)

        t0 = time.monotonic()
        tool_outcome = await retry_tool_call(
            lambda: ctx.tools.execute(name, arguments, tool_context=ctx.tool_context),
            policy=_tool_retry_policy(ctx, name),
            budget=ctx.budget,
            classify_result=lambda r: classify_tool_result(r, tool_name=name),
            is_ok=lambda r: parse_tool_result(r).ok,
            deadline=ctx.deadline,
            on_retry=_make_on_retry(ctx, name, arguments),
        )
        elapsed = time.monotonic() - t0
        result = tool_outcome.result

        # 终态失败 → 结构化错误信封
        if tool_outcome.outcome != Outcome.SUCCESS:
            envelope = tool_outcome.envelope or classify_tool_result(result, tool_name=name)
            result = envelope.to_tool_failure(tool_outcome.outcome, tool_outcome.attempts)

        parsed_result = parse_tool_result(result)
        is_error = not parsed_result.ok
        if name in RAG_TOOL_NAMES and parsed_result.ok:
            ctx.rag_used = True

        committed = await commit_tool_result(ctx, tc_id, result)
        if committed:
            if is_error:
                recovery.mark_failed(tc_id, parsed_result.error)
                if ctx.on_tool:
                    ctx.on_tool("error", name, arguments, result, elapsed)
            else:
                recovery.mark_completed(tc_id)
                if ctx.on_tool:
                    ctx.on_tool("end", name, arguments, result, elapsed)

        await _emit_recovery(ctx)
        ctx.task_mgr.update_from_tool(name, arguments, result)

        ctx.tool_context.pop("_idempotency_key", None)
        ctx.tool_context.pop("_action_fingerprint", None)

    # 批次结束后推送 task state
    if ctx.on_task_state and ctx.task_mgr.state:
        ctx.on_task_state(ctx.task_mgr.state.to_dict())

    # 协议完整性检查：缺失 result 的 call 合成终态结果
    incomplete = recovery.check_completeness()
    if incomplete:
        logger.error(
            "Protocol incompleteness: %d tool_call_id(s) missing result: %s",
            len(incomplete), incomplete,
        )
        for tc_id in incomplete:
            record = recovery.get_record(tc_id)
            if record and record.status in (ToolCallStatus.PENDING, ToolCallStatus.EXECUTING):
                status = (
                    ToolCallStatus.UNKNOWN_OUTCOME
                    if record.idempotency == "non_idempotent"
                    else ToolCallStatus.INTERRUPTED
                )
                synthetic = _make_synthetic_result(tc_id, record.tool_name, status, "Incompleteness detected")
                await commit_tool_result(ctx, tc_id, synthetic)
                recovery.mark_tool_call_terminal(tc_id, status, "Incompleteness detected")

    # post-turn 自动压缩
    from novare.graph.nodes.model import maybe_auto_compact
    await maybe_auto_compact(ctx)

    updates["rag_used"] = ctx.rag_used
    updates["task_state"] = (
        ctx.task_mgr.state.to_dict() if ctx.task_mgr.state else None
    )
    return updates


async def commit_tool_result(ctx: RunContext, tool_call_id: str, result: str) -> bool:
    """幂等提交 tool result（session + recovery 双记账 + on_message 回调）。"""
    from novare.recovery.state import commit_tool_result_once

    return bool(await commit_tool_result_once(
        ctx.session, ctx.recovery_state, tool_call_id, result, ctx.on_message,
    ))


def _make_on_retry(ctx: RunContext, name: str, arguments: dict):
    def _on_retry(attempt: int, max_attempts: int, delay: float, error_code: str) -> None:
        if ctx.on_tool:
            info = json.dumps({
                "attempt": attempt,
                "max_attempts": max_attempts,
                "delay": round(delay, 3),
                "error_code": error_code,
            }, ensure_ascii=False)
            ctx.on_tool("retry", name, arguments, info, None)
    return _on_retry


def route_after_tools(state: GraphState) -> str:
    """execute_tools 之后的条件路由（纯 state 函数）：
    - cancelled → finalize
    - 达到最大迭代 → finalize
    - 否则回到 call_model
    """
    if state.get("run_status") == "cancelled":
        return "finalize"
    if state.get("iteration", 0) >= state.get("iteration_limit", 0):
        return "max_iterations"
    return "call_model"
