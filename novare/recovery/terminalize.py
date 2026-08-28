"""novare/recovery/terminalize.py — 工具调用终态化

为未完成的 tool call 生成合成结果并提交，确保协议完整性。
处理 timeout、cancel、exception 和进程恢复场景。
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable

from novare.recovery.state import (
    RecoveryState,
    RunStatus,
    ToolCallRecord,
    ToolCallStatus,
    _make_synthetic_result,
)

if TYPE_CHECKING:
    from novare.session import Session

logger = logging.getLogger("novare.recovery.terminalize")


async def terminalize_pending_calls(
    recovery_state: RecoveryState,
    session: Session,
    *,
    target_status: ToolCallStatus,
    reason: str = "",
    run_status: RunStatus | None = None,
    commit_fn: Callable[[Session, RecoveryState, str, str], Awaitable[None]] | None = None,
    timeout_seconds: float = 5.0,
) -> int:
    """终态化所有未完成的 tool call。

    Args:
        recovery_state: 当前 RecoveryState
        session: 会话对象
        target_status: 目标终态（CANCELLED/TIMED_OUT/SKIPPED/INTERRUPTED）
        reason: 终态化原因
        run_status: 可选的 run 状态更新
        commit_fn: 可选的提交函数（用于持久化合成结果）
        timeout_seconds: 提交操作的超时

    Returns:
        终态化的 tool call 数量
    """
    pending_ids = recovery_state.check_completeness()
    if not pending_ids:
        return 0

    count = 0
    for tc_id in pending_ids:
        record = recovery_state.get_record(tc_id)
        if not record:
            continue

        # 已有终态结果的不重复处理
        if record.status in (
            ToolCallStatus.COMPLETED,
            ToolCallStatus.FAILED,
            ToolCallStatus.CANCELLED,
            ToolCallStatus.TIMED_OUT,
            ToolCallStatus.SKIPPED,
            ToolCallStatus.UNKNOWN_OUTCOME,
            ToolCallStatus.INTERRUPTED,
        ):
            continue

        # 根据工具状态决定终态
        if record.status == ToolCallStatus.EXECUTING:
            # 已开始执行但未完成：根据幂等性决定
            if record.idempotency == "non_idempotent":
                # 非幂等工具已开始，不知道是否完成
                final_status = ToolCallStatus.UNKNOWN_OUTCOME
            else:
                # 只读/幂等写入可以安全重试，但这里先标记为中断
                final_status = target_status
        else:
            # PENDING 状态
            final_status = target_status

        synthetic = _make_synthetic_result(
            tc_id, record.tool_name, final_status, reason,
        )

        # 提交合成结果
        try:
            if commit_fn:
                await asyncio.wait_for(
                    commit_fn(session, recovery_state, tc_id, synthetic),
                    timeout=timeout_seconds,
                )
            else:
                # 内存模式：直接写 session
                session.add_tool_result(tc_id, synthetic)

            recovery_state.mark_tool_call_terminal(tc_id, final_status, reason)
            count += 1
            logger.info(
                "Terminalized tool call %s (%s): %s",
                tc_id, record.tool_name, final_status.value,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Timeout terminalizing tool call %s (%s)",
                tc_id, record.tool_name,
            )
        except Exception as e:
            logger.error(
                "Failed to terminalize tool call %s (%s): %s",
                tc_id, record.tool_name, e,
            )

    # 更新 run status
    if run_status:
        recovery_state.set_run_status(run_status)

    return count


async def terminalize_on_timeout(
    recovery_state: RecoveryState,
    session: Session,
    commit_fn: Callable | None = None,
) -> int:
    """timeout 时终态化所有 pending calls"""
    return await terminalize_pending_calls(
        recovery_state,
        session,
        target_status=ToolCallStatus.TIMED_OUT,
        reason="Turn timed out",
        run_status=RunStatus.TIMED_OUT,
        commit_fn=commit_fn,
        timeout_seconds=3.0,
    )


async def terminalize_on_cancel(
    recovery_state: RecoveryState,
    session: Session,
    commit_fn: Callable | None = None,
) -> int:
    """协作式取消时终态化所有 pending calls"""
    return await terminalize_pending_calls(
        recovery_state,
        session,
        target_status=ToolCallStatus.CANCELLED,
        reason="Task cancelled by user",
        run_status=RunStatus.CANCELLED,
        commit_fn=commit_fn,
        timeout_seconds=3.0,
    )


async def terminalize_on_exception(
    recovery_state: RecoveryState,
    session: Session,
    error: Exception,
    commit_fn: Callable | None = None,
) -> int:
    """异常时终态化所有 pending calls"""
    return await terminalize_pending_calls(
        recovery_state,
        session,
        target_status=ToolCallStatus.INTERRUPTED,
        reason=f"Exception: {type(error).__name__}",
        run_status=RunStatus.FAILED,
        commit_fn=commit_fn,
        timeout_seconds=3.0,
    )


async def terminalize_on_max_iterations(
    recovery_state: RecoveryState,
    session: Session,
    commit_fn: Callable | None = None,
) -> int:
    """达到最大迭代次数时终态化"""
    return await terminalize_pending_calls(
        recovery_state,
        session,
        target_status=ToolCallStatus.SKIPPED,
        reason="Max iterations reached",
        run_status=RunStatus.FAILED,
        commit_fn=commit_fn,
        timeout_seconds=3.0,
    )
