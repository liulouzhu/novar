"""novare/graph/nodes/verify.py — verify_answer / finalize / bootstrap 节点

- bootstrap：turn 初始化（任务状态、用户消息、取消检查）
- verify_answer：rag_used 时的只读证据核验（对应 legacy 最终回答分支）
- finalize：终态收敛（正常回答 / 取消 / 超迭代）+ RecoveryState 终态化
"""

from __future__ import annotations

import asyncio
import logging

from novare.graph.context import RunContext, check_cancel
from novare.graph.state import GraphState
from novare.recovery.terminalize import (
    terminalize_on_cancel,
    terminalize_on_max_iterations,
)
from novare.recovery.state import RunStatus

logger = logging.getLogger("novare.graph.nodes.finalize")


async def _emit_message(ctx: RunContext, message: dict) -> None:
    if ctx.on_message is None:
        return
    result = ctx.on_message(dict(message))
    if asyncio.iscoroutine(result):
        await result


async def _emit_recovery(ctx: RunContext) -> None:
    if ctx.on_recovery_state is None:
        return
    result = ctx.on_recovery_state(ctx.recovery_state.to_dict())
    if asyncio.iscoroutine(result):
        await result


async def bootstrap(state: GraphState, ctx: RunContext) -> dict:
    """turn 初始化：TaskState、用户消息、注入 reviewer 模型、取消预检。"""
    session = ctx.session
    assert session is not None, "GraphRunner 保证 session 存在"

    user_input = state.get("user_input", "")
    ctx.task_mgr.init_turn(user_input)

    if ctx.reviewer_llm and "reviewer_llm" not in ctx.tool_context:
        ctx.tool_context["reviewer_llm"] = ctx.reviewer_llm

    updates: dict = {"iteration": 0, "run_status": "running"}

    if await check_cancel(ctx.should_cancel):
        await terminalize_on_cancel(ctx.recovery_state, session)
        await _emit_recovery(ctx)
        updates["run_status"] = "cancelled"
        return updates

    session.add_user_message(user_input)
    await _emit_message(ctx, session.messages[-1])
    updates["messages"] = [session.messages[-1]]
    return updates


async def verify_answer(state: GraphState, ctx: RunContext) -> dict:
    """对最终回答执行幻觉校验（rag_used 且 verifier 启用时到达此节点）。

    校验失败 / 超时由 verifier 内部降级返回原始回答，不中断主任务。
    """
    session = ctx.session
    assert session is not None

    answer = state.get("final_answer", "")
    verification = None
    if ctx.verifier is not None:
        result = await ctx.verifier.verify(
            answer=answer,
            user_question=state.get("user_input", ""),
            tool_context=ctx.tool_context,
        )
        answer = result.corrected_answer
        verification = result.to_dict()
        if ctx.on_verification and verification is not None:
            callback_result = ctx.on_verification(dict(verification))
            if asyncio.iscoroutine(callback_result):
                await callback_result

    session.add_assistant_message(answer)
    message = session.messages[-1]
    if verification is not None:
        message["_verification"] = verification
    await _emit_message(ctx, message)

    # 校验后的最终回答（可能已被修正）流式补发给前端
    if ctx.on_text and answer:
        ctx.on_text(answer)

    updates: dict = {
        "final_answer": answer,
        "verification": verification,
        # 告知 finalize：assistant 消息已随校验报告提交，无需重复
        "assistant_committed": True,
    }
    if verification is not None:
        updates["messages"] = [message]
    return updates


async def finalize(state: GraphState, ctx: RunContext) -> dict:
    """终态收敛：提交最终回答 / 取消提示 / 超迭代提示。"""
    session = ctx.session
    assert session is not None
    recovery = ctx.recovery_state

    status = state.get("run_status", "running")
    updates: dict = {"run_status": status}

    if status == "cancelled":
        answer = "任务已取消。"
        session.add_assistant_message(answer)
        await _emit_message(ctx, session.messages[-1])
        updates["final_answer"] = answer
        if ctx.on_text:
            ctx.on_text(answer)
        recovery.set_run_status(RunStatus.CANCELLED)
        await _emit_recovery(ctx)
        return updates

    if status == "max_iterations":
        await terminalize_on_max_iterations(recovery, session)
        answer = "达到最大迭代次数（{}），请简化问题后重试。".format(ctx.options.max_iterations)
        # legacy 语义：max_iterations 终态归为 FAILED（terminalize_on_max_iterations）
        recovery.set_run_status(RunStatus.FAILED)
        await _emit_recovery(ctx)
        updates["final_answer"] = answer
        session.add_assistant_message(answer)
        await _emit_message(ctx, session.messages[-1])
        return updates

    # 正常完成：verify 路径已提交 assistant 消息；直接路径在此提交
    answer = state.get("final_answer", "")
    if not state.get("assistant_committed"):
        session.add_assistant_message(answer)
        await _emit_message(ctx, session.messages[-1])

    recovery.set_run_status(RunStatus.COMPLETED)
    await _emit_recovery(ctx)
    updates["run_status"] = "completed"
    return updates


def route_after_model(state: GraphState, ctx: RunContext) -> str:
    """call_model 之后的条件路由。"""
    if state.get("run_status") == "cancelled":
        return "finalize"
    if state.get("tool_calls_present"):
        return "execute_tools"
    if state.get("rag_used") and ctx.verifier is not None and ctx.verifier.enabled:
        return "verify_answer"
    return "finalize"
