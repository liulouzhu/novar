"""novare/graph/nodes/model.py — call_model 节点

职责：构建请求消息 → 预检压缩 → 带重试调用 ModelPort → 提交 assistant
消息与 usage。最终回答的校验和提交放在 verify / finalize 节点。
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from novare.context_manager import TokenUsage, estimate_messages_tokens, estimate_tools_tokens
from novare.graph.context import RunContext
from novare.graph.state import GraphState
from novare.recovery.classifier import classify_exception
from novare.recovery.executor import RetryExecutor
from novare.recovery.policy import RetryPolicy

logger = logging.getLogger("novare.graph.nodes.model")

_MESSAGE_KEYS = ("role", "content", "name", "tool_calls", "tool_call_id")


def build_request_messages(state: GraphState, ctx: RunContext) -> list[dict]:
    """组装发送给模型的 messages（system + 任务状态块 + 历史消息）。

    与 legacy AgentLoop._build_messages 语义一致：
    - system prompt 拼接 TaskState prompt 块
    - 过滤消息中的私有元数据键（_compacted / _verification 等）
    """
    messages: list[dict] = []
    system_content = state.get("system_prompt") or ""
    if not system_content and ctx.session is not None:
        system_content = getattr(ctx.session, "system_prompt", "") or ""
    task_state = ctx.task_mgr.state
    if system_content:
        if task_state:
            system_content += "\n\n" + task_state.to_prompt_block()
        messages.append({"role": "system", "content": system_content})
    if ctx.session is not None:
        source = ctx.session.messages
    else:
        source = state.get("messages") or []
    messages.extend(
        {key: value for key, value in m.items() if key in _MESSAGE_KEYS}
        for m in source
    )
    return messages


def _dump_args(args: dict) -> str:
    try:
        return json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


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


def _llm_retry_policy(ctx: RunContext) -> RetryPolicy:
    opts = ctx.options
    return RetryPolicy(
        max_attempts=opts.llm_retry_attempts,
        base_delay=opts.retry_base_delay,
        max_delay=opts.retry_max_delay,
        retry_after_max_delay=opts.retry_after_max_delay,
    )


async def call_model(state: GraphState, ctx: RunContext) -> dict:
    """调用模型一次（流式 + 传输层重试），返回增量状态更新。"""
    messages = build_request_messages(state, ctx)

    # 预检压缩：估算总量超过阈值时先压缩，再重建请求消息
    if await maybe_preflight_compact(ctx):
        messages = build_request_messages(state, ctx)

    tools = ctx.tools.to_openai_tools()

    # RAG buffering：启用校验时最终回答先缓冲，不直接流式给用户
    should_buffer = bool(ctx.rag_used and ctx.verifier and ctx.verifier.enabled)

    emitted = False

    def _on_delta(delta) -> None:
        nonlocal emitted
        if not delta.text:
            return
        emitted = True
        if not should_buffer and ctx.on_text:
            ctx.on_text(delta.text)

    executor = RetryExecutor(
        _llm_retry_policy(ctx),
        ctx.budget,
        deadline=ctx.deadline,
        on_retry=lambda a, m, d, code: logger.warning(
            "LLM stream retry %d/%d after %.2fs (error_code=%s, emitted=%s)",
            a, m, d, code, emitted,
        ),
    )
    outcome = await executor.run(
        lambda: ctx.model.astream_collect(messages, tools=tools, on_delta=_on_delta),
        classify_exception,
        # 已向用户输出过字符后断流，不得透明重试
        abort_retry=lambda: emitted,
    )
    response = outcome.result

    # usage 记账
    usage = getattr(response, "usage", None)
    if usage and ctx.session is not None:
        ctx.session.usage_tracker.add(TokenUsage(
            input_tokens=usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0) or usage.get("output_tokens", 0),
        ))

    tool_calls = list(getattr(response, "tool_calls", []))
    content = getattr(response, "content", "") or ""
    has_tools = bool(tool_calls)

    updates: dict = {
        "iteration": state.get("iteration", 0) + 1,
        "tool_calls_present": has_tools,
        "run_status": "running",
    }

    if not has_tools:
        # 最终回答：内容暂存 state，校验与 session 提交在 verify/finalize 节点
        updates["final_answer"] = content
        return updates

    # 有工具调用：先去重 id（协议完整性要求每个 tool_call_id 唯一），再提交
    seen: set[str] = set()
    for tc in tool_calls:
        if tc.id in seen:
            tc.id = f"{tc.id}_dedup_{uuid.uuid4().hex[:8]}"
        seen.add(tc.id)

    # 待执行批次写入 state，由 execute_tools 节点消费
    updates["pending_tool_calls"] = [
        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
        for tc in tool_calls
    ]

    tool_calls_dicts = [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.name, "arguments": _dump_args(tc.arguments)},
        }
        for tc in tool_calls
    ]

    if ctx.session is not None:
        ctx.session.add_assistant_message(content, tool_calls=tool_calls_dicts)
        assistant_msg = ctx.session.messages[-1]
        ctx.recovery_state.assistant_message_committed = True
        await _emit_message(ctx, assistant_msg)
        ctx.recovery_state.register_tool_calls_batch(
            [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls],
            ctx.tools,
        )
        await _emit_recovery(ctx)
        updates["messages"] = [assistant_msg]
    else:
        updates["messages"] = [{
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls_dicts,
        }]

    return updates


# ── 压缩（preflight / post-turn 共用）─────────────────────────


async def maybe_preflight_compact(ctx: RunContext) -> bool:
    """调用前预检：估算 token 超过阈值（×0.8）则压缩。返回是否发生压缩。"""
    if ctx.session is None or ctx.options.auto_compact_threshold <= 0:
        return False
    preflight_threshold = int(ctx.options.auto_compact_threshold * 0.8)
    tools = ctx.tools.to_openai_tools()
    estimated = (
        estimate_messages_tokens(ctx.session.messages) + estimate_tools_tokens(tools)
    )
    if estimated < preflight_threshold and not ctx.compactor.needs_compaction(ctx.session.messages):
        return False
    return await do_compact(ctx, reason="preflight")


async def maybe_auto_compact(ctx: RunContext) -> bool:
    """工具批次结束后检查：usage 累计或消息体积触发自动压缩。"""
    if ctx.session is None or ctx.options.auto_compact_threshold <= 0:
        return False
    if (
        not ctx.session.usage_tracker.should_compact(ctx.options.auto_compact_threshold)
        and not ctx.compactor.needs_compaction(ctx.session.messages)
    ):
        return False
    return await do_compact(ctx, reason="post_turn")


async def do_compact(ctx: RunContext, reason: str) -> bool:
    """执行压缩并同步 session / usage tracker / on_compact 回调。"""
    if ctx.session is None:
        return False
    old_count = len(ctx.session.messages)
    old_tokens = estimate_messages_tokens(ctx.session.messages)
    result = await ctx.compactor.compact(ctx.session.messages)
    if not result.did_compact:
        if result.budget_overflow:
            logger.warning(
                "Context budget overflow cannot be reduced safely: tokens=%d budget=%d",
                result.estimated_tokens, ctx.compactor.token_budget,
            )
        return False

    ctx.session.messages = result.messages
    ctx.session.usage_tracker.reset_after_compact()
    if ctx.autosave:
        ctx.session.save()
    if ctx.on_compact:
        callback_result = ctx.on_compact(ctx.session)
        if asyncio.iscoroutine(callback_result):
            await callback_result
    logger.info(
        "Context compaction complete: reason=%s strategy=%s messages=%d->%d tokens=%d->%d turns=%d overflow=%s llm_calls=%d",
        reason, result.strategy, old_count, len(result.messages),
        old_tokens, result.estimated_tokens, result.selected_turns,
        result.budget_overflow, result.llm_calls,
    )
    return True
