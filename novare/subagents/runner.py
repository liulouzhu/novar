"""novare/subagents/runner.py — 子智能体执行逻辑

核心函数 run_subagent() 创建一个独立的 AgentLoop + Session，
使用受限的 SubagentToolExecutor 执行子任务。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Callable

from novare.agent_loop import AgentLoop
from novare.session import Session
from novare.subagents.tool_executor import SubagentToolExecutor
from novare.subagents.types import (
    SubagentType,
    get_allowlist,
    SUBAGENT_TOOL_ALLOWLISTS,
)

if TYPE_CHECKING:
    from novare.llm_client import LLMClient
    from novare.subagents.registry import SubagentRegistry
    from novare.tools.registry import ToolRegistry

logger = logging.getLogger("novare.subagents.runner")


# ── 子智能体系统提示词模板 ────────────────────────────────────

_SUBAGENT_SYSTEM_PROMPT = """\
## 子智能体模式

你正在作为 **{type}** 类型的子智能体运行。

### 你的任务
{task}

### 约束
- 只使用分配给你的工具，不要尝试调用其他工具
- 专注于完成任务，不做超出范围的操作
- 输出清晰、结构化的结果，供主智能体整合
- 如果任务无法完成，明确说明原因
- 使用中文输出结果

### 可用工具
{tools_desc}

{context_section}"""


async def run_subagent(
    subagent_id: str,
    task: str,
    subagent_type: SubagentType,
    parent_registry: ToolRegistry,
    llm_client: LLMClient,
    system_prompt: str,
    registry: SubagentRegistry,
    max_iterations: int = 16,
    context: dict | None = None,
    on_tool: Callable | None = None,
    tool_context: dict | None = None,
    turn_timeout: int = 600,
) -> str:
    """执行子智能体任务

    创建独立的 AgentLoop + Session，使用受限工具集执行任务。
    子智能体的会话是纯内存的（不持久化到 JSONL/DB）。

    Args:
        subagent_id: 子智能体 ID
        task: 任务描述
        subagent_type: 子智能体类型
        parent_registry: 父级工具注册表
        llm_client: LLM 客户端
        system_prompt: 父级系统提示词（会被追加子智能体指令）
        registry: 子智能体注册表
        max_iterations: 最大工具调用轮次
        context: 额外上下文（如论文 ID 列表）
        on_tool: 工具状态回调
        tool_context: 传递给子智能体工具的上下文（如 user_id，用于多用户隔离）
        turn_timeout: 单轮超时秒数（默认 600，即 10 分钟）

    Returns:
        子智能体的最终文本输出
    """
    start_time = time.monotonic()

    try:
        # 1. 构建工具白名单
        all_tool_names = {t["function"]["name"] for t in parent_registry.to_openai_tools()}
        allowed = get_allowlist(subagent_type, all_tool_names)

        # 2. 创建受限工具执行器
        executor = SubagentToolExecutor(parent_registry, allowed)

        # 3. 构建子智能体系统提示词
        sub_prompt = _build_subagent_prompt(task, subagent_type, allowed, parent_registry, context)
        full_system_prompt = system_prompt + "\n\n" + sub_prompt

        # 4. 创建内存 Session（不持久化）
        session = Session()

        # 5. 创建子智能体 AgentLoop
        loop = AgentLoop(
            llm_client=llm_client,
            tool_registry=executor,  # type: ignore[arg-type] — duck typing
            system_prompt=full_system_prompt,
            max_iterations=max_iterations,
            auto_compact_threshold=0,  # 子智能体禁用自动压缩
            turn_timeout=turn_timeout,
        )

        # 6. 执行
        logger.info("Running subagent %s (type=%s, max_iter=%d)", subagent_id, subagent_type.value, max_iterations)
        result = await loop.run_turn(session, task, on_tool=on_tool, tool_context=tool_context)

        # 7. 统计工具调用次数
        tool_calls = sum(1 for m in session.messages if m.get("role") == "tool")
        record = registry.get(subagent_id)
        if record:
            record.tool_calls_made = tool_calls

        elapsed = time.monotonic() - start_time
        logger.info("Subagent %s completed in %.1fs (%d tool calls)", subagent_id, elapsed, tool_calls)

        # 8. 标记完成
        registry.complete(subagent_id, result)
        return result

    except Exception as e:
        elapsed = time.monotonic() - start_time
        error_msg = f"{type(e).__name__}: {e}"
        logger.exception("Subagent %s failed after %.1fs", subagent_id, elapsed)
        registry.fail(subagent_id, error_msg)
        return f"Error: 子智能体执行失败 — {error_msg}"


def _build_subagent_prompt(
    task: str,
    subagent_type: SubagentType,
    allowed: set[str],
    parent_registry: ToolRegistry,
    context: dict | None = None,
) -> str:
    """构建子智能体的系统提示词追加部分"""

    # 获取允许工具的简要描述
    tool_descs = []
    for tool_def in parent_registry.list_tools():
        if tool_def.name in allowed:
            tool_descs.append(f"- {tool_def.name}: {tool_def.description[:80]}")
    tools_desc = "\n".join(tool_descs) if tool_descs else "（无可用工具）"

    # 构建上下文部分
    context_section = ""
    if context:
        parts = []
        for k, v in context.items():
            parts.append(f"- {k}: {v}")
        context_section = "### 额外上下文\n" + "\n".join(parts)

    return _SUBAGENT_SYSTEM_PROMPT.format(
        type=subagent_type.value,
        task=task,
        tools_desc=tools_desc,
        context_section=context_section,
    )
