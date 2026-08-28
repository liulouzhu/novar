"""novare/subagents/tools.py — spawn_subagent / check_subagent 工具处理器

这些处理器通过 tool_context 接收依赖项（registry, llm_client 等），
与 novare/tools/registry.py 中注册的工具定义配合使用。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Callable

from novare.subagents.registry import SubagentRegistry
from novare.subagents.runner import run_subagent
from novare.subagents.types import SubagentStatus, SubagentType

if TYPE_CHECKING:
    from novare.llm_client import LLMClient
    from novare.tools.registry import ToolRegistry

logger = logging.getLogger("novare.subagents.tools")


# ── 工具处理器 ─────────────────────────────────────────────────

async def handle_spawn_subagent(args: dict, **kwargs) -> str:
    """spawn_subagent 工具处理器

    创建并启动一个子智能体。默认异步执行（fire-and-forget），
    设置 await_result=true 时同步等待结果。

    kwargs 中需要以下 tool_context 项：
    - subagent_registry: SubagentRegistry
    - parent_tool_registry: ToolRegistry
    - llm_client: LLMClient
    - system_prompt: str
    - workspace: Path
    """
    # 从 tool_context 获取依赖（放在参数解析之前，因为 max_iterations 默认值依赖它）
    registry: SubagentRegistry = kwargs.get("subagent_registry")
    parent_registry: ToolRegistry = kwargs.get("parent_tool_registry")
    llm_client: LLMClient = kwargs.get("llm_client")
    system_prompt: str = kwargs.get("system_prompt", "")
    user_id: str | None = kwargs.get("user_id")
    workspace = kwargs.get("workspace")
    default_max_iterations: int = kwargs.get("default_max_iterations", 16)
    turn_timeout: int = kwargs.get("turn_timeout", 600)

    # 解析参数
    type_str = args.get("subagent_type", "general")
    task = args.get("task", "")
    max_iterations = args.get("max_iterations", default_max_iterations)
    context = args.get("context")
    await_result = args.get("await_result", False)

    if not task.strip():
        return json.dumps({"error": "task 不能为空"}, ensure_ascii=False)

    # 验证类型
    try:
        subagent_type = SubagentType(type_str)
    except ValueError:
        valid = [t.value for t in SubagentType]
        return json.dumps({"error": f"无效的子智能体类型: {type_str}，可选: {valid}"}, ensure_ascii=False)

    if not all([registry, parent_registry, llm_client]):
        return json.dumps({"error": "子智能体系统未正确初始化"}, ensure_ascii=False)

    # 创建子智能体记录
    record = registry.create(subagent_type, task)

    # 构建子智能体 tool_context（继承父 agent 的 user_id 和 workspace）
    child_tool_context = None
    if user_id:
        child_tool_context = {"user_id": user_id}
        if workspace:
            child_tool_context["workspace"] = str(workspace)

    # 构建协程
    coro = run_subagent(
        subagent_id=record.subagent_id,
        task=task,
        subagent_type=subagent_type,
        parent_registry=parent_registry,
        llm_client=llm_client,
        system_prompt=system_prompt,
        registry=registry,
        max_iterations=max_iterations,
        context=context,
        tool_context=child_tool_context,
        turn_timeout=turn_timeout,
    )

    if await_result:
        # 同步模式：等待子智能体完成
        logger.info("Spawn subagent %s (await_result=true)", record.subagent_id)
        await registry.start(record.subagent_id, coro)
        try:
            result = await asyncio.wait_for(record.asyncio_task, timeout=300)
            output = registry.get_output(record.subagent_id)
            return json.dumps(output.to_dict() if output else {"error": "结果获取失败"}, ensure_ascii=False)
        except asyncio.TimeoutError:
            registry.fail(record.subagent_id, "执行超时（300s）")
            return json.dumps({"error": "子智能体执行超时", "subagent_id": record.subagent_id}, ensure_ascii=False)
    else:
        # 异步模式：启动后立即返回
        logger.info("Spawn subagent %s (fire-and-forget)", record.subagent_id)
        await registry.start(record.subagent_id, coro)
        return json.dumps({
            "subagent_id": record.subagent_id,
            "status": "running",
            "type": subagent_type.value,
            "message": f"子智能体已启动，使用 check_subagent(subagent_id='{record.subagent_id}') 查询结果",
        }, ensure_ascii=False)


async def handle_check_subagent(args: dict, **kwargs) -> str:
    """check_subagent 工具处理器

    查询子智能体的状态和结果。
    """
    subagent_id = args.get("subagent_id", "")
    registry: SubagentRegistry = kwargs.get("subagent_registry")

    if not registry:
        return json.dumps({"error": "子智能体系统未初始化"}, ensure_ascii=False)

    output = registry.get_output(subagent_id)
    if not output:
        return json.dumps({"error": f"未找到子智能体: {subagent_id}"}, ensure_ascii=False)

    return json.dumps(output.to_dict(), ensure_ascii=False)


async def handle_list_subagents(args: dict, **kwargs) -> str:
    """list_subagents 工具处理器

    列出所有活跃的子智能体。
    """
    registry: SubagentRegistry = kwargs.get("subagent_registry")

    if not registry:
        return json.dumps({"error": "子智能体系统未初始化"}, ensure_ascii=False)

    records = registry.list_all()
    if not records:
        return json.dumps({"subagents": [], "message": "当前没有子智能体"}, ensure_ascii=False)

    return json.dumps({
        "subagents": [r.to_dict() for r in records],
    }, ensure_ascii=False)


# ── 注册函数 ───────────────────────────────────────────────────

def register_subagent_tools(
    tool_registry: ToolRegistry,
    subagent_registry: SubagentRegistry,
    llm_client: LLMClient,
    system_prompt: str,
    workspace,
    default_max_iterations: int = 16,
    turn_timeout: int = 600,
) -> None:
    """在工具注册表中注册子智能体相关的工具

    将 spawn_subagent、check_subagent、list_subagents 注册为 builtin:context 工具，
    并通过闭包注入 tool_context 依赖。

    Args:
        tool_registry: 工具注册表
        subagent_registry: 子智能体注册表
        llm_client: LLM 客户端
        system_prompt: 系统提示词
        workspace: 工作空间路径
    """
    from novare.tools.registry import ToolDef

    # 共享的 tool_context —— 每次工具调用时传入 handler 的 kwargs
    shared_context = {
        "subagent_registry": subagent_registry,
        "parent_tool_registry": tool_registry,
        "llm_client": llm_client,
        "system_prompt": system_prompt,
        "workspace": workspace,
        "default_max_iterations": default_max_iterations,
        "turn_timeout": turn_timeout,
    }

    # ── spawn_subagent ──
    tool_registry.register_tool(ToolDef(
        name="spawn_subagent",
        description=(
            "创建一个子智能体来执行研究子任务。"
            "子智能体类型：search（文献搜索）、analyzer（数据分析）、"
            "writer（报告撰写）、explorer（只读探索）、"
            "verifier（只读证据核验）、general（通用）。"
            "默认异步执行，返回子智能体 ID；设置 await_result=true 可等待结果。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "subagent_type": {
                    "type": "string",
                    "enum": [t.value for t in SubagentType],
                    "description": "子智能体类型",
                },
                "task": {
                    "type": "string",
                    "description": "子智能体的任务描述（越详细越好）",
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "最大工具调用轮次（默认 16）",
                },
                "context": {
                    "type": "object",
                    "description": "可选的额外上下文信息（如 paper_ids 列表、搜索范围等）",
                },
                "await_result": {
                    "type": "boolean",
                    "description": "是否等待子智能体完成后再返回（默认 false，异步执行）",
                },
            },
            "required": ["subagent_type", "task"],
        },
        handler=handle_spawn_subagent,
        source="builtin:context",
    ))

    # ── check_subagent ──
    tool_registry.register_tool(ToolDef(
        name="check_subagent",
        description=(
            "查询子智能体的状态和结果。"
            "如果子智能体已完成，返回完整结果；如果仍在运行，返回进度信息。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "subagent_id": {
                    "type": "string",
                    "description": "子智能体 ID（由 spawn_subagent 返回）",
                },
            },
            "required": ["subagent_id"],
        },
        handler=handle_check_subagent,
        source="builtin:context",
    ))

    # ── list_subagents ──
    tool_registry.register_tool(ToolDef(
        name="list_subagents",
        description="列出所有子智能体及其状态。",
        parameters={"type": "object", "properties": {}},
        handler=handle_list_subagents,
        source="builtin:context",
    ))

    # 将共享上下文存储到 tool_registry，供 execute() 自动注入到 builtin:context 工具
    tool_registry.set_default_tool_context(shared_context)

    logger.info("Subagent tools registered (spawn, check, list)")
