"""novare/subagents/tool_executor.py — 工具白名单包装器

借鉴 claw-code 的 SubagentToolExecutor 模式：
包装父 ToolRegistry，按白名单过滤子智能体可用的工具。
不创建新的 ToolRegistry 实例，父注册表的变更自动同步。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from novare.tools.registry import ToolRegistry

logger = logging.getLogger("novare.subagents.executor")


class SubagentToolExecutor:
    """受工具白名单约束的执行器，包装父 ToolRegistry

    子智能体的 AgentLoop 使用此执行器替代 ToolRegistry，
    确保子智能体只能调用其类型允许的工具。
    """

    def __init__(self, parent_registry: ToolRegistry, allowed_tools: set[str]):
        self._parent = parent_registry
        self._allowed = allowed_tools
        logger.debug(
            "SubagentToolExecutor: %d tools allowed: %s",
            len(allowed_tools), sorted(allowed_tools),
        )

    @property
    def allowed_tools(self) -> set[str]:
        return self._allowed.copy()

    def to_openai_tools(self) -> list[dict]:
        """返回白名单内的工具定义（OpenAI function calling 格式）"""
        all_tools = self._parent.to_openai_tools()
        filtered = [t for t in all_tools if t["function"]["name"] in self._allowed]
        return filtered

    def retry_policy_for(self, name: str):
        """委托查询父注册表的工具重试策略（白名单外返回 None）。"""
        if name not in self._allowed:
            return None
        return self._parent.retry_policy_for(name)

    def idempotency_for(self, name: str) -> str:
        """委托查询父注册表的工具幂等性（白名单外保守返回 non_idempotent）。"""
        if name not in self._allowed:
            return "non_idempotent"
        return self._parent.idempotency_for(name)

    async def execute(
        self, name: str, arguments: dict, tool_context: dict | None = None,
    ) -> str:
        """执行工具 — 仅允许白名单内的工具

        Args:
            name: 工具名称
            arguments: 工具参数
            tool_context: 工具上下文（传递给 handler）

        Returns:
            工具执行结果字符串，或错误信息
        """
        if name not in self._allowed:
            return (
                f"Error: Tool '{name}' is not allowed for this subagent type. "
                f"Allowed tools: {sorted(self._allowed)}"
            )
        return await self._parent.execute(name, arguments, tool_context=tool_context)

    def __repr__(self) -> str:
        return f"SubagentToolExecutor(allowed={len(self._allowed)} tools)"
