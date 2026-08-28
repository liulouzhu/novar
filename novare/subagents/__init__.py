"""novare/subagents — 子智能体系统

借鉴 claw-code 的 subagent 架构，允许主智能体将研究子任务
委托给专用子智能体并行执行。子智能体拥有独立的 session 和受限的工具集。
"""

from novare.subagents.types import (
    SubagentType,
    SubagentStatus,
    SubagentInput,
    SubagentOutput,
    SubagentRecord,
)
from novare.subagents.registry import SubagentRegistry
from novare.subagents.tool_executor import SubagentToolExecutor

__all__ = [
    "SubagentType",
    "SubagentStatus",
    "SubagentInput",
    "SubagentOutput",
    "SubagentRecord",
    "SubagentRegistry",
    "SubagentToolExecutor",
]
