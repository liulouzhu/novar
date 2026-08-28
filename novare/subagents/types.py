"""novare/subagents/types.py — 子智能体数据结构"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum


class SubagentType(str, Enum):
    """子智能体类型，每种类型有不同的工具白名单"""

    SEARCH = "search"        # 文献搜索：paper_search, innovation_search, rag_query
    ANALYZER = "analyzer"    # 数据分析：code_execute, rag_query, read_file, glob/grep
    WRITER = "writer"        # 报告撰写：read/write/edit_file, rag_query, glob/grep
    EXPLORER = "explorer"    # 只读探索：read_file, glob/grep, rag_query
    VERIFIER = "verifier"    # 幻觉检测：反向 RAG + 只读证据核验
    GENERAL = "general"      # 通用：除 spawn/check/reviewer 外的所有工具


class SubagentStatus(str, Enum):
    """子智能体生命周期状态"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ── 工具白名单映射 ─────────────────────────────────────────────

# 不允许任何子智能体使用的工具（防止递归生成和依赖主智能体上下文）
_EXCLUDED_TOOLS = {"spawn_subagent", "check_subagent", "list_subagents", "reviewer_evaluate"}

SUBAGENT_TOOL_ALLOWLISTS: dict[SubagentType, set[str]] = {
    SubagentType.SEARCH: {
        "paper_search", "innovation_search", "rag_query", "read_file",
    },
    SubagentType.ANALYZER: {
        "code_execute", "rag_query", "read_file", "glob_search", "grep_search",
    },
    SubagentType.WRITER: {
        "read_file", "write_file", "edit_file", "rag_query", "glob_search", "grep_search",
    },
    SubagentType.EXPLORER: {
        "read_file", "glob_search", "grep_search", "rag_query",
    },
    SubagentType.VERIFIER: {
        "rag_query", "read_file", "glob_search", "grep_search",
    },
    SubagentType.GENERAL: set(),  # 空集 = 运行时动态填充（排除 _EXCLUDED_TOOLS）
}


def get_allowlist(subagent_type: SubagentType, all_tool_names: set[str] | None = None) -> set[str]:
    """获取指定类型的工具白名单

    Args:
        subagent_type: 子智能体类型
        all_tool_names: 所有可用工具名称（GENERAL 类型需要此参数来计算差集）
    """
    if subagent_type == SubagentType.GENERAL:
        if all_tool_names is None:
            # 保守回退：返回已知的通用工具
            return {
                "read_file", "write_file", "edit_file", "glob_search", "grep_search",
                "paper_search", "paper_parse", "rag_query", "knowledge_graph",
                "code_execute", "innovation_search",
            }
        return all_tool_names - _EXCLUDED_TOOLS
    return SUBAGENT_TOOL_ALLOWLISTS[subagent_type]


# ── 数据结构 ───────────────────────────────────────────────────

@dataclass
class SubagentInput:
    """子智能体输入参数"""

    subagent_type: SubagentType
    task: str
    max_iterations: int = 16
    context: dict | None = None

    def __post_init__(self):
        if not self.task or not self.task.strip():
            raise ValueError("task 不能为空")
        if self.max_iterations < 1:
            raise ValueError("max_iterations 必须 >= 1")


@dataclass
class SubagentOutput:
    """子智能体输出结果"""

    subagent_id: str
    status: SubagentStatus
    result: str
    error: str | None = None
    tool_calls_made: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "subagent_id": self.subagent_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "tool_calls_made": self.tool_calls_made,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


@dataclass
class SubagentRecord:
    """内存中的子智能体生命周期记录"""

    subagent_id: str
    type: SubagentType
    task: str
    status: SubagentStatus = SubagentStatus.PENDING
    result: str = ""
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    tool_calls_made: int = 0
    asyncio_task: asyncio.Task | None = None

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.monotonic()
        return end - self.created_at

    def to_output(self) -> SubagentOutput:
        return SubagentOutput(
            subagent_id=self.subagent_id,
            status=self.status,
            result=self.result,
            error=self.error,
            tool_calls_made=self.tool_calls_made,
            elapsed_seconds=self.elapsed,
        )

    def to_dict(self) -> dict:
        return {
            "subagent_id": self.subagent_id,
            "type": self.type.value,
            "task": self.task[:100],
            "status": self.status.value,
            "elapsed": round(self.elapsed, 1),
            "tool_calls_made": self.tool_calls_made,
            "error": self.error,
        }
