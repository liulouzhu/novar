"""novare/graph/state.py — LangGraph State 定义

Graph State 只保存可序列化的运行状态；应用对象（session、model port、
事件发射器等）全部在 RunContext 中，不进入 checkpoint 序列化范围。

messages 保持 OpenAI dict 格式（与 session.messages / compactor /
WebSocket 事件协议一致），使用自定义 merge_messages reducer：
- 节点默认返回增量消息列表（追加）
- compact 节点通过 ReplaceMessages 哨兵全量替换压缩后的历史
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, TypedDict


@dataclass
class ReplaceMessages:
    """全量替换 state.messages 的哨兵（compact 节点专用）。"""

    messages: list[dict] = field(default_factory=list)


def merge_messages(
    old: list[dict] | ReplaceMessages | None,
    new: list[dict] | ReplaceMessages | None,
) -> list[dict]:
    """messages reducer：增量追加，或经 ReplaceMessages 全量替换。"""
    if isinstance(new, ReplaceMessages):
        return list(new.messages)
    base: list[dict] = []
    if isinstance(old, ReplaceMessages):
        base = list(old.messages)
    elif old:
        base = list(old)
    if not new:
        return base
    return [*base, *list(new)]


class GraphState(TypedDict, total=False):
    """一次 run_turn 的完整执行状态（全部可序列化）。"""

    messages: Annotated[list[dict], merge_messages]

    # ── 输入 ──
    user_input: str
    system_prompt: str

    # ── 路由常量（turn 开始时写入，条件边纯 state 化以支持 checkpoint 复用）──
    iteration_limit: int               # 最大迭代次数
    verify_enabled: bool               # 是否启用 RAG 校验子图

    # ── 执行进度 ──
    iteration: int                     # 已完成的模型调用轮数
    rag_used: bool                     # 是否调用过 rag_query（触发校验子图）
    tool_calls_present: bool           # 最近一次模型响应是否包含工具调用
    pending_tool_calls: list[dict]     # 待执行工具调用批次 [{id, name, arguments}]
    assistant_committed: bool          # 最终 assistant 消息是否已提交（verify 路径）

    # ── 终态 ──
    run_status: str                    # completed | cancelled | timeout | max_iterations | error
    final_answer: str
    error: str | None
    verification: dict | None          # 幻觉校验报告
    stop_reason: str | None            # 提前停止的说明（取消原因等）
