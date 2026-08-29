"""novare/graph/context.py — RunContext：运行期依赖容器

替代旧 AgentLoop 的两条隐藏通道：
1. 30+ 个构造参数 / 10 个 run_turn 回调参数 → 集中为 RunContext 字段
2. tool_context 双向魔法 dict（_idempotency_key / _action_fingerprint / user_id /
   workspace / reviewer_llm）→ 显式字段

RunContext 不进入 Graph State，不参与 checkpoint 序列化。
每次 run_turn 创建一个实例（turn-scoped），与 legacy 的
RecoveryState / RetryBudget 生命周期对齐。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from novare.context_compactor import HybridContextCompactor
from novare.graph.adapters.model import ModelPort
from novare.hallucination_verifier import HallucinationVerifier
from novare.recovery.policy import RetryBudget
from novare.recovery.state import RecoveryState
from novare.session import Session
from novare.task_state import TaskStateManager

# 事件回调：接收 OpenAI dict 消息（与 on_message 回调一致）
MessageCallback = Callable[[dict], Awaitable[None] | None]
# 取消检查：返回 bool 或 Awaitable[bool]
CancelChecker = Callable[[], Awaitable[bool] | bool]


async def check_cancel(should_cancel: CancelChecker | None) -> bool:
    """统一的协作式取消检查（收敛 legacy 中重复 4 处的 iscoroutine 样板）。"""
    if should_cancel is None:
        return False
    result = should_cancel()
    if asyncio.iscoroutine(result):
        result = await result
    return bool(result)


@dataclass
class RuntimeOptions:
    """图运行时选项（源自 NovareConfig，构造期不变）。"""

    max_iterations: int = 20
    turn_timeout: int = 300
    auto_compact_threshold: int = 100_000
    llm_retry_attempts: int = 3
    retry_base_delay: float = 0.5
    retry_max_delay: float = 8.0
    max_retries_per_turn: int = 6
    retry_after_max_delay: float = 30.0


@dataclass
class RunContext:
    """turn-scoped 运行上下文：静态依赖 + per-turn 可变状态。"""

    # ── 静态依赖 ──
    model: ModelPort                                   # 模型端口（ModelPort 协议）
    tools: object                                      # ToolRegistry / SubagentToolExecutor
    compactor: HybridContextCompactor
    verifier: HallucinationVerifier | None = None
    options: RuntimeOptions = field(default_factory=RuntimeOptions)
    reviewer_llm: object | None = None                 # 评审模型（注入 tool_context）

    # ── 会话与权限 ──
    session: Session | None = None
    tool_context: dict = field(default_factory=dict)   # user_id / workspace（只读透传给工具）
    autosave: bool = True

    # ── turn-scoped 状态 ──
    recovery_state: RecoveryState = field(default_factory=RecoveryState)
    budget: RetryBudget | None = None
    deadline: float = field(default_factory=lambda: time.monotonic() + 300.0)
    task_mgr: TaskStateManager = field(default_factory=TaskStateManager)
    rag_used: bool = False

    # ── 事件回调（GraphRunner 桥接旧 AgentLoop 回调签名）──
    on_text: Callable[[str], None] | None = None
    on_tool: Callable[[str, str, dict, str | None, float | None], None] | None = None
    on_message: MessageCallback | None = None
    on_task_state: Callable[[dict], None] | None = None
    on_compact: Callable[[object], Awaitable[None] | None] | None = None
    on_verification: MessageCallback | None = None
    on_recovery_state: MessageCallback | None = None
    should_cancel: CancelChecker | None = None

    # ── 运行期内部状态 ──
    streamed_final: bool = False                       # 最终回答是否已流式输出
    resumed: bool = False                              # 从 checkpoint 恢复的重入 turn

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def deadline_passed(self) -> bool:
        return time.monotonic() >= self.deadline
