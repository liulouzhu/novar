"""novare/graph/runner.py — GraphRunner：LangGraph 运行时入口

对外保持与 AgentLoop.run_turn 完全一致的调用签名，Web / CLI / Channels
无需感知运行时差异即可切换（NOVARE_AGENT_RUNTIME=langgraph）。

阶段 2（checkpoint 持久化）：
- 静态编译图：GraphRunner 构造时 build 一次并挂载 checkpointer；
  per-turn 依赖经 config["configurable"]["run_ctx"] 注入（不可序列化对象
  不进 checkpoint）
- thread_id = user_id:session_id:run_id（turn 级执行现场，见 checkpointer.py）
- resume=True 时从 checkpoint 继续执行（graph.ainvoke(None)），
  配合注入的 RecoveryState 账本完成工具重放对账
- 正常完成的 turn 自动清理 checkpoint；cancelled / timeout / error /
  max_iterations 保留 checkpoint 供显式恢复

与 legacy 的行为对齐点：
- timeout / cancel / exception 三路终态化（terminalize_on_*）
- 回调集合与触发时机（on_text / on_tool / on_message / on_task_state /
  on_recovery_state / on_verification / on_compact）
- 取消返回 "任务已取消。"；超时返回友好提示

MVP 范围外（graph 模式暂不支持，构造时显式拒绝）：
- Reflexion（阶段 3 迁移为子图）
"""

from __future__ import annotations

import asyncio
import logging
import random as _random
import time
from typing import Awaitable, Callable

from novare.context_compactor import HybridContextCompactor
from novare.graph.adapters.model import ModelPort
from novare.graph.builder import build_graph
from novare.graph.context import RunContext, RuntimeOptions
from novare.graph.state import GraphState
from novare.recovery.policy import RetryBudget
from novare.recovery.terminalize import (
    terminalize_on_cancel,
    terminalize_on_exception,
    terminalize_on_timeout,
)
from novare.recovery.state import RecoveryState, RunStatus
from novare.session import Session

logger = logging.getLogger("novare.graph.runner")


def _compactor_llm(model: ModelPort):
    """为 HybridContextCompactor 提供 .chat() 兼容的 LLM。

    LegacyModelPort 直接暴露底层 LLMClient；LangChainModelPort
    通过 ChatCompatLLM 适配；测试替身等无底层模型时返回 None
    （compactor 显式支持 None，仅禁用 LLM 摘要）。
    """
    client = getattr(model, "client", None)
    if client is not None:
        return client
    chat_model = getattr(model, "chat_model", None)
    if chat_model is not None:
        from novare.graph.adapters.model import ChatCompatLLM
        return ChatCompatLLM(chat_model)
    logger.warning("ModelPort %s 未暴露底层 LLM，上下文压缩禁用 LLM 摘要", type(model).__name__)
    return None


class GraphRunner:
    """LangGraph 运行时，签名兼容 AgentLoop。"""

    def __init__(
        self,
        model: ModelPort,
        tool_registry,
        system_prompt: str = "",
        max_iterations: int = 20,
        turn_timeout: int = 300,
        auto_compact_threshold: int = 100_000,
        context_max_turns: int = 3,
        context_token_budget: int = 12_000,
        context_summary_max_tokens: int = 2_500,
        context_tool_result_max_tokens: int = 1_200,
        context_llm_timeout: float = 30.0,
        context_llm_enabled: bool = True,
        context_compactor: HybridContextCompactor | None = None,
        hallucination_verifier=None,
        llm_retry_attempts: int = 3,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 8.0,
        max_retries_per_turn: int = 6,
        retry_after_max_delay: float = 30.0,
        retry_sleep: Callable[[float], Awaitable[None]] | None = None,
        retry_random: Callable[[float, float], float] | None = None,
        reviewer_llm=None,
        checkpointer=None,
    ):
        self.model = model
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer
        self.options = RuntimeOptions(
            max_iterations=max_iterations,
            turn_timeout=turn_timeout,
            auto_compact_threshold=auto_compact_threshold,
            llm_retry_attempts=llm_retry_attempts,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
            max_retries_per_turn=max_retries_per_turn,
            retry_after_max_delay=retry_after_max_delay,
        )
        self.context_compactor = context_compactor
        self._compactor_kwargs = dict(
            max_turns=context_max_turns,
            token_budget=context_token_budget,
            summary_max_tokens=context_summary_max_tokens,
            tool_result_max_tokens=context_tool_result_max_tokens,
            llm_timeout=context_llm_timeout,
            llm_enabled=context_llm_enabled,
        )
        self.hallucination_verifier = hallucination_verifier
        self.reviewer_llm = reviewer_llm
        # 兼容 legacy 注入点（测试 / 上层可能替换 sleep、random）
        self._retry_sleep = retry_sleep or asyncio.sleep
        self._retry_random = retry_random or _random.uniform
        # 静态编译图：checkpointer 挂载在编译期，RunContext 走 config 注入
        self._graph = build_graph(checkpointer)

    async def run_turn(
        self,
        session: Session,
        user_input: str,
        on_text: Callable[[str], None] | None = None,
        on_tool: Callable[[str, str, dict, str | None, float | None], None] | None = None,
        tool_context: dict | None = None,
        on_task_state: Callable[[dict], None] | None = None,
        system_prompt: str | None = None,
        autosave: bool = True,
        on_compact: Callable[[object], Awaitable[None] | None] | None = None,
        should_cancel: Callable[[], Awaitable[bool] | bool] | None = None,
        on_message: Callable[[dict], Awaitable[None] | None] | None = None,
        on_verification: Callable[[dict], Awaitable[None] | None] | None = None,
        on_recovery_state: Callable[[dict], Awaitable[None] | None] | None = None,
        *,
        thread_id: str | None = None,
        resume: bool = False,
        recovery_state: RecoveryState | None = None,
        **_ignored,
    ) -> str:
        """执行一轮对话。签名与 AgentLoop.run_turn 一致。

        阶段 2 新增（keyword-only，legacy 运行时通过 **_ignored 吸收对应参数）：
        - thread_id: checkpoint 线程标识；缺省从 tool_context 的 user_id +
          session_id + recovery run_id 推导
        - resume: True 时从 thread_id 的最后一个 checkpoint 继续执行
          （user_input 仍用于恢复 RunContext / TaskState 的目标描述）
        - recovery_state: 恢复的副作用账本（RecoveryState.from_dict 产物）；
          resume 时应传入，新 turn 传 None 自动新建
        """
        if on_verification is not None and (
            self.hallucination_verifier is None
            or not self.hallucination_verifier.enabled
        ):
            # verifier 未启用时上层不应期待校验回调被触发
            on_verification = None

        ctx = self._build_context(
            session=session,
            user_input=user_input,
            system_prompt=system_prompt,
            autosave=autosave,
            tool_context=tool_context,
            on_text=on_text,
            on_tool=on_tool,
            on_task_state=on_task_state,
            on_compact=on_compact,
            should_cancel=should_cancel,
            on_message=on_message,
            on_verification=on_verification,
            on_recovery_state=on_recovery_state,
            recovery_state=recovery_state,
            resumed=resume,
        )

        if thread_id is None:
            user_id = ctx.tool_context.get("user_id") or "anonymous"
            session_id = getattr(session, "session_id", "") or "no-session"
            thread_id = f"{user_id}:{session_id}:{ctx.recovery_state.run_id}"
        # run_id 与 thread_id 建立映射（随快照持久化到 recovery_states 表）
        if not ctx.recovery_state.thread_id:
            ctx.recovery_state.thread_id = thread_id

        config = {
            "configurable": {
                "thread_id": thread_id,
                "run_ctx": ctx,
            },
            "recursion_limit": self._recursion_limit(),
        }

        if resume:
            # checkpoint 恢复：None 输入表示从最后一个 checkpoint 继续，
            # state（messages / iteration / pending_tool_calls）由框架还原
            invoke_input = None
        else:
            invoke_input = self._initial_state(user_input, system_prompt)

        try:
            final_state = await asyncio.wait_for(
                self._graph.ainvoke(invoke_input, config=config),
                timeout=self.options.turn_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Turn timed out after %ds (user_input=%s)",
                self.options.turn_timeout, user_input[:80],
            )
            await terminalize_on_timeout(ctx.recovery_state, session)
            await self._emit_recovery_state(on_recovery_state, ctx.recovery_state)
            # 保留 checkpoint 供显式恢复
            return (
                f"本轮任务超时（超过 {self.options.turn_timeout} 秒），"
                "请简化问题或拆分为更小的子任务后重试。"
            )
        except asyncio.CancelledError:
            try:
                await asyncio.wait_for(
                    terminalize_on_cancel(ctx.recovery_state, session),
                    timeout=3.0,
                )
            except Exception:
                logger.warning("Failed to terminalize on CancelledError")
            raise
        except Exception as e:
            await terminalize_on_exception(ctx.recovery_state, session, e)
            await self._emit_recovery_state(on_recovery_state, ctx.recovery_state)
            raise

        # 正常完成的 turn 清理 checkpoint（无恢复价值）；
        # cancelled 保留 —— 前端可显式恢复
        if final_state.get("run_status") == "completed":
            await self._discard_checkpoint(thread_id)

        return final_state.get("final_answer") or ""

    def _initial_state(self, user_input: str, system_prompt: str | None) -> GraphState:
        effective_prompt = system_prompt if system_prompt is not None else self.system_prompt
        return {
            "user_input": user_input,
            "system_prompt": effective_prompt,
            "messages": [],
            "iteration": 0,
            "rag_used": False,
            "tool_calls_present": False,
            "pending_tool_calls": [],
            "run_status": "running",
            # 条件边路由常量：写入 state 使路由决策随 checkpoint 可恢复
            "iteration_limit": self.options.max_iterations,
            "verify_enabled": bool(
                self.hallucination_verifier and self.hallucination_verifier.enabled
            ),
        }

    async def _discard_checkpoint(self, thread_id: str) -> None:
        """正常完成后删除 thread 的 checkpoint，防止表无限增长。"""
        delete = getattr(self._graph.checkpointer, "adelete_thread", None) if self._graph.checkpointer else None
        if delete is None:
            return
        try:
            await delete(thread_id)
        except Exception:
            logger.debug("checkpoint discard failed for %s", thread_id, exc_info=True)

    def get_thread_state(self, thread_id: str):
        """读取 thread 的当前 checkpoint 状态（诊断 / 恢复预检用）。"""
        return self._graph.get_state({"configurable": {"thread_id": thread_id}})

    def _recursion_limit(self) -> int:
        # 每次迭代消耗 2 个 super-step（call_model + execute_tools），
        # 加上 bootstrap / finalize 等固定节点后留出安全余量
        return max(50, self.options.max_iterations * 2 + 20)

    def _build_context(
        self,
        *,
        session: Session,
        user_input: str,
        system_prompt: str | None,
        autosave: bool,
        tool_context: dict | None,
        on_text, on_tool, on_task_state, on_compact,
        should_cancel, on_message, on_verification, on_recovery_state,
        recovery_state: RecoveryState | None,
        resumed: bool = False,
    ) -> RunContext:
        compactor = self.context_compactor
        if compactor is None:
            compactor = HybridContextCompactor(
                _compactor_llm(self.model),
                **self._compactor_kwargs,
            )

        tool_context = dict(tool_context or {})
        return RunContext(
            model=self.model,
            tools=self.tool_registry,
            compactor=compactor,
            verifier=self.hallucination_verifier,
            options=self.options,
            reviewer_llm=self.reviewer_llm,
            session=session,
            tool_context=tool_context,
            autosave=autosave,
            recovery_state=recovery_state or RecoveryState(),
            budget=RetryBudget(max_retries=self.options.max_retries_per_turn),
            deadline=time.monotonic() + self.options.turn_timeout,
            on_text=on_text,
            on_tool=on_tool,
            on_task_state=on_task_state,
            on_compact=on_compact,
            should_cancel=should_cancel,
            on_message=on_message,
            on_verification=on_verification,
            on_recovery_state=on_recovery_state,
            resumed=resumed,
        )

    @staticmethod
    async def _emit_recovery_state(callback, state) -> None:
        if callback is None:
            return
        result = callback(state.to_dict())
        if asyncio.iscoroutine(result):
            await result
