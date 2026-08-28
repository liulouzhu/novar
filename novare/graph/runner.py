"""novare/graph/runner.py — GraphRunner：LangGraph 运行时入口

对外保持与 AgentLoop.run_turn 完全一致的调用签名，Web / CLI / Channels
无需感知运行时差异即可切换（NOVARE_AGENT_RUNTIME=langgraph）。

与 legacy 的行为对齐点：
- timeout / cancel / exception 三路终态化（terminalize_on_*）
- 回调集合与触发时机（on_text / on_tool / on_message / on_task_state /
  on_recovery_state / on_verification / on_compact）
- 取消返回 "任务已取消。"；超时返回友好提示
- tool_context 作为 user_id / workspace 的只读透传通道保留

MVP 范围外（graph 模式暂不支持，构造时显式拒绝）：
- Reflexion（计划阶段 3 迁移为子图）
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
from novare.recovery.policy import RetryBudget
from novare.recovery.terminalize import (
    terminalize_on_cancel,
    terminalize_on_exception,
    terminalize_on_timeout,
)
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
    ):
        self.model = model
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
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
        **_ignored,
    ) -> str:
        """执行一轮对话。签名与 AgentLoop.run_turn 一致。

        **_ignored 吸收 legacy 专有参数（on_reflexion_event 等），
        保证 agent_service 可以无差别地调用两种运行时。
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
        )
        graph = build_graph(ctx)

        initial_state = {
            "user_input": user_input,
            "system_prompt": system_prompt if system_prompt is not None else self.system_prompt,
            "messages": [],
            "iteration": 0,
            "rag_used": False,
            "tool_calls_present": False,
            "pending_tool_calls": [],
            "run_status": "running",
        }

        try:
            final_state = await asyncio.wait_for(
                graph.ainvoke(initial_state, config={"recursion_limit": self._recursion_limit()}),
                timeout=self.options.turn_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Turn timed out after %ds (user_input=%s)",
                self.options.turn_timeout, user_input[:80],
            )
            await terminalize_on_timeout(ctx.recovery_state, session)
            await self._emit_recovery_state(on_recovery_state, ctx.recovery_state)
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

        return final_state.get("final_answer") or ""

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
    ) -> RunContext:
        effective_prompt = system_prompt if system_prompt is not None else self.system_prompt
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
        )

    @staticmethod
    async def _emit_recovery_state(callback, state) -> None:
        if callback is None:
            return
        result = callback(state.to_dict())
        if asyncio.iscoroutine(result):
            await result
