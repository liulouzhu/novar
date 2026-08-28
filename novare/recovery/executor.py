"""novare/recovery/executor.py — 通用异步重试执行器

两种使用模式：

1. RetryExecutor.run()      — 异常模式（LLM 流式调用等）：attempt_fn 抛异常即失败，
                              异常交给 classify 分类后决定是否重试。
2. retry_tool_call()        — 结果模式（工具调用）：attempt_fn 返回结果字符串，
                              用 is_ok/classify_result 判断成败后决定是否重试。

共同约束：
- 只自动重试 TRANSIENT（retryable=True）。
- asyncio.CancelledError 立即传播，绝不重试。
- 受单次 RetryPolicy、每轮 RetryBudget 和 turn deadline 限制。
- exponential backoff + full jitter；Retry-After 优先并受上限约束。
"""

from __future__ import annotations

import asyncio
import random as _random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from novare.recovery.policy import RetryBudget, RetryPolicy
from novare.recovery.types import ErrorEnvelope, Outcome, RetryExhaustedError

# 重试事件回调：(attempt, max_attempts, delay_seconds, error_code)
RetryCallback = Callable[[int, int, float, str], None]


@dataclass
class RetryOutcome:
    """异常模式（run）的成功结果。"""

    result: Any
    attempts: int
    outcome: Outcome


@dataclass
class ToolRetryOutcome:
    """结果模式（retry_tool_call）的终态结果。"""

    result: str
    attempts: int
    outcome: Outcome
    envelope: ErrorEnvelope | None = None


class RetryExecutor:
    """通用异步重试执行器（异常模式）。"""

    def __init__(
        self,
        policy: RetryPolicy,
        budget: RetryBudget | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random_fn: Callable[[float, float], float] = _random.uniform,
        deadline: float | None = None,  # time.monotonic() 截止时间
        on_retry: RetryCallback | None = None,
    ):
        self.policy = policy
        self.budget = budget
        self._sleep = sleep
        self._random_fn = random_fn
        self.deadline = deadline
        self.on_retry = on_retry

    def _deadline_passed(self) -> bool:
        return self.deadline is not None and time.monotonic() >= self.deadline

    async def run(
        self,
        attempt_fn: Callable[[], Awaitable[Any]],
        classify: Callable[[BaseException], ErrorEnvelope],
        *,
        abort_retry: Callable[[], bool] | None = None,
    ) -> RetryOutcome:
        """执行 attempt_fn 并处理重试。attempt_fn 抛异常表示失败。

        abort_retry: 每次失败后调用；返回 True 时停止重试并原样抛出异常
                     （用于“已向用户输出过字符后断流不得透明重试”）。
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                result = await attempt_fn()
                return RetryOutcome(result=result, attempts=attempt, outcome=Outcome.SUCCESS)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                envelope = classify(exc)
                retryable = envelope.retryable
                aborted = bool(abort_retry and abort_retry())

                if aborted or self._deadline_passed():
                    # 已输出内容 / 轮次截止 → 不再重试，原样抛出（不得透明重试）
                    raise
                if not retryable:
                    # 确定性错误（认证/参数/未知）→ 不重试
                    raise
                if not self.policy.can_retry(attempt) or (
                    self.budget is not None and not self.budget.can_retry()
                ):
                    # 策略或每轮预算耗尽 → RetryExhaustedError。
                    # 用 from None 抑制异常链：不保留原始异常（可能含 secret），
                    # 只保留 cause_type / status_code / error_code 等诊断字段。
                    raise RetryExhaustedError.from_exception(
                        exc,
                        attempts=attempt,
                        error_code=envelope.error_code,
                        status_code=envelope.status_code,
                    ) from None

                delay = self.policy.compute_delay(attempt, envelope.retry_after, self._random_fn)
                if self.deadline is not None:
                    delay = min(delay, max(0.0, self.deadline - time.monotonic()))
                if self.budget is not None:
                    self.budget.consume()
                if self.on_retry:
                    self.on_retry(attempt, self.policy.max_attempts, delay, envelope.error_code)
                await self._sleep(delay)


async def retry_tool_call(
    attempt_fn: Callable[[], Awaitable[str]],
    *,
    policy: RetryPolicy,
    budget: RetryBudget | None = None,
    classify_result: Callable[[str], ErrorEnvelope],
    is_ok: Callable[[str], bool],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    random_fn: Callable[[float, float], float] = _random.uniform,
    deadline: float | None = None,
    on_retry: RetryCallback | None = None,
) -> ToolRetryOutcome:
    """结果模式重试：attempt_fn 返回工具结果字符串（不抛异常）。

    终态失败时 result 保持最后一次的原始结果，由调用方包装为结构化 JSON。
    """
    attempt = 0
    while True:
        attempt += 1
        result = await attempt_fn()
        if is_ok(result):
            return ToolRetryOutcome(result=result, attempts=attempt, outcome=Outcome.SUCCESS)

        envelope = classify_result(result)
        retryable = envelope.retryable
        deadline_ok = deadline is None or time.monotonic() < deadline

        if not retryable or not deadline_ok:
            outcome = Outcome.NOT_APPLIED
        elif not policy.can_retry(attempt):
            # max_attempts=1 视为“策略不允许重试”（not_applied）；
            # 超过 1 的策略耗尽视为 retry_exhausted。
            outcome = Outcome.NOT_APPLIED if policy.max_attempts <= 1 else Outcome.RETRY_EXHAUSTED
        elif budget is not None and not budget.can_retry():
            outcome = Outcome.RETRY_EXHAUSTED
        else:
            delay = policy.compute_delay(attempt, envelope.retry_after, random_fn)
            if deadline is not None:
                delay = min(delay, max(0.0, deadline - time.monotonic()))
            if budget is not None:
                budget.consume()
            if on_retry:
                on_retry(attempt, policy.max_attempts, delay, envelope.error_code)
            await sleep(delay)
            continue

        return ToolRetryOutcome(result=result, attempts=attempt, outcome=outcome, envelope=envelope)
