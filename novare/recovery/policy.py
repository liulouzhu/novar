"""novare/recovery/policy.py — 重试策略与每轮共享重试预算

- RetryPolicy：单次调用的重试策略（exponential backoff + full jitter，
  Retry-After 优先并受上限约束）。
- RetryBudget：每轮共享的重试预算（LLM 重试与工具重试共用）。
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RetryPolicy:
    """单次调用的重试策略。

    max_attempts: 总尝试次数（含首次）。1 = 不重试（保守默认）。
    base_delay:   首次重试的退避基数（秒）。
    max_delay:    指数退避上限（秒）。
    retry_after_max_delay: 服务端 Retry-After 的上限（秒）。
    backoff_factor: 指数退避因子。
    jitter:       True 时使用 full jitter（0, cap] 均匀随机）。
    """

    max_attempts: int = 1
    base_delay: float = 0.5
    max_delay: float = 8.0
    retry_after_max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True

    def can_retry(self, attempt: int) -> bool:
        """attempt 为已完成（失败）的尝试次数（1-based），是否还能再试。"""
        return attempt < self.max_attempts

    def compute_delay(
        self,
        attempt: int,
        retry_after: float | None = None,
        random_fn: Callable[[float, float], float] | None = None,
    ) -> float:
        """计算第 attempt 次失败后的等待时间（秒）。

        - Retry-After 优先，但受 retry_after_max_delay 上限约束。
        - 否则 exponential backoff + full jitter：U(0, min(base * factor^(n-1), max))。
        """
        if retry_after is not None and retry_after > 0:
            return min(float(retry_after), self.retry_after_max_delay)
        exp_cap = min(
            self.base_delay * (self.backoff_factor ** (attempt - 1)),
            self.max_delay,
        )
        if self.jitter:
            rng = random_fn or _random.uniform
            return rng(0.0, exp_cap)
        return exp_cap


@dataclass
class RetryBudget:
    """每轮共享的重试预算 — LLM 重试与工具重试共用同一实例。

    Retry 不消耗 Agent 的 max_iterations，但消耗此预算。
    """

    max_retries: int = 6
    remaining: int | None = field(default=None)

    def __post_init__(self) -> None:
        if self.remaining is None:
            self.remaining = self.max_retries

    def can_retry(self) -> bool:
        return self.remaining > 0

    def consume(self) -> bool:
        """消耗一次重试配额。返回是否成功。"""
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True

    @property
    def used(self) -> int:
        return self.max_retries - self.remaining
