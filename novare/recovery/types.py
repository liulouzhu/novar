"""novare/recovery/types.py — 统一错误模型

定义失败分类（FailureKind）、结果语义（Outcome）、错误信封（ErrorEnvelope）
和重试耗尽错误（RetryExhaustedError）。

工具终态失败的可序列化格式：

    {
      "ok": false,
      "error": "Error executing paper_search: upstream timeout",
      "error_code": "UPSTREAM_TIMEOUT",
      "retryable": true,
      "outcome": "retry_exhausted",
      "attempts": 3
    }

error 保持字符串，兼容现有 parse_tool_result() 消费方；
error_code / retryable / outcome / attempts 为结构化字段，
不依赖错误消息字符串作为主要判断方式。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum


class FailureKind(str, Enum):
    """失败分类。只对 TRANSIENT 自动重试。"""

    TRANSIENT = "transient"      # 连接失败/连接重置/超时/408/429/500/502/503/504
    TERMINAL = "terminal"        # 401/403/明确的认证或权限错误 — 重试无意义
    SEMANTIC = "semantic"        # 400/参数或 schema 错误/未知工具/工具无 handler
    UNKNOWN = "unknown"          # 无法可靠分类 — 保守不重试


class Outcome(str, Enum):
    """一次调用的终态结果语义。"""

    SUCCESS = "success"                       # 成功
    NOT_APPLIED = "not_applied"               # 重试机制未应用（策略不允许或错误类型不重试）
    RETRY_EXHAUSTED = "retry_exhausted"       # 重试机制已应用但耗尽


@dataclass
class ErrorEnvelope:
    """统一错误信封 — 由异常 / HTTP 状态 / tool result 分类得到。

    字段均为脱敏后的安全信息，不包含 API key、Authorization header
    或完整敏感参数。

    retryable: 分类器最终决定的可重试性（已纳入 producer 显式否决）。
    producer_retryable / outcome: 工具（生产者）显式返回的结构化字段，
      仅当工具结果 JSON 明确提供时非 None；缺失（旧格式）时为 None，
      走保守的文本降级分类。
    """

    error: str                       # 脱敏后的错误消息
    error_code: str = "UNKNOWN"
    retryable: bool = False
    kind: FailureKind = FailureKind.UNKNOWN
    status_code: int | None = None
    retry_after: float | None = None
    # 工具显式返回的结构化字段（缺失时 None）
    producer_retryable: bool | None = None
    outcome: str | None = None

    def to_tool_failure(self, outcome: Outcome | str, attempts: int) -> str:
        """将终态错误序列化为工具结果字符串（兼容 parse_tool_result）。"""
        if isinstance(outcome, Outcome):
            outcome = outcome.value
        return json.dumps(
            {
                "ok": False,
                "error": self.error,
                "error_code": self.error_code,
                "retryable": self.retryable,
                "outcome": str(outcome),
                "attempts": attempts,
            },
            ensure_ascii=False,
        )


class RetryExhaustedError(Exception):
    """重试耗尽错误。

    安全约束（安全优先于保留原始异常对象）：
    - 不通过 __cause__ / __context__ / args / repr / 公开属性保留原始异常
      对象或未脱敏文本，避免 logger.exception() 把 secret 写进 traceback。
    - 仅保留非敏感诊断字段：attempts / error_code / cause_type / status_code。
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        error_code: str = "UNKNOWN",
        cause_type: str | None = None,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.error_code = error_code
        self.cause_type = cause_type
        self.status_code = status_code

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        attempts: int,
        error_code: str = "UNKNOWN",
        status_code: int | None = None,
    ) -> "RetryExhaustedError":
        # 消息脱敏：不把原始异常文本（可能含请求 URL / 敏感参数）透出到上层；
        # 不设置 __cause__，由调用方以 `raise ... from None` 抑制异常链。
        from novare.recovery.classifier import sanitize_error

        if not isinstance(status_code, int):
            # 兼容 httpx.HTTPStatusError（status_code 在 .response 上）
            status_code = getattr(exc, "status_code", None)
            if not isinstance(status_code, int):
                resp = getattr(exc, "response", None)
                if resp is not None:
                    status_code = getattr(resp, "status_code", None)
            if not isinstance(status_code, int):
                status_code = None
        return cls(
            f"Retry exhausted after {attempts} attempt(s): {sanitize_error(str(exc))}",
            attempts=attempts,
            error_code=error_code,
            cause_type=type(exc).__name__,
            status_code=status_code,
        )
