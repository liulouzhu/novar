"""novare/recovery — 统一错误模型、重试机制与执行恢复状态

- types.py:      失败分类（FailureKind）、结果语义（Outcome）、错误信封、RetryExhaustedError
- policy.py:     单次重试策略（RetryPolicy）与每轮共享预算（RetryBudget）
- classifier.py: 异常 / HTTP 状态 / tool result 的分类与脱敏
- executor.py:   通用异步重试执行器（异常模式 + 工具结果模式）
- state.py:      执行恢复状态（RecoveryState）— 协议完整性、幂等性、对账
- terminalize.py: 工具调用终态化 — timeout/cancel/exception 场景
"""

from novare.recovery.types import (
    ErrorEnvelope,
    FailureKind,
    Outcome,
    RetryExhaustedError,
)
from novare.recovery.policy import RetryBudget, RetryPolicy
from novare.recovery.classifier import (
    classify_exception,
    classify_status_code,
    classify_tool_result,
    sanitize_error,
)
from novare.recovery.executor import RetryExecutor, ToolRetryOutcome, retry_tool_call
from novare.recovery.state import (
    CURRENT_SCHEMA_VERSION,
    RecoveryState,
    RunStatus,
    ToolCallRecord,
    ToolCallStatus,
    _compute_action_fingerprint,
    _make_synthetic_result,
)
from novare.recovery.terminalize import (
    terminalize_on_cancel,
    terminalize_on_exception,
    terminalize_on_max_iterations,
    terminalize_on_timeout,
    terminalize_pending_calls,
)

__all__ = [
    "ErrorEnvelope",
    "FailureKind",
    "Outcome",
    "RetryExhaustedError",
    "RetryBudget",
    "RetryPolicy",
    "classify_exception",
    "classify_status_code",
    "classify_tool_result",
    "sanitize_error",
    "RetryExecutor",
    "ToolRetryOutcome",
    "retry_tool_call",
    "CURRENT_SCHEMA_VERSION",
    "RecoveryState",
    "RunStatus",
    "ToolCallRecord",
    "ToolCallStatus",
    "_compute_action_fingerprint",
    "_make_synthetic_result",
    "terminalize_on_cancel",
    "terminalize_on_exception",
    "terminalize_on_max_iterations",
    "terminalize_on_timeout",
    "terminalize_pending_calls",
]
