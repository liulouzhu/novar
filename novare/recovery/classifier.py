"""novare/recovery/classifier.py — 异常 / HTTP 状态 / tool result 的失败分类

优先使用异常类型、status_code、结构化字段（error_code）判断；
只有拿不到这些信息时才降级到消息文本模式匹配，且严格保守
（无法可靠分类 → UNKNOWN → 不重试）。
所有输出经 sanitize_error 脱敏，不记录 API key / Authorization / 敏感参数。
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx

from novare.recovery.types import ErrorEnvelope, FailureKind, RetryExhaustedError

# ── HTTP 状态码分类 ──────────────────────────────────────────────
_TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}
_TERMINAL_STATUS = {401, 403}
_SEMANTIC_STATUS = {400, 404, 422}

_STATUS_ERROR_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    408: "REQUEST_TIMEOUT",
    422: "INVALID_ARGUMENT",
    425: "TOO_EARLY",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
    502: "BAD_GATEWAY",
    503: "SERVICE_UNAVAILABLE",
    504: "GATEWAY_TIMEOUT",
}

# 结构化 error_code → 失败分类（error_code 优先于文本）
_ERROR_CODE_KIND: dict[str, FailureKind] = {
    # TRANSIENT
    "TIMEOUT": FailureKind.TRANSIENT,
    "UPSTREAM_TIMEOUT": FailureKind.TRANSIENT,
    "REQUEST_TIMEOUT": FailureKind.TRANSIENT,
    "GATEWAY_TIMEOUT": FailureKind.TRANSIENT,
    "CONNECTION_ERROR": FailureKind.TRANSIENT,
    "CONNECTION_RESET": FailureKind.TRANSIENT,
    "NETWORK_ERROR": FailureKind.TRANSIENT,
    "UPSTREAM_ERROR": FailureKind.TRANSIENT,
    "RATE_LIMITED": FailureKind.TRANSIENT,
    "TOO_EARLY": FailureKind.TRANSIENT,
    "INTERNAL_ERROR": FailureKind.TRANSIENT,
    "BAD_GATEWAY": FailureKind.TRANSIENT,
    "SERVICE_UNAVAILABLE": FailureKind.TRANSIENT,
    # 重试已耗尽：底层错误本质上是瞬时/可重试，只是预算用完；
    # classify 对 RetryExhaustedError 显式置 retryable=False，不再重试
    "RETRY_EXHAUSTED": FailureKind.TRANSIENT,
    # TERMINAL
    "UNAUTHORIZED": FailureKind.TERMINAL,
    "FORBIDDEN": FailureKind.TERMINAL,
    "PERMISSION_DENIED": FailureKind.TERMINAL,
    "AUTHENTICATION_ERROR": FailureKind.TERMINAL,
    "API_KEY_INVALID": FailureKind.TERMINAL,
    # SEMANTIC
    "BAD_REQUEST": FailureKind.SEMANTIC,
    "INVALID_REQUEST": FailureKind.SEMANTIC,
    "INVALID_ARGUMENT": FailureKind.SEMANTIC,
    "UNKNOWN_TOOL": FailureKind.SEMANTIC,
    "NO_HANDLER": FailureKind.SEMANTIC,
    "NOT_ALLOWED": FailureKind.SEMANTIC,
    "NOT_FOUND": FailureKind.SEMANTIC,
}

# ── 文本降级匹配（仅当无法拿到结构化字段时使用）──────────────────
_TRANSIENT_MARKERS = (
    "timeout", "timed out", "timed_out", "connection reset", "connection refused",
    "connect error", "connection error", "connection lost", "network error",
    "upstream", "temporarily unavailable", "service unavailable", "server error",
    "internal error", "bad gateway", "gateway timeout", "too many requests",
    "500", "502", "503", "504", "429", "408",
)
_TERMINAL_MARKERS = (
    "unauthorized", "forbidden", "authentication", "permission",
    "access denied", "invalid api key", "api key invalid",
)
_SEMANTIC_MARKERS = (
    "unknown tool", "no handler", "not allowed", "invalid parameter",
    "invalid argument", "invalid regex", "invalid request", "bad request",
    "schema", "not found", "400", "404", "422",
)

# ── 敏感信息脱敏 ──────────────────────────────────────────────────
# 顺序敏感：先整体替换 header / 带前缀的字段，最后才处理裸 token。
_SENSITIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Authorization 头整体（值可含空格，如 "Bearer abcdefgh" / "Basic abcdefgh"）
    (re.compile(r"(?i)\bauthorization\s*[=:]\s*[^\s,;]+(?:\s+[^\s,;]+)*"), "authorization=***"),
    # 独立 Bearer token（没有 Authorization 前缀时）
    (re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+"), "Bearer ***"),
    # api_key= / apiKey= / api-key= 整体值
    (re.compile(r"(?i)\bapi[_-]?key\s*[=:]\s*[^\s,;]+"), "api_key=***"),
    # URL 或普通文本中的裸 token（sk-/rk-/pk- 前缀）
    (re.compile(r"(?i)\b(?:sk|rk|pk)-[a-z0-9]{2,}"), "sk-***"),
]


def sanitize_error(text: str) -> str:
    """去除错误消息中的敏感信息（API key / Authorization header 等）。

    完整处理以下形式：
    - "Authorization: Bearer abcdefgh"
    - "Authorization=Basic abcdefgh"
    - "Bearer abcdefgh"
    - "api_key=sk-..."
    - URL 或普通文本中的 "sk-..." token
    """
    out = str(text)
    for pattern, repl in _SENSITIVE_PATTERNS:
        out = pattern.sub(repl, out)
    return out


def _extract_retry_after(headers: Any) -> float | None:
    if headers is None:
        return None
    try:
        value = headers.get("retry-after")
    except AttributeError:
        value = None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _kind_for_error_code(error_code: str) -> FailureKind:
    return _ERROR_CODE_KIND.get(error_code.upper(), FailureKind.UNKNOWN)


def classify_status_code(
    status_code: int,
    retry_after: float | None = None,
    error: str | None = None,
) -> ErrorEnvelope:
    """按 HTTP 状态码分类。"""
    code = _STATUS_ERROR_CODES.get(status_code, f"HTTP_{status_code}")
    message = error or f"HTTP {status_code}"
    if status_code in _TRANSIENT_STATUS:
        return ErrorEnvelope(
            error=sanitize_error(message),
            error_code=code,
            retryable=True,
            kind=FailureKind.TRANSIENT,
            status_code=status_code,
            retry_after=retry_after,
        )
    if status_code in _TERMINAL_STATUS:
        return ErrorEnvelope(
            error=sanitize_error(message),
            error_code=code,
            retryable=False,
            kind=FailureKind.TERMINAL,
            status_code=status_code,
        )
    if status_code in _SEMANTIC_STATUS:
        return ErrorEnvelope(
            error=sanitize_error(message),
            error_code=code,
            retryable=False,
            kind=FailureKind.SEMANTIC,
            status_code=status_code,
        )
    return ErrorEnvelope(
        error=sanitize_error(message),
        error_code=code,
        retryable=False,
        kind=FailureKind.UNKNOWN,
        status_code=status_code,
    )


def classify_exception(exc: BaseException) -> ErrorEnvelope:
    """对异常进行分类。

    注意：asyncio.CancelledError 必须由调用方立即传播、绝不重试，
    此函数不会被调用到它（executor 先拦截）。
    """
    # httpx.HTTPStatusError → 按状态码（含 Retry-After）
    if isinstance(exc, httpx.HTTPStatusError):
        resp = exc.response
        retry_after = _extract_retry_after(getattr(resp, "headers", None))
        return classify_status_code(
            resp.status_code,
            retry_after=retry_after,
            error=str(exc),
        )

    # httpx 传输层错误 → TRANSIENT
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ReadError,
            httpx.RemoteProtocolError,
            httpx.PoolTimeout,
        ),
    ):
        return ErrorEnvelope(
            error=sanitize_error(str(exc)),
            error_code=_code_for_httpx_exc(exc),
            retryable=True,
            kind=FailureKind.TRANSIENT,
        )

    # asyncio / 内置超时 → TRANSIENT
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return ErrorEnvelope(
            error="Operation timed out",
            error_code="TIMEOUT",
            retryable=True,
            kind=FailureKind.TRANSIENT,
        )

    # 连接类 / OS 错误（ConnectionResetError、BrokenPipeError 等）→ TRANSIENT
    if isinstance(exc, (ConnectionError, OSError)):
        return ErrorEnvelope(
            error=sanitize_error(str(exc)),
            error_code="CONNECTION_ERROR",
            retryable=True,
            kind=FailureKind.TRANSIENT,
        )

    # RetryExhaustedError：保留其安全诊断字段（error_code/status_code/attempts/cause_type），
    # 不因缺 status_code 退化为 UNKNOWN；且重试已耗尽，即使带 status_code 也不得再重试
    if isinstance(exc, RetryExhaustedError):
        error_code = exc.error_code or "RETRY_EXHAUSTED"
        kind = _kind_for_error_code(error_code)
        return ErrorEnvelope(
            error=sanitize_error(str(exc)),
            error_code=error_code,
            retryable=False,  # 重试已耗尽，禁止再重试
            kind=kind,
            status_code=exc.status_code if isinstance(exc.status_code, int) else None,
        )

    # duck typing：带 status_code / headers 的异常（测试或自定义客户端）
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        retry_after = _extract_retry_after(getattr(exc, "headers", None))
        return classify_status_code(
            status_code,
            retry_after=retry_after,
            error=str(exc),
        )

    return ErrorEnvelope(
        error=sanitize_error(str(exc)),
        retryable=False,
        kind=FailureKind.UNKNOWN,
    )


def _code_for_httpx_exc(exc: BaseException) -> str:
    if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ConnectTimeout)):
        return "TIMEOUT"
    if isinstance(exc, httpx.RemoteProtocolError):
        return "CONNECTION_RESET"
    if isinstance(exc, httpx.ConnectError):
        return "CONNECTION_ERROR"
    return "NETWORK_ERROR"


def _code_from_text(lower: str) -> str:
    """从（小写）错误文本推断 error_code。"""
    if "upstream" in lower and ("timeout" in lower or "timed out" in lower):
        return "UPSTREAM_TIMEOUT"
    if "rate limit" in lower or "too many requests" in lower or "429" in lower:
        return "RATE_LIMITED"
    if "503" in lower or "service unavailable" in lower:
        return "SERVICE_UNAVAILABLE"
    if "502" in lower or "bad gateway" in lower:
        return "BAD_GATEWAY"
    if "504" in lower or "gateway timeout" in lower:
        return "GATEWAY_TIMEOUT"
    if "500" in lower or "internal error" in lower or "server error" in lower:
        return "INTERNAL_ERROR"
    if "408" in lower or "request timeout" in lower:
        return "REQUEST_TIMEOUT"
    if "timeout" in lower or "timed out" in lower:
        return "TIMEOUT"
    if "connection reset" in lower:
        return "CONNECTION_RESET"
    if "connection" in lower or "connect" in lower or "network" in lower:
        return "CONNECTION_ERROR"
    if "upstream" in lower:
        return "UPSTREAM_ERROR"
    return "TRANSIENT_ERROR"


def _classify_text(text: str, tool_name: str | None = None) -> ErrorEnvelope:
    """降级：从工具返回的纯文本错误中分类（保守）。"""
    lower = text.lower()
    if "unknown tool" in lower:
        return ErrorEnvelope(
            error=sanitize_error(text), error_code="UNKNOWN_TOOL",
            retryable=False, kind=FailureKind.SEMANTIC,
        )
    if "no handler" in lower:
        return ErrorEnvelope(
            error=sanitize_error(text), error_code="NO_HANDLER",
            retryable=False, kind=FailureKind.SEMANTIC,
        )
    for marker in _TERMINAL_MARKERS:
        if marker in lower:
            code = (
                "UNAUTHORIZED" if "401" in lower or "unauthorized" in lower
                else "FORBIDDEN" if "403" in lower or "forbidden" in lower
                else "PERMISSION_DENIED" if "permission" in lower or "access denied" in lower
                else "AUTHENTICATION_ERROR"
            )
            return ErrorEnvelope(
                error=sanitize_error(text), error_code=code,
                retryable=False, kind=FailureKind.TERMINAL,
            )
    for marker in _TRANSIENT_MARKERS:
        if marker in lower:
            return ErrorEnvelope(
                error=sanitize_error(text), error_code=_code_from_text(lower),
                retryable=True, kind=FailureKind.TRANSIENT,
            )
    for marker in _SEMANTIC_MARKERS:
        if marker in lower:
            code = (
                "INVALID_ARGUMENT" if "invalid" in lower or "schema" in lower
                else "NOT_ALLOWED" if "not allowed" in lower
                else "NOT_FOUND" if "not found" in lower or "404" in lower
                else "BAD_REQUEST"
            )
            return ErrorEnvelope(
                error=sanitize_error(text), error_code=code,
                retryable=False, kind=FailureKind.SEMANTIC,
            )
    return ErrorEnvelope(
        error=sanitize_error(text), retryable=False, kind=FailureKind.UNKNOWN,
    )


def _parse_bool_strict(value: Any) -> bool | None:
    """严格解析布尔字段：只有真正的 bool 才返回；字符串 "false"/"true" 不信任。"""
    if isinstance(value, bool):
        return value
    return None


# 工具显式声明的 outcome：这些值表示“重试已被生产方应用/状态未知”，
# 禁止外层再次自动重试（避免嵌套重试）。
_OUTCOME_BLOCKS_RETRY = ("unknown", "applied", "retry_exhausted")


def classify_tool_result(result: str, tool_name: str | None = None) -> ErrorEnvelope:
    """对工具返回的结果字符串分类。

    优先结构化字段，且结构化字段拥有否决权：
    1. JSON 且带 error_code → 按 error_code 判断错误类型；
       - 显式 retryable:false 无条件禁止重试（error_code 不得把它升级为 true）；
       - outcome 为 unknown/applied/retry_exhausted 时禁止外层自动重试；
       - 仅当 outcome 为 not_applied（或缺失）且分类为 TRANSIENT 时才允许重试。
    2. JSON ok=false 但无 error_code → 降级到 error 文本分类；
    3. 纯文本 → 降级到文本分类（保守）。

    布尔字段严格解析：字符串 "false" 不会被当作 False（bool("false") 是 True）。
    """
    text = result or ""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if isinstance(parsed, dict) and parsed.get("ok") is False:
        producer_retryable = _parse_bool_strict(parsed.get("retryable"))
        outcome = parsed.get("outcome")
        if outcome is not None:
            outcome = str(outcome)
        error_code = str(parsed.get("error_code") or "").strip().upper()
        error_text = str(parsed.get("error") or text)

        if error_code and error_code != "UNKNOWN":
            kind = _kind_for_error_code(error_code)
        else:
            fallback = _classify_text(error_text, tool_name)
            kind = fallback.kind
            if not error_code or error_code == "UNKNOWN":
                error_code = fallback.error_code

        # ── 结构化否决逻辑 ──
        if producer_retryable is False:
            retryable = False                     # 显式 retryable:false 无条件禁止
        elif outcome in _OUTCOME_BLOCKS_RETRY:
            retryable = False                     # 已应用 / 状态未知 → 禁止嵌套重试
        elif producer_retryable is True and kind != FailureKind.TRANSIENT:
            retryable = False                     # 显式 true 但错误类型非瞬时
        else:
            retryable = kind == FailureKind.TRANSIENT

        return ErrorEnvelope(
            error=sanitize_error(error_text),
            error_code=error_code,
            retryable=retryable,
            kind=kind,
            producer_retryable=producer_retryable,
            outcome=outcome,
        )

    return _classify_text(text, tool_name)
