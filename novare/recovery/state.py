"""novare/recovery/state.py — 执行恢复状态（RecoveryState）

为 AgentLoop 的每次 run_turn 维护独立、可序列化的恢复状态，
保证每个已提交的 tool_call_id 最终恰好有一个 tool result。

设计要点：
- RecoveryState 与 TaskState 生命周期和职责分离
  - TaskState: 可观测提示（goal/completed/pending），注入 system prompt
  - RecoveryState: 协议完整性保障（tool_call 追踪、幂等性、对账）
- schema_version 为未来扩展保留，但不加入 Reflexion 字段
- action fingerprint 使用 SHA-256，不包含原始敏感参数
- run_status 追踪整个 run 的终态
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from novare.session import Session


# ── 当前 schema 版本 ─────────────────────────────────────────────
CURRENT_SCHEMA_VERSION = 2  # PR 2: 增加 run_status、完整 tool call 跟踪


async def commit_tool_result_once(
    session: "Session",
    recovery_state: "RecoveryState",
    tool_call_id: str,
    result: str,
    on_message: Callable[[dict], Awaitable[None] | None] | None = None,
) -> bool:
    """统一的 tool result 提交入口。

    保证：
    - 同时检查 RecoveryState 和 session.messages
    - 重复调用是幂等 no-op
    - 成功提交后再更新 committed 状态
    - callback 失败时不得错误标记为 committed

    Returns:
        True 如果成功提交，False 如果是重复提交
    """
    # 检查是否已 committed
    if recovery_state.has_committed(tool_call_id):
        return False

    # 检查 session.messages 中是否已有该 tool_call_id 的 tool result
    for msg in session.messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id") == tool_call_id:
            # 已存在，标记为 committed 但不重复写入
            recovery_state.committed_tool_result_ids.add(tool_call_id)
            return False

    # 提交 tool result 消息
    session.add_tool_result(tool_call_id, result)

    # 回调通知
    if on_message:
        msg = session.messages[-1]
        emit_result = on_message(dict(msg))
        if asyncio.iscoroutine(emit_result):
            await emit_result

    # 标记为 committed（在消息写入成功后）
    recovery_state.committed_tool_result_ids.add(tool_call_id)

    return True


class RunStatus(str, Enum):
    """整个 run 的终态状态"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"
    RECOVERED = "recovered"


class ToolCallStatus(str, Enum):
    """单个 tool call 的状态"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"
    UNKNOWN_OUTCOME = "unknown_outcome"
    INTERRUPTED = "interrupted"


def _compute_action_fingerprint(tool_name: str, arguments: dict) -> str:
    """计算工具调用的动作指纹（SHA-256）。

    fingerprint = SHA-256(tool_name + canonical_json(arguments))
    canonical_json: key 排序、无空格、ensure_ascii=False。

    安全约束：
    - 不包含原始敏感参数值（如 API key、password）
    - 参数中的敏感字段会被脱敏后再计算指纹
    """
    sanitized = _sanitize_arguments(arguments)
    canonical = json.dumps(
        {"tool": tool_name, "args": sanitized},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "api-key", "password", "secret", "token",
    "authorization", "auth", "bearer", "credential", "private_key",
})


def _sanitize_arguments(arguments: dict) -> dict:
    """脱敏参数中的敏感字段，用于 fingerprint 计算。"""
    sanitized = {}
    for key, value in arguments.items():
        lower_key = key.lower().replace("-", "_").replace(" ", "_")
        if any(s in lower_key for s in _SENSITIVE_KEYS):
            sanitized[key] = "***"
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_arguments(value)
        elif isinstance(value, list):
            sanitized[key] = [
                _sanitize_arguments(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    return sanitized


def _generate_idempotency_key(run_id: str, tool_call_id: str) -> str:
    """生成稳定的幂等性 key：run_id + tool_call_id 的组合。"""
    return f"{run_id}:{tool_call_id}"


def _make_synthetic_result(
    tool_call_id: str,
    tool_name: str,
    status: ToolCallStatus,
    reason: str = "",
) -> str:
    """生成合成的工具结果消息，用于终态化未完成的 tool call。

    所有合成结果都包含 ok=false 和明确的 error_code，
    确保 LLM 能理解该工具调用已终态化。
    """
    error_code_map = {
        ToolCallStatus.CANCELLED: "CANCELLED",
        ToolCallStatus.TIMED_OUT: "TIMED_OUT",
        ToolCallStatus.SKIPPED: "SKIPPED",
        ToolCallStatus.UNKNOWN_OUTCOME: "UNKNOWN_OUTCOME",
        ToolCallStatus.INTERRUPTED: "INTERRUPTED",
        ToolCallStatus.FAILED: "EXECUTION_ERROR",
    }
    error_code = error_code_map.get(status, "UNKNOWN")
    error_msg = f"Tool {tool_name} ({tool_call_id}): {status.value}"
    if reason:
        error_msg += f" — {reason}"

    return json.dumps({
        "ok": False,
        "error": error_msg,
        "error_code": error_code,
        "retryable": False,
        "outcome": "not_applied",
        "attempts": 0,
        "_synthetic": True,
        "_status": status.value,
    }, ensure_ascii=False)


@dataclass
class ToolCallRecord:
    """一个工具调用的完整记录（不丢弃元数据）"""

    tool_call_id: str
    tool_name: str
    action_fingerprint: str
    idempotency_key: str
    idempotency: str  # "read" | "idempotent_write" | "non_idempotent"
    status: ToolCallStatus = ToolCallStatus.PENDING
    attempts: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    last_error: str | None = None
    result_event_key: str | None = None  # 关联的 TOOL_COMPLETED 事件 key
    arguments: dict = field(default_factory=dict)  # 保留参数用于恢复

    def to_dict(self) -> dict:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "action_fingerprint": self.action_fingerprint,
            "idempotency_key": self.idempotency_key,
            "idempotency": self.idempotency,
            "status": self.status.value,
            "attempts": self.attempts,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "last_error": self.last_error,
            "result_event_key": self.result_event_key,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ToolCallRecord:
        status_str = data.get("status", "pending")
        try:
            status = ToolCallStatus(status_str)
        except ValueError:
            status = ToolCallStatus.PENDING
        return cls(
            tool_call_id=data["tool_call_id"],
            tool_name=data["tool_name"],
            action_fingerprint=data["action_fingerprint"],
            idempotency_key=data["idempotency_key"],
            idempotency=data["idempotency"],
            status=status,
            attempts=data.get("attempts", 0),
            created_at=data.get("created_at", ""),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            last_error=data.get("last_error"),
            result_event_key=data.get("result_event_key"),
            arguments=data.get("arguments", {}),
        )


@dataclass
class RecoveryState:
    """一次 run_turn 的执行恢复状态。

    职责：
    - 追踪每个 tool_call_id 的执行状态
    - 保证协议完整性（每个 tool_call_id 恰好有一个 tool result）
    - 提供幂等性 key 和动作指纹
    - 支持序列化/反序列化（用于持久化和恢复）

    与 TaskState 分离：
    - TaskState: goal/completed/pending/key_findings → 注入 system prompt
    - RecoveryState: tool_call 追踪、幂等性、对账 → 内部协议保障
    """

    schema_version: int = CURRENT_SCHEMA_VERSION
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    run_status: RunStatus = RunStatus.RUNNING
    iteration: int = 0
    retry_count: int = 0
    # 所有 tool call 的完整记录（包括已完成的，不丢弃）
    tool_calls: dict[str, ToolCallRecord] = field(default_factory=dict)
    # 已提交 tool result 的 tool_call_id 集合
    committed_tool_result_ids: set[str] = field(default_factory=set)
    # assistant message 已持久化的标记
    assistant_message_committed: bool = False
    # batch 中所有 tool_call_id（用于完整性检查）
    batch_tool_call_ids: list[str] = field(default_factory=list)

    def register_tool_calls_batch(
        self,
        tool_calls: list[dict],  # [{"id": ..., "name": ..., "arguments": ...}]
        tool_registry=None,
    ) -> list[ToolCallRecord]:
        """一次性注册整个 batch 的工具调用。

        Args:
            tool_calls: LLM 返回的 tool_calls 列表
            tool_registry: 可选，用于查询 idempotency

        Returns:
            注册的 ToolCallRecord 列表
        """
        records = []
        for tc in tool_calls:
            tc_id = tc["id"]
            tc_name = tc["name"]
            tc_args = tc.get("arguments", {})

            # 查询工具幂等性
            idempotency = "non_idempotent"
            if tool_registry:
                idem_getter = getattr(tool_registry, "idempotency_for", None)
                if callable(idem_getter):
                    try:
                        candidate = idem_getter(tc_name)
                        if candidate in ("read", "idempotent_write", "non_idempotent"):
                            idempotency = candidate
                    except Exception:
                        idempotency = "non_idempotent"

            record = self.register_tool_call(tc_id, tc_name, tc_args, idempotency)
            records.append(record)

        # 记录 batch 中的所有 ID
        self.batch_tool_call_ids = [tc["id"] for tc in tool_calls]
        return records

    def register_tool_call(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: dict,
        idempotency: str = "non_idempotent",
    ) -> ToolCallRecord:
        """注册一个新的工具调用。"""
        fingerprint = _compute_action_fingerprint(tool_name, arguments)
        idem_key = _generate_idempotency_key(self.run_id, tool_call_id)

        record = ToolCallRecord(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            action_fingerprint=fingerprint,
            idempotency_key=idem_key,
            idempotency=idempotency,
            status=ToolCallStatus.PENDING,
            arguments=arguments,
        )
        self.tool_calls[tool_call_id] = record
        return record

    def mark_executing(self, tool_call_id: str) -> None:
        """标记工具调用开始执行"""
        record = self.tool_calls.get(tool_call_id)
        if record and record.status == ToolCallStatus.PENDING:
            record.status = ToolCallStatus.EXECUTING
            record.started_at = datetime.now(timezone.utc).isoformat()
            record.attempts += 1

    def mark_completed(self, tool_call_id: str, result_event_key: str | None = None) -> None:
        """标记工具调用完成"""
        record = self.tool_calls.get(tool_call_id)
        if record:
            record.status = ToolCallStatus.COMPLETED
            record.completed_at = datetime.now(timezone.utc).isoformat()
            record.result_event_key = result_event_key
            self.committed_tool_result_ids.add(tool_call_id)

    def mark_failed(self, tool_call_id: str, error: str | None = None) -> None:
        """标记工具调用失败（终态）"""
        record = self.tool_calls.get(tool_call_id)
        if record:
            record.status = ToolCallStatus.FAILED
            record.completed_at = datetime.now(timezone.utc).isoformat()
            record.last_error = error
            self.committed_tool_result_ids.add(tool_call_id)

    def mark_tool_call_terminal(
        self,
        tool_call_id: str,
        status: ToolCallStatus,
        error: str | None = None,
    ) -> None:
        """标记工具调用为终态（cancel/timeout/skipped/unknown/interrupted）"""
        record = self.tool_calls.get(tool_call_id)
        if record:
            record.status = status
            record.completed_at = datetime.now(timezone.utc).isoformat()
            if error:
                record.last_error = error
            self.committed_tool_result_ids.add(tool_call_id)

    def has_committed(self, tool_call_id: str) -> bool:
        """检查 tool_call_id 是否已有 committed 的 result"""
        return tool_call_id in self.committed_tool_result_ids

    def get_record(self, tool_call_id: str) -> ToolCallRecord | None:
        """获取工具调用的完整记录"""
        return self.tool_calls.get(tool_call_id)

    def get_pending(self, tool_call_id: str) -> ToolCallRecord | None:
        """获取待完成的工具调用（兼容旧接口）"""
        record = self.tool_calls.get(tool_call_id)
        if record and record.status in (ToolCallStatus.PENDING, ToolCallStatus.EXECUTING):
            return record
        return None

    def get_idempotency_key(self, tool_call_id: str) -> str | None:
        """获取工具调用的幂等性 key"""
        record = self.tool_calls.get(tool_call_id)
        return record.idempotency_key if record else None

    def get_action_fingerprint(self, tool_call_id: str) -> str | None:
        """获取工具调用的动作指纹"""
        record = self.tool_calls.get(tool_call_id)
        return record.action_fingerprint if record else None

    def check_completeness(self, expected_ids: list[str] | None = None) -> list[str]:
        """检查协议完整性：返回所有缺少 tool result 的 tool_call_id。

        Args:
            expected_ids: 预期的 tool_call_id 列表（assistant batch 中的）。
                         如果为 None，使用 self.batch_tool_call_ids。
        """
        ids_to_check = expected_ids if expected_ids is not None else self.batch_tool_call_ids
        return [
            tc_id for tc_id in ids_to_check
            if tc_id not in self.committed_tool_result_ids
        ]

    def set_run_status(self, status: RunStatus) -> None:
        """设置整个 run 的终态"""
        self.run_status = status

    def increment_iteration(self) -> None:
        """增加迭代计数"""
        self.iteration += 1

    def increment_retry(self) -> None:
        """增加重试计数"""
        self.retry_count += 1

    def to_dict(self) -> dict:
        """序列化为可存储的字典"""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "run_status": self.run_status.value,
            "iteration": self.iteration,
            "retry_count": self.retry_count,
            "tool_calls": {k: v.to_dict() for k, v in self.tool_calls.items()},
            "committed_tool_result_ids": list(self.committed_tool_result_ids),
            "assistant_message_committed": self.assistant_message_committed,
            "batch_tool_call_ids": list(self.batch_tool_call_ids),
        }

    @classmethod
    def from_dict(cls, data: dict) -> RecoveryState:
        """从字典反序列化"""
        run_status_str = data.get("run_status", "running")
        try:
            run_status = RunStatus(run_status_str)
        except ValueError:
            run_status = RunStatus.RUNNING

        state = cls(
            schema_version=data.get("schema_version", CURRENT_SCHEMA_VERSION),
            run_id=data.get("run_id", uuid.uuid4().hex[:16]),
            turn_id=data.get("turn_id", uuid.uuid4().hex[:16]),
            run_status=run_status,
            iteration=data.get("iteration", 0),
            retry_count=data.get("retry_count", 0),
            committed_tool_result_ids=set(data.get("committed_tool_result_ids", [])),
            assistant_message_committed=data.get("assistant_message_committed", False),
            batch_tool_call_ids=data.get("batch_tool_call_ids", []),
        )
        for tc_id, tc_data in data.get("tool_calls", {}).items():
            state.tool_calls[tc_id] = ToolCallRecord.from_dict(tc_data)
        return state

    def to_event_log(self) -> dict:
        """生成事件日志快照（不含敏感参数）。"""
        return {
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "run_status": self.run_status.value,
            "iteration": self.iteration,
            "retry_count": self.retry_count,
            "committed_count": len(self.committed_tool_result_ids),
            "tool_calls": [
                {
                    "tool_call_id": tc_id,
                    "tool_name": tc.tool_name,
                    "fingerprint": tc.action_fingerprint[:12] + "...",
                    "status": tc.status.value,
                    "attempts": tc.attempts,
                }
                for tc_id, tc in self.tool_calls.items()
            ],
        }
