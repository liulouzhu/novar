"""novare/reflexion/types.py — Reflexion 结构化状态与记录

ReflexionState 与 TaskState / RecoveryState 职责分离：
- TaskState:      用户目标、已完成步骤、待办、关键发现（注入 system prompt）
- RecoveryState:  tool-call 协议、幂等性、执行恢复
- ReflexionState: 语义失败、触发历史、禁止重复动作、修订计划

安全约束：
- 不保存完整 chain-of-thought，只保存简洁、可审计的 diagnosis（限长）
- 不保存未脱敏错误、密钥或完整敏感工具参数
- suggested_next_action 只是建议，不能绕过 AgentLoop 直接执行
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable

CURRENT_SCHEMA_VERSION = 3

# 诊断 / 变更 / 计划条目的最大长度（字符）
MAX_DIAGNOSIS_LENGTH = 300
MAX_PLAN_ITEM_LENGTH = 200
MAX_RECORDS_KEPT = 20

# 计数上限（防止损坏数据导致异常状态）
MAX_REFLECTION_COUNT = 1000
MAX_NO_PROGRESS_COUNT = 10000

# 集合数量上限（确定性、有界）
MAX_PROGRESS_SIGNAL_DIGESTS = 512
MAX_FORBIDDEN_FINGERPRINTS = 512
MAX_REFLECTED_FINGERPRINTS = 1000
MAX_BLOCKED_REASON_LENGTH = 500

# ReflectionRecord 字段限制
MAX_RECORD_ID_LENGTH = 64
MAX_RECORD_TRIGGER_FP_LENGTH = 200
MAX_RECORD_LIST_ITEMS = 20
MAX_EVIDENCE_REFS = 20
MAX_EVIDENCE_REF_LENGTH = 100
MAX_FAILURE_TYPE_LENGTH = 80

_ACTION_FP_RE = re.compile(r"[0-9a-f]{64}")


class ReflectionTrigger(str, Enum):
    """确定性触发规则。Reflexion 只能由这些规则触发。"""

    SEMANTIC_TOOL_FAILURE = "semantic_tool_failure"
    REPEATED_FAILED_ACTION = "repeated_failed_action"
    NO_PROGRESS = "no_progress"
    CONFLICTING_OBSERVATIONS = "conflicting_observations"
    RETRY_EXHAUSTED_WITH_ALTERNATIVE = "retry_exhausted_with_alternative"


# 触发器 → trigger_fingerprint 前缀（严格恢复校验用）
_TRIGGER_FP_PREFIXES = {
    ReflectionTrigger.SEMANTIC_TOOL_FAILURE.value: "semantic_tool_failure:",
    ReflectionTrigger.REPEATED_FAILED_ACTION.value: "repeated_failed_action:",
    ReflectionTrigger.NO_PROGRESS.value: "no_progress:",
    ReflectionTrigger.CONFLICTING_OBSERVATIONS.value: "conflicting_observations:",
    ReflectionTrigger.RETRY_EXHAUSTED_WITH_ALTERNATIVE.value: "retry_exhausted:",
}


class ReflectionDecision(str, Enum):
    """反思模型允许返回的决策枚举。不允许"自动重试相同动作"。"""

    REPLAN = "REPLAN"
    ASK_USER = "ASK_USER"
    STOP = "STOP"
    CONTINUE_WITH_LIMITATION = "CONTINUE_WITH_LIMITATION"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(text: str | None, limit: int) -> str:
    if not text:
        return ""
    text = str(text).strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _truncate_list(items: list | None, limit: int) -> list[str]:
    if not items:
        return []
    out = []
    for item in items:
        out.append(_truncate(str(item), limit))
    return out[: limit]


@dataclass
class ReflectionRecord:
    """一条已提交（validated+applied）的反思记录。"""

    reflection_id: str
    trigger: str
    trigger_fingerprint: str
    evidence_refs: list[str]
    failure_type: str
    diagnosis: str
    preserve: list[str]
    changes: list[str]
    forbidden_action_fingerprints: list[str]
    revised_plan: list[str]
    suggested_next_action: dict | None
    decision: str
    validated: bool
    applied: bool
    created_at: str

    def to_dict(self) -> dict:
        return {
            "reflection_id": self.reflection_id,
            "trigger": self.trigger,
            "trigger_fingerprint": self.trigger_fingerprint,
            "evidence_refs": list(self.evidence_refs),
            "failure_type": self.failure_type,
            "diagnosis": self.diagnosis,
            "preserve": list(self.preserve),
            "changes": list(self.changes),
            "forbidden_action_fingerprints": list(self.forbidden_action_fingerprints),
            "revised_plan": list(self.revised_plan),
            "suggested_next_action": self.suggested_next_action,
            "decision": self.decision,
            "validated": self.validated,
            "applied": self.applied,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReflectionRecord":
        """宽松解析（内部工具 / 旧数据使用；恢复不可信数据请用 from_dict_strict）。"""
        return cls(
            reflection_id=str(data.get("reflection_id", "")),
            trigger=str(data.get("trigger", "")),
            trigger_fingerprint=str(data.get("trigger_fingerprint", "")),
            evidence_refs=[str(x) for x in data.get("evidence_refs", [])],
            failure_type=str(data.get("failure_type", "")),
            diagnosis=str(data.get("diagnosis", "")),
            preserve=[str(x) for x in data.get("preserve", [])],
            changes=[str(x) for x in data.get("changes", [])],
            forbidden_action_fingerprints=[
                str(x) for x in data.get("forbidden_action_fingerprints", [])
            ],
            revised_plan=[str(x) for x in data.get("revised_plan", [])],
            suggested_next_action=data.get("suggested_next_action"),
            decision=str(data.get("decision", "")),
            validated=bool(data.get("validated", False)),
            applied=bool(data.get("applied", False)),
            created_at=str(data.get("created_at", "")),
        )

    @classmethod
    def from_dict_strict(cls, data: dict) -> "ReflectionRecord":
        """严格恢复不可信持久化数据。

        逐字段类型/长度/格式校验，禁止 str(value) 自动转换错误类型，
        禁止把标量当作 list。任何字段非法抛 InvalidReflexionStateError。
        """
        if not isinstance(data, dict):
            raise InvalidReflexionStateError("record must be a dict")

        reflection_id = _strict_str(data.get("reflection_id"), "reflection_id", required=True, max_len=MAX_RECORD_ID_LENGTH)
        trigger = _strict_str(data.get("trigger"), "trigger", required=True, max_len=MAX_RECORD_ID_LENGTH)
        if trigger not in ReflectionTrigger._value2member_map_:
            raise InvalidReflexionStateError(f"invalid record trigger: {trigger!r}")
        trigger_fingerprint = _strict_str(
            data.get("trigger_fingerprint"), "trigger_fingerprint",
            required=True, max_len=MAX_RECORD_TRIGGER_FP_LENGTH,
        )
        prefix = _TRIGGER_FP_PREFIXES.get(trigger)
        if not prefix or not trigger_fingerprint.startswith(prefix):
            raise InvalidReflexionStateError(
                f"trigger_fingerprint does not match trigger {trigger!r}"
            )
        failure_type = _strict_str(data.get("failure_type"), "failure_type", required=True, max_len=MAX_FAILURE_TYPE_LENGTH)
        diagnosis = _strict_str(data.get("diagnosis"), "diagnosis", required=True, max_len=MAX_DIAGNOSIS_LENGTH)
        created_at = _strict_str(data.get("created_at"), "created_at", required=True, max_len=MAX_RECORD_ID_LENGTH)

        evidence_refs = _strict_str_list(data.get("evidence_refs"), "evidence_refs", max_items=MAX_EVIDENCE_REFS, max_item_len=MAX_EVIDENCE_REF_LENGTH)
        preserve = _strict_str_list(data.get("preserve"), "preserve", max_items=MAX_RECORD_LIST_ITEMS, max_item_len=MAX_PLAN_ITEM_LENGTH)
        changes = _strict_str_list(data.get("changes"), "changes", max_items=MAX_RECORD_LIST_ITEMS, max_item_len=MAX_PLAN_ITEM_LENGTH, require_nonempty=True)
        revised_plan = _strict_str_list(data.get("revised_plan"), "revised_plan", max_items=MAX_RECORD_LIST_ITEMS, max_item_len=MAX_PLAN_ITEM_LENGTH, require_nonempty=True)

        forbidden_raw = data.get("forbidden_action_fingerprints", [])
        if not isinstance(forbidden_raw, list):
            raise InvalidReflexionStateError("forbidden_action_fingerprints must be a list")
        forbidden: list[str] = []
        for fp in forbidden_raw:
            if not isinstance(fp, str) or not _ACTION_FP_RE.fullmatch(fp):
                raise InvalidReflexionStateError("invalid forbidden fingerprint in record")
            forbidden.append(fp)

        suggested = data.get("suggested_next_action")
        if suggested is not None:
            if not isinstance(suggested, dict):
                raise InvalidReflexionStateError("suggested_next_action must be a dict or null")
            tool = suggested.get("tool")
            arguments = suggested.get("arguments")
            if not isinstance(tool, str) or not tool or len(tool) > MAX_RECORD_ID_LENGTH:
                raise InvalidReflexionStateError("suggested_next_action.tool invalid")
            if not isinstance(arguments, dict):
                raise InvalidReflexionStateError("suggested_next_action.arguments must be a dict")

        decision = _strict_str(data.get("decision"), "decision", required=True, max_len=MAX_RECORD_ID_LENGTH)
        if decision not in ReflectionDecision._value2member_map_:
            raise InvalidReflexionStateError(f"invalid record decision: {decision!r}")
        validated = data.get("validated")
        applied = data.get("applied")
        if validated is not True or applied is not True:
            raise InvalidReflexionStateError("record validated/applied must be strictly True")

        return cls(
            reflection_id=reflection_id,
            trigger=trigger,
            trigger_fingerprint=trigger_fingerprint,
            evidence_refs=evidence_refs,
            failure_type=failure_type,
            diagnosis=diagnosis,
            preserve=preserve,
            changes=changes,
            forbidden_action_fingerprints=forbidden,
            revised_plan=revised_plan,
            suggested_next_action=suggested,
            decision=decision,
            validated=validated,
            applied=applied,
            created_at=created_at,
        )


class InvalidReflexionStateError(ValueError):
    """ReflexionState 恢复数据不合法（schema/type/invariant 校验失败）。"""


@dataclass
class ReflexionState:
    """一次 run_turn 的 Reflexion 状态（turn-scoped，可序列化恢复）。"""

    schema_version: int = CURRENT_SCHEMA_VERSION
    reflection_count: int = 0
    no_progress_count: int = 0
    last_progress_fingerprint: str | None = None
    reflected_trigger_fingerprints: set[str] = field(default_factory=set)
    forbidden_action_fingerprints: set[str] = field(default_factory=set)
    records: list[ReflectionRecord] = field(default_factory=list)
    blocked_reason: str | None = None
    # 进展信号 digest 集合（64-hex，仅 digest 不含明文 summary/文本），
    # 用于跨进程恢复进展检测；fingerprint = SHA-256(canonical(sorted(digests)))
    progress_signal_digests: set[str] = field(default_factory=set)

    # ── 查询 ──
    def budget_exhausted(self, max_reflections: int) -> bool:
        return self.reflection_count >= max_reflections

    def already_reflected(self, trigger_fingerprint: str) -> bool:
        return trigger_fingerprint in self.reflected_trigger_fingerprints

    def is_forbidden(self, action_fingerprint: str) -> bool:
        return action_fingerprint in self.forbidden_action_fingerprints

    # ── 变更 ──
    def record_progress(self) -> None:
        """出现真实进展：no_progress_count 归零。"""
        self.no_progress_count = 0

    def record_no_progress(self) -> None:
        self.no_progress_count += 1

    def add_reflection(self, record: ReflectionRecord) -> None:
        self.records.append(record)
        if len(self.records) > MAX_RECORDS_KEPT:
            self.records = self.records[-MAX_RECORDS_KEPT:]
        self.reflection_count += 1
        self.reflected_trigger_fingerprints.add(record.trigger_fingerprint)
        for fp in record.forbidden_action_fingerprints:
            if fp:
                self.forbidden_action_fingerprints.add(fp)

    def block(self, reason: str) -> None:
        self.blocked_reason = reason

    # ── 序列化 ──
    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "reflection_count": self.reflection_count,
            "no_progress_count": self.no_progress_count,
            "last_progress_fingerprint": self.last_progress_fingerprint,
            "reflected_trigger_fingerprints": sorted(self.reflected_trigger_fingerprints),
            "forbidden_action_fingerprints": sorted(self.forbidden_action_fingerprints),
            "records": [r.to_dict() for r in self.records],
            "blocked_reason": self.blocked_reason,
            "progress_signal_digests": sorted(self.progress_signal_digests),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "ReflexionState":
        """从 dict 恢复 ReflexionState（严格 schema/type/invariant 校验）。

        - 未知 schema_version 拒绝（fail closed，需显式 migration）；
        - v1 / v2 → v3 显式 migration（见 _from_v1_or_v2）；
        - v3 数据逐字段严格校验 + 状态 invariant 校验；
        - 非法数据抛 InvalidReflexionStateError，不返回部分污染状态。
        """
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise InvalidReflexionStateError("reflexion_state must be a dict")

        schema_version = data.get("schema_version", CURRENT_SCHEMA_VERSION)
        if schema_version in (1, 2):
            return cls._from_v1_or_v2(data)
        if schema_version != CURRENT_SCHEMA_VERSION:
            raise InvalidReflexionStateError(
                f"unsupported reflexion_state schema_version: {schema_version!r}"
            )
        return cls._from_v3(data)

    @classmethod
    def _from_v1_or_v2(cls, data: dict) -> "ReflexionState":
        """显式 migration v1 / v2 → v3（宽容但无损可审计）。

        - v2 的 cumulative_success_signals（可能为明文）哈希为 digest 并丢弃明文；
        - v1 无该字段 → 空 digest 集合；
        - v2 未持久化 completed/key_findings 原文，无法无损重建旧 fingerprint：
          以"由迁移后 digest 集合计算的 fingerprint"作为安全 baseline
          （可仅凭持久化状态重建，不宣称完整恢复）；
        - records 尝试严格解析，失败则丢弃（不污染状态）。
        """
        old_signals = data.get("cumulative_success_signals", []) or []
        signal_digests: set[str] = set()
        if isinstance(old_signals, list):
            for signal in old_signals:
                if isinstance(signal, str) and signal:
                    signal_digests.add(hashlib.sha256(signal.encode("utf-8")).hexdigest())
        signal_digests = _cap_set(signal_digests, MAX_PROGRESS_SIGNAL_DIGESTS)
        last_progress_fingerprint = compute_fingerprint_from_digests(signal_digests)

        reflection_count = _bounded_int(data.get("reflection_count", 0), "reflection_count", MAX_REFLECTION_COUNT)
        no_progress_count = _bounded_int(data.get("no_progress_count", 0), "no_progress_count", MAX_NO_PROGRESS_COUNT)

        reflected = _loose_string_set(data.get("reflected_trigger_fingerprints"))
        forbidden = _loose_fingerprint_set(data.get("forbidden_action_fingerprints"))
        blocked_reason = _loose_str(data.get("blocked_reason"), MAX_BLOCKED_REASON_LENGTH)

        records: list[ReflectionRecord] = []
        records_raw = data.get("records", []) or []
        if not isinstance(records_raw, list):
            records_raw = []
        for rec in records_raw:
            if not isinstance(rec, dict):
                continue
            if rec.get("validated") is not True or rec.get("applied") is not True:
                continue
            try:
                records.append(ReflectionRecord.from_dict_strict(rec))
            except InvalidReflexionStateError:
                continue  # 旧格式记录无法严格解析 → 丢弃，不污染状态
        records = records[-MAX_RECORDS_KEPT:]

        # 由保留记录重建 invariant（trigger_fingerprint ∈ reflected，forbidden ⊆ state）
        for rec in records:
            reflected.add(rec.trigger_fingerprint)
            forbidden.update(rec.forbidden_action_fingerprints)
        reflected = _cap_set(reflected, MAX_REFLECTED_FINGERPRINTS)
        forbidden = _cap_set(forbidden, MAX_FORBIDDEN_FINGERPRINTS)

        return cls(
            schema_version=CURRENT_SCHEMA_VERSION,
            reflection_count=reflection_count,
            no_progress_count=no_progress_count,
            last_progress_fingerprint=last_progress_fingerprint,
            reflected_trigger_fingerprints=reflected,
            forbidden_action_fingerprints=forbidden,
            records=records,
            blocked_reason=blocked_reason,
            progress_signal_digests=signal_digests,
        )

    @classmethod
    def _from_v3(cls, data: dict) -> "ReflexionState":
        """当前 schema 严格恢复（逐字段 + invariant 校验）。"""
        reflection_count = _bounded_int(data.get("reflection_count", 0), "reflection_count", MAX_REFLECTION_COUNT)
        no_progress_count = _bounded_int(data.get("no_progress_count", 0), "no_progress_count", MAX_NO_PROGRESS_COUNT)

        last_progress_fingerprint = data.get("last_progress_fingerprint")
        if last_progress_fingerprint is not None:
            if not isinstance(last_progress_fingerprint, str) or not _ACTION_FP_RE.fullmatch(last_progress_fingerprint):
                raise InvalidReflexionStateError("last_progress_fingerprint must be 64-hex or null")

        blocked_reason = data.get("blocked_reason")
        if blocked_reason is not None:
            if not isinstance(blocked_reason, str) or len(blocked_reason) > MAX_BLOCKED_REASON_LENGTH:
                raise InvalidReflexionStateError("blocked_reason must be a short string or null")

        reflected = _strict_string_set(data.get("reflected_trigger_fingerprints", []), "reflected_trigger_fingerprints")
        if len(reflected) > MAX_REFLECTED_FINGERPRINTS:
            raise InvalidReflexionStateError("reflected_trigger_fingerprints exceeds limit")
        forbidden = _strict_fingerprint_set(data.get("forbidden_action_fingerprints", []), "forbidden_action_fingerprints")
        if len(forbidden) > MAX_FORBIDDEN_FINGERPRINTS:
            raise InvalidReflexionStateError("forbidden_action_fingerprints exceeds limit")
        digests = _strict_fingerprint_set(data.get("progress_signal_digests", []), "progress_signal_digests")
        if len(digests) > MAX_PROGRESS_SIGNAL_DIGESTS:
            raise InvalidReflexionStateError("progress_signal_digests exceeds limit")

        records_raw = data.get("records", [])
        if not isinstance(records_raw, list):
            raise InvalidReflexionStateError("records must be a list")
        records: list[ReflectionRecord] = []
        for rec in records_raw:
            if not isinstance(rec, dict):
                raise InvalidReflexionStateError("each record must be a dict")
            if rec.get("validated") is not True or rec.get("applied") is not True:
                raise InvalidReflexionStateError("only validated+applied records are restorable")
            records.append(ReflectionRecord.from_dict_strict(rec))

        # ── 状态 invariant ──
        if len(records) > MAX_RECORDS_KEPT:
            raise InvalidReflexionStateError("records exceeds MAX_RECORDS_KEPT")
        if reflection_count < len(records):
            raise InvalidReflexionStateError("reflection_count < len(records)")
        for rec in records:
            if rec.trigger_fingerprint not in reflected:
                raise InvalidReflexionStateError("record trigger_fingerprint missing from reflected_trigger_fingerprints")
            for fp in rec.forbidden_action_fingerprints:
                if fp not in forbidden:
                    raise InvalidReflexionStateError("record forbidden fingerprint missing from state forbidden set")

        return cls(
            schema_version=CURRENT_SCHEMA_VERSION,
            reflection_count=reflection_count,
            no_progress_count=no_progress_count,
            last_progress_fingerprint=last_progress_fingerprint,
            reflected_trigger_fingerprints=reflected,
            forbidden_action_fingerprints=forbidden,
            records=records,
            blocked_reason=blocked_reason,
            progress_signal_digests=digests,
        )


def _strict_string_set(value, field: str) -> set[str]:
    """严格校验 list[str]（元素为非空字符串）→ set。非法抛 InvalidReflexionStateError。"""
    if not isinstance(value, list):
        raise InvalidReflexionStateError(f"{field} must be a list of strings")
    out: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise InvalidReflexionStateError(f"{field} items must be non-empty strings")
        out.add(item)
    return out


def _strict_fingerprint_set(value, field: str) -> set[str]:
    """严格校验 list[str] 且每项为 64 位小写十六进制 SHA-256 → set。"""
    if not isinstance(value, list):
        raise InvalidReflexionStateError(f"{field} must be a list of fingerprints")
    out: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not _ACTION_FP_RE.fullmatch(item):
            raise InvalidReflexionStateError(f"{field} items must be 64-hex SHA-256 fingerprints")
        out.add(item)
    return out


def _strict_str(value, field: str, *, required: bool, max_len: int) -> str:
    """严格非空、限长字符串（禁止 str(value) 自动转换错误类型）。"""
    if not isinstance(value, str):
        raise InvalidReflexionStateError(f"{field} must be a string")
    value = value.strip()
    if required and not value:
        raise InvalidReflexionStateError(f"{field} must be a non-empty string")
    if len(value) > max_len:
        raise InvalidReflexionStateError(f"{field} too long")
    return value


def _strict_str_list(value, field: str, *, max_items: int, max_item_len: int, require_nonempty: bool = False) -> list[str]:
    """严格 list[str]（数量/每项长度受限；禁止标量自动转 list）。"""
    if not isinstance(value, list):
        raise InvalidReflexionStateError(f"{field} must be a list of strings")
    if len(value) > max_items:
        raise InvalidReflexionStateError(f"{field} exceeds item limit")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise InvalidReflexionStateError(f"{field} items must be non-empty strings")
        if len(item) > max_item_len:
            raise InvalidReflexionStateError(f"{field} item too long")
        out.append(item.strip())
    if require_nonempty and not out:
        raise InvalidReflexionStateError(f"{field} must be non-empty")
    return out


def _bounded_int(value, field: str, limit: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > limit:
        raise InvalidReflexionStateError(f"{field} must be a non-negative int <= {limit}")
    return value


def _cap_set(values: set[str], limit: int) -> set[str]:
    """确定性、有界：超出 limit 时按字典序移除最小项。"""
    while len(values) > limit:
        values.remove(min(values))
    return values


def _loose_str(value, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value:
        return None
    return value[:limit]


def _loose_string_set(value) -> set[str]:
    if not isinstance(value, list):
        return set()
    out: set[str] = set()
    for item in value:
        if isinstance(item, str) and item.strip():
            out.add(item.strip())
    return out


def _loose_fingerprint_set(value) -> set[str]:
    if not isinstance(value, list):
        return set()
    out: set[str] = set()
    for item in value:
        if isinstance(item, str) and _ACTION_FP_RE.fullmatch(item):
            out.add(item)
    return out


def compute_fingerprint_from_digests(digests: Iterable[str]) -> str:
    """由 digest 集合计算进展指纹（与 progress.progress 包一致，供恢复重建用）。"""
    from novare.reflexion.progress import compute_progress_fingerprint

    return compute_progress_fingerprint(signal_digests=digests)


def new_reflection_id() -> str:
    return f"refl_{uuid.uuid4().hex[:12]}"


def make_reflection_record(
    *,
    trigger: ReflectionTrigger | str,
    trigger_fingerprint: str,
    evidence_refs: list[str],
    failure_type: str,
    diagnosis: str,
    preserve: list[str],
    changes: list[str],
    forbidden_action_fingerprints: list[str],
    revised_plan: list[str],
    suggested_next_action: dict | None,
    decision: ReflectionDecision | str,
    validated: bool,
    applied: bool,
) -> ReflectionRecord:
    """构造 ReflectionRecord，自动截断超长字段。"""
    return ReflectionRecord(
        reflection_id=new_reflection_id(),
        trigger=trigger.value if isinstance(trigger, ReflectionTrigger) else str(trigger),
        trigger_fingerprint=str(trigger_fingerprint),
        evidence_refs=list(evidence_refs or []),
        failure_type=_truncate(failure_type, 80),
        diagnosis=_truncate(diagnosis, MAX_DIAGNOSIS_LENGTH),
        preserve=_truncate_list(preserve, MAX_PLAN_ITEM_LENGTH),
        changes=_truncate_list(changes, MAX_PLAN_ITEM_LENGTH),
        forbidden_action_fingerprints=list(forbidden_action_fingerprints or []),
        revised_plan=_truncate_list(revised_plan, MAX_PLAN_ITEM_LENGTH),
        suggested_next_action=suggested_next_action,
        decision=decision.value if isinstance(decision, ReflectionDecision) else str(decision),
        validated=validated,
        applied=applied,
        created_at=_now_iso(),
    )


def dump_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)
