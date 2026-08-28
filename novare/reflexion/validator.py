"""novare/reflexion/validator.py — 结构化输出与安全验证

验证失败：允许一次格式修复；再次失败则记录 REFLECTION_REJECTED，
不应用计划修改，不无限调用反思模型。
"""

from __future__ import annotations

import json
import re
from typing import Callable, Iterable

from novare.reflexion.types import ReflectionDecision, MAX_DIAGNOSIS_LENGTH, MAX_PLAN_ITEM_LENGTH
from novare.reflexion.triggers import compute_action_fingerprint

# 删除性/绕过性动词（用于安全约束保护）
_DELETION_MARKERS = (
    "删除", "移除", "取消", "disable", "remove", "drop", "bypass", "绕过", "忽略", "ignore",
)
_GOAL_CHANGE_MARKERS = (
    "修改目标", "改变目标", "替换目标", "rewrite the goal", "change the goal",
    "replace the goal", "modify the goal", "new goal",
)
_SENSITIVE_MARKERS = (
    "sk-", "rk-", "pk-", "bearer ", "authorization", "api_key", "apikey", "api-key",
    "password", "secret", "token=",
)

VALID_DECISIONS = {d.value for d in ReflectionDecision}


def _items(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def _normalize_string_list(value, field: str, *, require_nonempty: bool) -> list[str] | None:
    """严格归一化字符串列表。

    只接受 list[str]（每项 trim 后非空，若 require_nonempty）；
    禁止把字符串/dict/数字自动转换成 list。非法返回 None。
    """
    if value is None:
        if require_nonempty:
            return None
        return []
    if not isinstance(value, list):
        return None
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        stripped = item.strip()
        if not stripped:
            return None
        out.append(stripped)
    if require_nonempty and not out:
        return None
    return out


def _plain_text(value) -> str:
    return " ".join(_items(value))


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    lower = text.lower()
    return any(m.lower() in lower for m in markers)


class ReflectionValidator:
    """验证反思模型输出。

    Args:
        tool_registry: 工具执行器（查询工具定义 / allowlist）
        available_tool_names: 当前 Agent 允许使用的工具名
        user_goal: 不可变用户目标
        safety_constraints: 安全约束/权限限制（不得被删除）
        real_event_ids: 本次轨迹中真实存在的事件 ID（evidence_refs 必须引用它们）
        triggering_action_fingerprint: 导致触发的失败动作 fingerprint（裸）
        failed_action_fingerprint: 刚失败的相同动作 fingerprint（与上面相同或关联）
        failed_tool: 失败工具名
        failed_arguments: 失败参数
        idempotency: 失败工具的幂等性
    """

    def __init__(
        self,
        *,
        tool_registry=None,
        available_tool_names: Iterable[str] | None = None,
        user_goal: str = "",
        safety_constraints: Iterable[str] | None = None,
        real_event_ids: Iterable[str] | None = None,
        triggering_action_fingerprint: str | None = None,
        required_evidence_refs: Iterable[str] | None = None,
        existing_forbidden_action_fingerprints: Iterable[str] | None = None,
        failed_tool: str | None = None,
        failed_arguments: dict | None = None,
        idempotency: str = "non_idempotent",
    ):
        self.tool_registry = tool_registry
        self.available_tool_names = set(available_tool_names or [])
        self.user_goal = user_goal or ""
        self.safety_constraints = list(safety_constraints or [])
        self.real_event_ids = set(real_event_ids or [])
        self.triggering_action_fingerprint = triggering_action_fingerprint
        # 失败触发器必须引用的 evidence reference（触发器提供，如 "event:<id>"）
        self.required_evidence_refs = list(required_evidence_refs or [])
        # ReflexionState 已有 forbidden 集合（历史反思 + 本次恢复）
        self.existing_forbidden_action_fingerprints = set(
            existing_forbidden_action_fingerprints or []
        )
        self.failed_tool = failed_tool
        self.failed_arguments = failed_arguments or {}
        self.idempotency = idempotency
        self._tool_getter = getattr(tool_registry, "get_tool", None) if tool_registry else None

    # ── 主入口 ──
    def validate(self, output: dict) -> tuple[bool, str]:
        """返回 (是否通过, 拒绝原因)。兼容旧调用方。"""
        normalized, reason = self.normalize_and_validate(output)
        return (normalized is not None), reason

    def normalize_and_validate(self, output: dict) -> tuple[dict | None, str]:
        """严格结构校验 + 业务校验。

        返回 (归一化后的输出 dict, 拒绝原因)。归一化输出是 engine 构建
        ReflectionRecord 的唯一数据源；字符串/数字不会被自动转换成 list。
        """
        normalized, reason = self._normalize_structure(output)
        if normalized is None:
            return None, reason
        ok, reason = self._validate_normalized(normalized)
        if not ok:
            return None, reason
        return normalized, ""

    # ── 严格结构校验（禁止标量/数字自动转 list）──
    @staticmethod
    def _normalize_structure(output: dict) -> tuple[dict | None, str]:
        if not isinstance(output, dict):
            return None, "output must be a dict"
        # 模型不得携带/修改用户目标字段
        if "goal" in output or "user_goal" in output:
            return None, "output must not modify user goal"

        decision = output.get("decision")
        if not isinstance(decision, str) or not decision.strip():
            return None, "decision must be a non-empty string"
        if decision not in VALID_DECISIONS:
            return None, f"invalid decision: {decision!r}"

        failure_type = output.get("failure_type")
        if not isinstance(failure_type, str):
            return None, "failure_type must be a string"

        diagnosis = output.get("diagnosis")
        if not isinstance(diagnosis, str) or not diagnosis.strip():
            return None, "diagnosis must be a non-empty string"

        norm_changes = _normalize_string_list(output.get("changes"), "changes", require_nonempty=True)
        if norm_changes is None:
            return None, "changes must be a non-empty list of non-empty strings"
        norm_plan = _normalize_string_list(output.get("revised_plan"), "revised_plan", require_nonempty=True)
        if norm_plan is None:
            return None, "revised_plan must be a non-empty list of non-empty strings"
        norm_preserve = _normalize_string_list(output.get("preserve"), "preserve", require_nonempty=False)
        if norm_preserve is None:
            return None, "preserve must be a list of strings"
        norm_evidence = _normalize_string_list(output.get("evidence_refs"), "evidence_refs", require_nonempty=False)
        if norm_evidence is None:
            return None, "evidence_refs must be a list of strings"
        norm_forbidden = _normalize_string_list(output.get("forbidden_repeat"), "forbidden_repeat", require_nonempty=False)
        if norm_forbidden is None:
            return None, "forbidden_repeat must be a list of strings"

        # suggested_next_action：只能是 null 或 dict{tool: str, arguments: dict}
        suggested = output.get("suggested_next_action")
        norm_suggested = None
        if suggested is not None:
            if not isinstance(suggested, dict):
                return None, "suggested_next_action must be a dict or null"
            tool = suggested.get("tool")
            arguments = suggested.get("arguments")
            if not isinstance(tool, str) or not tool.strip():
                return None, "suggested_next_action.tool must be a non-empty string"
            if not isinstance(arguments, dict):
                return None, "suggested_next_action.arguments must be a dict"
            norm_suggested = {"tool": tool.strip(), "arguments": arguments}

        return {
            "decision": decision,
            "failure_type": failure_type,
            "diagnosis": diagnosis.strip(),
            "changes": norm_changes,
            "revised_plan": norm_plan,
            "preserve": norm_preserve,
            "evidence_refs": norm_evidence,
            "forbidden_repeat": norm_forbidden,
            "suggested_next_action": norm_suggested,
        }, ""

    # ── 业务校验（输入已归一化）──
    def _validate_normalized(self, output: dict) -> tuple[bool, str]:
        checks = [
            self._check_goal_unchanged,
            self._check_safety_constraints,
            self._check_field_lengths,
            self._check_evidence_refs,
            self._check_forbidden_repeat,
            self._check_no_sensitive_content,
            self._check_suggested_action,
        ]
        for check in checks:
            ok, reason = check(output)
            if not ok:
                return False, reason
        return True, ""

    # ── 单项检查 ──
    def _check_goal_unchanged(self, output: dict) -> tuple[bool, str]:
        for field in ("diagnosis", "changes", "revised_plan", "preserve"):
            text = _plain_text(output.get(field))
            if _contains_any(text, _GOAL_CHANGE_MARKERS) and self.user_goal:
                return False, "output must not modify user goal"
        return True, ""

    def _check_safety_constraints(self, output: dict) -> tuple[bool, str]:
        if not self.safety_constraints:
            return True, ""
        for field in ("changes", "revised_plan", "diagnosis"):
            text = _plain_text(output.get(field))
            if not _contains_any(text, _DELETION_MARKERS):
                continue
            for constraint in self.safety_constraints:
                if constraint and constraint.lower() in text.lower():
                    return False, f"output must not remove safety constraint: {constraint}"
        return True, ""

    def _check_field_lengths(self, output: dict) -> tuple[bool, str]:
        if len(str(output.get("diagnosis") or "")) > MAX_DIAGNOSIS_LENGTH:
            return False, "diagnosis too long"
        for field in ("changes", "revised_plan", "preserve"):
            for item in output.get(field, []):
                if len(str(item)) > MAX_PLAN_ITEM_LENGTH:
                    return False, f"{field} item too long"
        return True, ""

    def _check_evidence_refs(self, output: dict) -> tuple[bool, str]:
        evidence_refs = output.get("evidence_refs", [])
        # 有确定性 evidence 的失败触发器：evidence_refs 非空且包含触发器提供的 reference
        if self.required_evidence_refs:
            for ref in self.required_evidence_refs:
                if ref not in evidence_refs:
                    return False, f"evidence_refs must include trigger evidence: {ref}"
        if self.triggering_action_fingerprint and not evidence_refs:
            return False, "evidence_refs must be non-empty for failed-action triggers"
        if self.real_event_ids:
            for ref in evidence_refs:
                if not ref.startswith("event:"):
                    return False, f"invalid evidence ref: {ref}"
                event_id = ref[len("event:") :]
                if event_id not in self.real_event_ids:
                    return False, f"evidence ref does not reference real event: {ref}"
        return True, ""

    def _check_forbidden_repeat(self, output: dict) -> tuple[bool, str]:
        forbidden = [x for x in output.get("forbidden_repeat", []) if x.strip()]
        # 非空 fingerprint 必须是合法 64 位十六进制 SHA-256 action fingerprint
        for fp in forbidden:
            if not re.fullmatch(r"[0-9a-f]{64}", fp):
                return False, f"invalid action fingerprint in forbidden_repeat: {fp!r}"
        # 对具有 triggering failed action 的触发器，必须包含该 fingerprint
        if self.triggering_action_fingerprint:
            if self.triggering_action_fingerprint not in forbidden:
                return False, "forbidden_repeat must include the triggering failure fingerprint"
        return True, ""

    def _check_no_sensitive_content(self, output: dict) -> tuple[bool, str]:
        try:
            raw = json.dumps(output, ensure_ascii=False)
        except (TypeError, ValueError):
            return False, "output is not JSON-serializable"
        if _contains_any(raw, _SENSITIVE_MARKERS):
            return False, "output must not contain keys or authorization material"
        return True, ""

    def _check_suggested_action(self, output: dict) -> tuple[bool, str]:
        suggested = output.get("suggested_next_action")
        if suggested is None:
            return True, ""
        tool = suggested.get("tool")
        arguments = suggested.get("arguments")
        if not isinstance(tool, str) or not tool or not isinstance(arguments, dict):
            return False, "suggested_next_action must have tool and arguments"

        # 工具必须存在且允许
        if self.available_tool_names and tool not in self.available_tool_names:
            return False, f"suggested tool not allowed: {tool}"
        if self._tool_getter is not None and callable(self._tool_getter):
            try:
                tool_def = self._tool_getter(tool)
            except Exception:
                tool_def = None
            if tool_def is None:
                return False, f"suggested tool does not exist: {tool}"
            schema_ok, schema_reason = validate_arguments_against_schema(arguments, tool_def.parameters)
            if not schema_ok:
                return False, f"suggested arguments invalid: {schema_reason}"

        # 不能等于刚失败的动作 / 不能重放
        if tool == self.failed_tool and arguments == self.failed_arguments:
            return False, "suggested action must not equal the failed action"
        if self.idempotency == "non_idempotent" and tool == self.failed_tool:
            return False, "non_idempotent failure must not suggest replay of the same tool"

        # fingerprint 校验
        fp = compute_action_fingerprint(tool, arguments)
        if fp == self.triggering_action_fingerprint:
            return False, "suggested action fingerprint equals the failed action"
        if fp in self.existing_forbidden_action_fingerprints:
            return False, "suggested action is in the existing forbidden set"
        # 本次输出声明的 forbidden_repeat（非空且合法）同样禁止
        forbidden_here = {x for x in output.get("forbidden_repeat", []) if x.strip()}
        if fp in forbidden_here:
            return False, "suggested action is in forbidden_repeat"
        return True, ""


def validate_arguments_against_schema(arguments: dict, parameters: dict) -> tuple[bool, str]:
    """对工具参数执行最小 JSON Schema 校验（不引入第三方依赖）。

    支持：required / type（string/number/integer/boolean/object/array）/
    properties / required / enum / array.items / 嵌套 object 与 array。
    未知字段不拒绝（宽松兼容）。
    """
    return _validate_value(arguments, parameters, "$")


def _validate_value(value, schema: dict, path: str) -> tuple[bool, str]:
    """递归校验单个值（支持嵌套 object / array / items）。"""
    if not isinstance(schema, dict):
        return True, ""
    expected_type = schema.get("type")
    if expected_type and not _matches_type(value, expected_type):
        return False, f"{path}: expected {expected_type}, got {type(value).__name__}"
    enum_values = schema.get("enum")
    if enum_values and value not in enum_values:
        return False, f"{path}: value not in enum"

    if expected_type == "object" and isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                return False, f"{path}: missing required argument: {req}"
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                ok, reason = _validate_value(item, properties[key], f"{path}.{key}")
                if not ok:
                    return False, reason

    if expected_type == "array" and isinstance(value, list):
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for index, item in enumerate(value):
                ok, reason = _validate_value(item, items_schema, f"{path}[{index}]")
                if not ok:
                    return False, reason
    return True, ""


def _matches_type(value, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    return True


def parse_model_json(raw: str) -> dict | None:
    """解析模型输出 JSON（容忍围栏）。失败返回 None（由调用方决定修复）。"""
    if not raw:
        return None
    text = raw.strip()
    # 去掉 Markdown 围栏
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取第一个 { ... } 块
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    return parsed
