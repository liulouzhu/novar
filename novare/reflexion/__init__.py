"""novare/reflexion — Reflexion Engine（PR 3）

- types.py:      ReflectionTrigger / ReflectionDecision / ReflectionRecord / ReflexionState
- progress.py:   确定性进展指纹与 no-progress 检测
- triggers.py:   确定性触发规则
- prompts.py:    反思模型 system/user prompt（含 untrusted_data 区域）
- engine.py:     ReflexionEngine（复用 PR 1 传输层 Retry）
- validator.py:  结构化输出与安全验证
"""

from novare.reflexion.types import (
    CURRENT_SCHEMA_VERSION,
    InvalidReflexionStateError,
    ReflectionDecision,
    ReflectionRecord,
    ReflectionTrigger,
    ReflexionState,
    make_reflection_record,
)
from novare.reflexion.progress import ProgressTracker, compute_progress_fingerprint
from novare.reflexion.triggers import (
    ToolEventSummary,
    TriggerEvaluation,
    compute_action_fingerprint,
    evaluate_triggers,
)
from novare.reflexion.engine import ReflectionContext, ReflexionEngine
from novare.reflexion.validator import ReflectionValidator

__all__ = [
    "ReflectionDecision",
    "ReflectionRecord",
    "ReflectionTrigger",
    "ReflexionState",
    "make_reflection_record",
    "ProgressTracker",
    "compute_progress_fingerprint",
    "ToolEventSummary",
    "TriggerEvaluation",
    "compute_action_fingerprint",
    "evaluate_triggers",
    "ReflectionContext",
    "ReflexionEngine",
    "ReflectionValidator",
]
