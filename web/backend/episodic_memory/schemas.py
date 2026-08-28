"""情景记忆 Pydantic 请求/响应模型。"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator


# 允许的 memory_type 值
ALLOWED_MEMORY_TYPES = frozenset({
    "research_decision",
    "task_outcome",
    "experiment_result",
    "failure_lesson",
    "continuation_context",
})


def _validate_score_01(v: float, name: str) -> float:
    """验证值为有限数字且在 [0, 1] 范围内。"""
    if not isinstance(v, (int, float)):
        raise ValueError(f"{name} must be a number")
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"{name} must be a finite number (not NaN or Inf)")
    if f < 0.0 or f > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {f}")
    return f


class EpisodicMemoryExtract(BaseModel):
    """LLM 提取的单条情景记忆。"""
    should_store: bool = True
    memory_type: str = "research_decision"
    summary: str = ""
    context: str = ""
    action: str = ""
    outcome: str = ""
    topics: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("importance")
    @classmethod
    def _validate_importance(cls, v: float) -> float:
        return _validate_score_01(v, "importance")

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, v: float) -> float:
        return _validate_score_01(v, "confidence")


class EpisodicMemoryExtractResult(BaseModel):
    """LLM 提取结果的顶层结构。"""
    memories: list[EpisodicMemoryExtract] = Field(default_factory=list)


class EpisodicMemoryOut(BaseModel):
    """情景记忆 API 响应模型。"""
    id: str
    memory_type: str
    summary: str
    context: str = ""
    action: str = ""
    outcome: str = ""
    topics: list[str] = Field(default_factory=list)
    importance: float = 0.5
    confidence: float = 0.5
    status: str = "active"
    pinned: bool = False
    session_id: str | None = None
    occurred_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    last_retrieved_at: str | None = None
    retrieval_count: int = 0
    index_status: str = "pending"
