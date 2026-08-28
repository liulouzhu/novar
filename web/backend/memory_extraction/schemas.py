"""统一记忆提取的 Pydantic Schema 定义。"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator

from web.backend.episodic_memory.schemas import EpisodicMemoryExtract


def _finite_non_negative(v: float) -> float:
    """验证值为有限数字且在 [0, 1] 范围内。"""
    if not isinstance(v, (int, float)):
        raise ValueError("must be a number")
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        raise ValueError("must be a finite number (not NaN or Inf)")
    if f < 0.0 or f > 1.0:
        raise ValueError(f"must be between 0.0 and 1.0, got {f}")
    return f


class ProfileMemoryCandidate(BaseModel):
    """用户画像提取候选。"""

    category: str = "research_preference"
    key: str = ""
    value: str = ""
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, v: float) -> float:
        return _finite_non_negative(v)


class UnifiedMemoryExtractionResult(BaseModel):
    """统一 LLM 提取的结构化结果。"""

    schema_version: int = 1
    profile_updates: list[ProfileMemoryCandidate] = Field(default_factory=list)
    episodes: list[EpisodicMemoryExtract] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _check_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Unsupported schema_version: {v}")
        return v
