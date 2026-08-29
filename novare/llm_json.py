"""novare/llm_json.py — LLM JSON 输出的统一解析 + 校验 + 错误反馈重试

收敛全项目三套重复实现：
- context_compactor._parse_json_object / _summarize_with_llm 重试循环
- hallucination_verifier._parse_json_object / _call_json 重试循环
- reflexion.validator.parse_model_json（容忍模式）

两种使用方式：
1. parse_json_object(text)                — 严格解析（恰好一个 JSON 对象）
   parse_json_object(text, tolerant=True) — 容忍模式（失败返回 None，调用方自决）
2. await call_llm_json(...)               — 调用 LLMClient.chat() 要求 JSON 输出，
   解析/校验失败时把错误反馈给模型重试（默认 2 次尝试）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable

logger = logging.getLogger("novare.llm_json")

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_json_object(text: str, *, tolerant: bool = False) -> dict | None:
    """解析 LLM 输出中的单个 JSON 对象。

    严格模式（默认）：剥离 Markdown 围栏后必须恰好包含一个 JSON 对象，
    否则抛 ValueError（供错误反馈重试使用）。
    容忍模式：解析失败时尝试提取首个 {...} 块，仍失败返回 None。
    """
    if not text:
        if tolerant:
            return None
        raise ValueError("empty LLM response")

    stripped = text.strip()
    fence = _FENCE_RE.fullmatch(stripped)
    if fence:
        stripped = fence.group(1).strip()

    try:
        value, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        value, end = None, -1

    if value is not None and isinstance(value, dict) and not stripped[end:].strip():
        return value

    if tolerant:
        # 提取首个 { ... } 块再试（推理模型常见的围栏外文本）
        start, rfind = stripped.find("{"), stripped.rfind("}")
        if start != -1 and rfind > start:
            try:
                value2 = json.loads(stripped[start : rfind + 1])
            except json.JSONDecodeError:
                return None
            if isinstance(value2, dict):
                return value2
        return None

    raise ValueError("LLM response must contain exactly one JSON object")


def parse_model_json(raw: str) -> dict | None:
    """reflexion 兼容入口：容忍模式解析，失败返回 None（由调用方决定修复）。"""
    return parse_json_object(raw, tolerant=True)


async def call_llm_json(
    llm_client,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    validate: Callable[[dict], None] | None = None,
    convert: Callable[[dict], Any] | None = None,
    timeout: float = 30.0,
    max_attempts: int = 2,
) -> tuple[Any, int]:
    """调用 LLM 要求 JSON 输出，解析/校验失败时带错误反馈重试。

    validate: 校验函数，只断言不转换（失败抛异常触发重试）。
    convert:  可选转换函数，校验通过后调用，返回值作为结果透出
              （如 compactor 的摘要提取）。
    返回: (result, attempts)。convert 缺省时 result 为原始 payload。
    全部尝试耗尽后抛最后一次的异常（调用方负责降级）。
    """
    last_error = ""
    result: Any = None

    for attempt in range(1, max_attempts + 1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if last_error:
            messages.append({
                "role": "user",
                "content": "Previous output was rejected: " + last_error
                           + ". Return corrected JSON only.",
            })
        try:
            response = await asyncio.wait_for(
                llm_client.chat(messages, tools=None, max_tokens=max_tokens),
                timeout=timeout,
            )
            content = str(response.content or "").strip()
            if not content:
                # 推理模型把输出预算耗在思考上时常见；给出可行动的诊断
                finish_reason = getattr(response, "stop_reason", "unknown")
                reasoning_chars = len(str(getattr(response, "reasoning_content", "") or ""))
                raise ValueError(
                    "Model returned no final JSON content "
                    f"(finish_reason={finish_reason!r}, reasoning_chars={reasoning_chars}). "
                    "Use a non-reasoning model, or increase the output budget."
                )
            payload = parse_json_object(content)
            result = payload
            if validate is not None:
                validate(payload)
            if convert is not None:
                result = convert(payload)
            return result, attempt
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_error = str(exc)[:400]
            if attempt >= max_attempts:
                raise
            logger.info("LLM JSON call rejected (attempt %d/%d): %s",
                        attempt, max_attempts, last_error)

    raise RuntimeError("unreachable")
