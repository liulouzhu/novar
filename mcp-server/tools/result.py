"""mcp-server/tools/result.py — 统一工具结果 helper

所有 MCP 工具 handler 返回 JSON 字符串，格式统一为：
{
  "schema_version": 1,
  "tool": "<tool_name>",
  "ok": true/false,
  "summary": "人类可读的一行摘要",
  "data": { ... },
  "sources": [ ... ],      // 证据/引用来源（对象数组）
  "providers": [ ... ],    // 数据提供方（字符串数组）
  "warnings": [ ... ],
  "error": null / "..."
}

使用 ok() / fail() 工厂函数构建，自动填 schema_version 和 tool。
"""

from __future__ import annotations

import json
from typing import Any

SCHEMA_VERSION = 1

# ── 大小限制常量 ─────────────────────────────────────────────

MAX_ABSTRACT = 1000        # paper abstract 截断
MAX_CHUNK_TEXT = 800       # rag chunk text 截断
MAX_SECTION_PREVIEW = 200  # 每条 section preview
MAX_SECTIONS = 10          # sections_preview 最多条数
MAX_REFS = 10              # references_preview 最多条数
MAX_REF_LEN = 150          # 每条 reference 截断
MAX_STDOUT = 4000          # code_execute stdout 截断
MAX_STDERR = 2000          # code_execute stderr 截断


# ── 工厂函数 ─────────────────────────────────────────────────

def ok(
    tool: str,
    data: Any,
    *,
    summary: str,
    sources: list | None = None,
    providers: list[str] | None = None,
    warnings: list[str] | None = None,
) -> str:
    """成功结果 → JSON 字符串。自动填 schema_version 和 tool。"""
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "tool": tool,
            "ok": True,
            "summary": summary,
            "data": data,
            "sources": sources or [],
            "providers": providers or [],
            "warnings": warnings or [],
            "error": None,
        },
        ensure_ascii=False,
    )


def fail(
    tool: str,
    error: str,
    *,
    data: Any = None,
) -> str:
    """失败结果 → JSON 字符串。自动填 schema_version 和 tool。"""
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "tool": tool,
            "ok": False,
            "summary": error,
            "data": data,
            "sources": [],
            "providers": [],
            "warnings": [],
            "error": error,
        },
        ensure_ascii=False,
    )


# ── 大小截断辅助 ─────────────────────────────────────────────

def truncate(text: str, max_len: int) -> str:
    """截断文本到 max_len 字符，超出部分用 … 替代"""
    if not text or len(text) <= max_len:
        return text or ""
    return text[:max_len] + "…"


def truncate_pair(stdout: str, stderr: str) -> tuple[str, str]:
    """截断 stdout / stderr 到各自上限"""
    return truncate(stdout, MAX_STDOUT), truncate(stderr, MAX_STDERR)
