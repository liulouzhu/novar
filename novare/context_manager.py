"""novare/context_manager.py — 上下文 token 估算与规则摘要

借鉴 Claw Code 的设计：
- 启发式 token 估算（chars/4）
- API usage 追踪（累积 input_tokens）
- 无 LLM 的规则摘要 fallback

主运行路径的完整轮次、token 预算和 LLM 混合压缩位于
novare.context_compactor。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Token 估算 ────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """启发式 token 估算（借鉴 Claw Code 的 chars/4 策略）

    英文约 1 token / 4 chars，中文约 1 token / 1.5 chars。
    混合文本取加权估算。
    """
    if not text:
        return 0
    # 统计中文字符数
    cjk_chars = len(re.findall(r'[一-鿿　-〿＀-￯]', text))
    total_chars = len(text)
    other_chars = total_chars - cjk_chars
    # 中文 ~1.5 token/char，英文 ~0.25 token/char
    return max(1, int(cjk_chars * 0.7 + other_chars * 0.25))


def estimate_message_tokens(message: dict) -> int:
    """估算单条消息的 token 数"""
    total = 0
    content = message.get("content", "")
    if isinstance(content, str):
        total += estimate_tokens(content)

    # tool_calls（assistant 消息中）
    for tc in (message.get("tool_calls") or []):
        func = tc.get("function", {})
        total += estimate_tokens(func.get("name", ""))
        total += estimate_tokens(func.get("arguments", ""))

    # role 和 tool_call_id 的开销
    total += 10  # 固定开销估算
    return total


def estimate_messages_tokens(messages: list[dict]) -> int:
    """估算消息列表的总 token 数"""
    return sum(estimate_message_tokens(m) for m in messages)


def estimate_tools_tokens(tools: list[dict]) -> int:
    """估算工具定义（OpenAI function schema）的 token 开销

    将每个 tool 的 JSON 序列化后做启发式估算。
    """
    import json
    total = 0
    for t in tools:
        total += estimate_tokens(json.dumps(t, ensure_ascii=False))
    return total


# ── Usage 追踪 ────────────────────────────────────────────────

@dataclass
class TokenUsage:
    """单次 API 调用的 token 用量"""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class UsageTracker:
    """累积 token 用量追踪（用于触发自动压缩）"""
    cumulative_input: int = 0
    cumulative_output: int = 0
    turn_count: int = 0

    def add(self, usage: TokenUsage):
        self.cumulative_input += usage.input_tokens
        self.cumulative_output += usage.output_tokens
        self.turn_count += 1

    def should_compact(self, threshold: int) -> bool:
        """累积 input tokens 是否超过阈值"""
        if threshold <= 0:
            return False
        return self.cumulative_input >= threshold

    def reset_after_compact(self):
        """压缩后重置计数器（避免重复触发）"""
        self.cumulative_input = 0
        self.turn_count = 0

    def summary(self) -> str:
        return (f"cumulative_input={self.cumulative_input}, "
                f"cumulative_output={self.cumulative_output}, "
                f"turns={self.turn_count}")


# ── 消息压缩 ──────────────────────────────────────────────────

def _extract_user_requests(messages: list[dict], max_items: int = 3) -> list[str]:
    """提取最近的用户请求（截断 160 字符）"""
    requests = []
    for msg in messages:
        if msg.get("role") == "user":
            content = str(msg.get("content", "")).strip()
            if content:
                truncated = content[:160] + ("..." if len(content) > 160 else "")
                requests.append(truncated)
    return requests[-max_items:]


def _extract_tool_names(messages: list[dict]) -> list[str]:
    """提取使用过的工具名（去重）"""
    names = set()
    for msg in messages:
        # 从 assistant 的 tool_calls 中提取
        for tc in (msg.get("tool_calls") or []):
            func_name = tc.get("function", {}).get("name", "")
            if func_name:
                names.add(func_name)
        # 从 tool 消息中提取
        if msg.get("role") == "tool":
            # tool_call_id 不直接包含工具名，从前面的 assistant 消息关联
            pass
    return sorted(names)


def _extract_key_findings(messages: list[dict], max_chars: int = 300) -> str:
    """从最后一条非空 assistant 消息中提取关键发现"""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = str(msg.get("content", "")).strip()
            if content and len(content) > 20:
                return content[:max_chars] + ("..." if len(content) > max_chars else "")
    return ""


def _extract_paper_titles(messages: list[dict], max_items: int = 5) -> list[str]:
    """从 tool results 中提取论文标题"""
    titles = []
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = str(msg.get("content", ""))
        # 匹配常见论文标题模式
        for match in re.findall(r'(?:Title|标题)[：:]\s*(.+?)(?:\n|$)', content):
            title = match.strip()[:100]
            if title and title not in titles:
                titles.append(title)
                if len(titles) >= max_items:
                    return titles
    return titles


def generate_summary(removed_messages: list[dict]) -> str:
    """从被移除的消息中生成结构化摘要（纯文本提取，不调 LLM）

    借鉴 Claw Code 的 summarize_messages() 策略。
    """
    user_count = sum(1 for m in removed_messages if m.get("role") == "user")
    assistant_count = sum(1 for m in removed_messages if m.get("role") == "assistant")
    tool_count = sum(1 for m in removed_messages if m.get("role") == "tool")

    lines = [
        "[历史摘要 - 原始 {} 条消息 (用户={}, 助手={}, 工具={})]".format(
            len(removed_messages), user_count, assistant_count, tool_count
        ),
    ]

    # 工具列表
    tool_names = _extract_tool_names(removed_messages)
    if tool_names:
        lines.append("使用工具：{}".format(", ".join(tool_names)))

    # 最近用户请求
    user_requests = _extract_user_requests(removed_messages)
    if user_requests:
        lines.append("用户讨论了：")
        for req in user_requests:
            lines.append("  - {}".format(req))

    # 论文标题
    paper_titles = _extract_paper_titles(removed_messages)
    if paper_titles:
        lines.append("涉及论文：{}".format(", ".join(paper_titles)))

    # 关键发现
    findings = _extract_key_findings(removed_messages)
    if findings:
        lines.append("关键发现：{}".format(findings))

    return "\n".join(lines)


def merge_compaction_summaries(existing_summary: str, new_timeline: str) -> str:
    """合并已有的压缩摘要与新产生的 timeline（避免嵌套膨胀）

    借鉴 Claw Code 的 merge_compact_summaries() 策略：
    不嵌套，而是平铺合并。
    """
    # 提取已有摘要中 "- 之前的压缩上下文:" 之前的部分
    # 保留已有的结构化信息（工具列表、用户请求等）
    lines = []

    # 从已有摘要中提取关键信息（跳过头部标记行）
    for line in existing_summary.split("\n"):
        stripped = line.strip()
        if stripped.startswith("[历史摘要"):
            continue
        if stripped:
            lines.append(stripped)

    # 添加新压缩的上下文
    if lines:
        lines.append("")
        lines.append("新增压缩上下文：")
    for line in new_timeline.split("\n"):
        stripped = line.strip()
        if stripped.startswith("[历史摘要"):
            continue
        if stripped:
            lines.append("  {}".format(stripped))

    header = "[历史摘要 - 包含多次压缩的上下文记录]"
    return header + "\n" + "\n".join(lines)
