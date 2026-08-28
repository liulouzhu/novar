"""novare/task_state.py — Turn-scoped 任务状态追踪

为 AgentLoop 的每次 run_turn 提供轻量级任务状态管理。
状态更新靠工具结果的启发式提取（regex + 关键词），不调 LLM。

设计要点：
- TaskState 是可观测提示，不是强制工作流引擎
- TaskStateManager 在 run_turn 内部创建，每次调用独立持有
- Web 多用户并发时互不干扰
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from novare.tool_result import parse_tool_result


@dataclass
class TaskState:
    """一次 run_turn 的任务状态快照"""

    goal: str = ""                          # 用户本轮目标
    completed: list[str] = field(default_factory=list)   # 已完成步骤
    pending: list[str] = field(default_factory=list)      # 待办步骤
    tools_used: list[str] = field(default_factory=list)   # 已调用工具（去重，保持首次出现顺序）
    key_findings: list[str] = field(default_factory=list) # 关键发现（最多 10 条）
    missing_info: list[str] = field(default_factory=list) # 缺失信息

    _MAX_FINDINGS: int = 10

    def to_prompt_block(self) -> str:
        """序列化为注入 system prompt 的文本块"""
        lines = ["[当前任务状态]"]

        if self.goal:
            lines.append(f"目标：{self.goal}")

        if self.completed:
            lines.append("已完成：")
            for step in self.completed:
                lines.append(f"  - {step}")

        if self.pending:
            lines.append("待办：")
            for step in self.pending:
                lines.append(f"  - {step}")

        if self.tools_used:
            lines.append(f"已用工具：{self.tool_summary}")

        if self.key_findings:
            lines.append("关键发现：")
            for f in self.key_findings:
                lines.append(f"  - {f}")

        if self.missing_info:
            lines.append("缺失信息：")
            for info in self.missing_info:
                lines.append(f"  - {info}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """序列化为 WebSocket / JSON 格式"""
        return {
            "goal": self.goal,
            "completed": list(self.completed),
            "pending": list(self.pending),
            "tools_used": list(self.tools_used),
            "key_findings": list(self.key_findings),
            "missing_info": list(self.missing_info),
        }

    @property
    def tool_summary(self) -> str:
        """工具使用统计：paper_search×2, paper_parse×1, ..."""
        counts = Counter(self.tools_used)
        return ", ".join(f"{name}×{cnt}" for name, cnt in counts.items())


class TaskStateManager:
    """管理 TaskState 的生命周期和自动更新。

    每个实例绑定一次 run_turn 调用，不在多个 turn 或多个用户之间共享。
    """

    def __init__(self) -> None:
        self.state: TaskState | None = None

    # ── 生命周期 ─────────────────────────────────────────────

    def init_turn(self, user_input: str) -> None:
        """新轮次开始时初始化状态"""
        goal = _extract_goal(user_input)
        pending = _infer_pending_steps(user_input)
        self.state = TaskState(goal=goal, pending=pending)

    def clear(self) -> None:
        """轮次结束后清空"""
        self.state = None

    # ── 工具结果更新 ─────────────────────────────────────────

    def update_from_tool(self, tool_name: str, arguments: dict, result: str) -> None:
        """根据工具调用结果更新状态 — 纯启发式，无 LLM 调用"""
        if self.state is None:
            return

        state = self.state

        # 记录工具使用（去重，保持首次顺序）
        if tool_name not in state.tools_used:
            state.tools_used.append(tool_name)

        # 按工具类型提取信息
        if tool_name == "paper_search":
            _handle_paper_search(state, arguments, result)
        elif tool_name == "paper_parse":
            _handle_paper_parse(state, arguments, result)
        elif tool_name == "rag_query":
            _handle_rag_query(state, arguments, result)
        elif tool_name == "knowledge_graph":
            _handle_knowledge_graph(state, arguments, result)
        elif tool_name == "code_execute":
            _handle_code_execute(state, arguments, result)
        elif tool_name == "reviewer_evaluate":
            _handle_reviewer_evaluate(state, arguments, result)
        elif tool_name == "innovation_search":
            _handle_innovation_search(state, arguments, result)

    def record_completion(self, step: str) -> None:
        """记录已完成步骤（自动去重）"""
        if self.state is None:
            return
        if step not in self.state.completed:
            self.state.completed.append(step)
        # 从 pending 中移除匹配项
        _remove_from_pending(self.state, step)

    def record_finding(self, finding: str) -> None:
        """记录关键发现（超过上限时 FIFO 淘汰）"""
        if self.state is None:
            return
        if finding and finding not in self.state.key_findings:
            self.state.key_findings.append(finding)
            # FIFO 淘汰
            if len(self.state.key_findings) > self.state._MAX_FINDINGS:
                self.state.key_findings.pop(0)


# ── 目标与待办推断 ──────────────────────────────────────────

def _extract_goal(user_input: str) -> str:
    """从用户输入提取目标（截断到合理长度）"""
    text = user_input.strip().replace("\n", " ")
    if len(text) > 200:
        text = text[:197] + "..."
    return text


def _infer_pending_steps(user_input: str) -> list[str]:
    """根据用户输入关键词启发式推断待办步骤"""
    text = user_input.lower()
    pending: list[str] = []

    # 搜索相关
    if any(kw in text for kw in ("搜索", "查找", "找", "search", "find", "look")):
        pending.append("搜索相关论文")

    # 解析相关
    if any(kw in text for kw in ("解析", "阅读", "读", "parse", "read", "pdf")):
        pending.append("解析目标论文")

    # RAG / 语义检索
    if any(kw in text for kw in ("检索", "问答", "rag", "query", "question")):
        pending.append("在已解析论文中检索")

    # 分析 / 对比
    if any(kw in text for kw in ("分析", "对比", "比较", "analyze", "compare", "contrast")):
        pending.append("深度分析和对比")

    # 知识图谱
    if any(kw in text for kw in ("图谱", "关系", "知识", "graph", "knowledge")):
        pending.append("构建知识图谱")

    # 代码 / 数据
    if any(kw in text for kw in ("代码", "数据", "统计", "可视化", "code", "data", "plot")):
        pending.append("数据分析和可视化")

    # 综述 / 总结
    if any(kw in text for kw in ("综述", "总结", "survey", "summary", "review")):
        pending.append("综合撰写回答")

    # 至少有一个待办
    if not pending:
        pending.append("回答用户问题")

    return pending


# ── 工具结果提取 ────────────────────────────────────────────

def _handle_paper_search(state: TaskState, arguments: dict, result: str) -> None:
    """处理 paper_search 结果 — JSON 优先，regex 降级"""
    parsed = parse_tool_result(result)
    if parsed.is_json and parsed.ok:
        data = parsed.data or {}
        total = data.get("total", 0)
        query = data.get("query", arguments.get("query", ""))
        _add_completion(state, f"搜索 '{query}' 找到 {total} 篇论文")
        for paper in data.get("papers", [])[:5]:
            title = paper.get("title", "")
            if title:
                _add_finding(state, title)
        # 从 sources 提取引用来源
        for src in parsed.sources[:3]:
            title = src.get("title", "")
            if title and title not in state.key_findings:
                _add_finding(state, title)
    else:
        # 降级：旧 regex 逻辑
        _handle_paper_search_legacy(state, arguments, result)


def _handle_paper_parse(state: TaskState, arguments: dict, result: str) -> None:
    """处理 paper_parse 结果 — JSON 优先，regex 降级"""
    parsed = parse_tool_result(result)
    paper_id = arguments.get("paper_id", arguments.get("pdf_url", ""))

    if parsed.is_json:
        if parsed.ok:
            data = parsed.data or {}
            pid = data.get("paper_id", paper_id)
            _add_completion(state, f"解析论文 '{pid}'")
            # 从 sources 提取标题
            for src in parsed.sources[:2]:
                title = src.get("title", "")
                if title:
                    _add_finding(state, f"已解析：{title}")
        else:
            state.missing_info.append(f"解析 {paper_id} 失败: {parsed.error or ''}")
    else:
        # 降级
        if "错误" in result or "Error" in result:
            state.missing_info.append(f"解析 {paper_id} 失败")
        else:
            _add_completion(state, f"解析论文 '{paper_id}'")
            title_match = re.search(r'(?:Title|标题)[：:]\s*(.+?)(?:\n|$)', result)
            if title_match:
                title = title_match.group(1).strip()[:120]
                if title:
                    _add_finding(state, f"已解析：{title}")


def _handle_rag_query(state: TaskState, arguments: dict, result: str) -> None:
    """处理 rag_query 结果 — JSON 优先，regex 降级"""
    parsed = parse_tool_result(result)
    question = arguments.get("question", "")
    truncated_q = question[:60] + ("..." if len(question) > 60 else "")

    if parsed.is_json and parsed.ok:
        data = parsed.data or {}
        n_results = len(data.get("results", []))
        _add_completion(state, f"语义检索 '{truncated_q}' ({n_results} 条结果)")
        # 从 sources 提取引用
        for src in parsed.sources[:3]:
            title = src.get("title", "")
            section = src.get("section", "")
            if title:
                _add_finding(state, f"RAG: {title} [{section}]")
    else:
        _add_completion(state, f"语义检索 '{truncated_q}'")


def _handle_knowledge_graph(state: TaskState, arguments: dict, result: str) -> None:
    """处理 knowledge_graph 结果 — JSON 优先，regex 降级"""
    parsed = parse_tool_result(result)
    action = arguments.get("action", "query")

    if parsed.is_json and parsed.ok:
        data = parsed.data or {}
        _add_completion(state, f"知识图谱: {action}")
        # 从 data 提取关系信息
        for rel in data.get("relations", [])[:3]:
            s = rel.get("subject", "")
            p = rel.get("predicate", "")
            o = rel.get("object", "")
            if s and p and o:
                _add_finding(state, f"{s} --[{p}]--> {o}")
    else:
        _add_completion(state, f"知识图谱操作: {action}")


def _handle_code_execute(state: TaskState, arguments: dict, result: str) -> None:
    """处理 code_execute 结果 — JSON 优先，regex 降级"""
    parsed = parse_tool_result(result)
    if parsed.is_json:
        if parsed.ok:
            _add_completion(state, "执行代码分析")
        else:
            state.missing_info.append(f"代码执行失败: {parsed.error or ''}")
    else:
        _add_completion(state, "执行代码分析")


def _handle_reviewer_evaluate(state: TaskState, arguments: dict, result: str) -> None:
    """处理 reviewer_evaluate 结果"""
    _add_completion(state, "完成评审")


def _handle_innovation_search(state: TaskState, arguments: dict, result: str) -> None:
    """处理 innovation_search 结果 — JSON 优先，regex 降级"""
    parsed = parse_tool_result(result)
    action = arguments.get("action", "landscape")

    if parsed.is_json and parsed.ok:
        data = parsed.data or {}
        total = data.get("total_papers", 0)
        _add_completion(state, f"创新搜索: {action} ({total} 篇)")
        for paper in data.get("papers", [])[:3]:
            title = paper.get("title", "")
            if title:
                _add_finding(state, title)
    else:
        _add_completion(state, f"创新搜索: {action}")


# ── 辅助函数 ────────────────────────────────────────────────

def _add_finding(state: TaskState, finding: str) -> None:
    """添加关键发现（去重 + FIFO 淘汰）"""
    if finding and finding not in state.key_findings:
        state.key_findings.append(finding)
        if len(state.key_findings) > state._MAX_FINDINGS:
            state.key_findings.pop(0)


def _add_completion(state: TaskState, step: str) -> None:
    """添加已完成步骤（去重）"""
    if step not in state.completed:
        state.completed.append(step)
    _remove_from_pending(state, step)


def _remove_from_pending(state: TaskState, completed_step: str) -> None:
    """根据已完成步骤，从 pending 中移除语义匹配的条目"""
    # 关键词映射：已完成的步骤类型 → pending 中对应的关键词
    removal_hints = {
        "搜索": ("搜索", "search", "查找"),
        "解析": ("解析", "parse", "阅读"),
        "检索": ("检索", "rag", "query"),
        "分析": ("分析", "analyze", "对比"),
        "图谱": ("图谱", "graph", "knowledge"),
        "代码": ("代码", "code", "数据"),
        "评审": ("评审", "review"),
        "创新搜索": ("创新", "innovation"),
    }

    lower_step = completed_step.lower()
    for _, keywords in removal_hints.items():
        if any(kw in lower_step for kw in keywords):
            # 找到匹配的 pending 条目并移除
            for pending in list(state.pending):
                if any(kw in pending.lower() for kw in keywords):
                    state.pending.remove(pending)
                    break
            break


# ── Legacy regex 降级 ────────────────────────────────────────

def _handle_paper_search_legacy(state: TaskState, arguments: dict, result: str) -> None:
    """paper_search 的旧 regex 提取逻辑（JSON parse 失败时的兜底）"""
    query = arguments.get("query", "")
    count_match = re.search(r"找到\s*(\d+)\s*篇|(\d+)\s*results?|共\s*(\d+)", result, re.IGNORECASE)
    if count_match:
        n = next(g for g in count_match.groups() if g)
        step = f"搜索 '{query}' 找到 {n} 篇论文"
    else:
        step = f"搜索 '{query}'"
    _add_completion(state, step)

    for match in re.findall(r'(?:Title|标题)[：:]\s*(.+?)(?:\n|$)', result):
        title = match.strip()[:120]
        if title:
            _add_finding(state, title)
