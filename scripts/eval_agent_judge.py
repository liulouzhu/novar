"""Agent 执行过程 LLM-as-Judge 评分脚本

使用独立的评审模型对 Agent 的执行过程进行多维度评分。

评估维度：
1. 任务理解 (task_understanding) — Agent 是否正确理解了用户意图
2. 工具选择 (tool_selection) — 是否选择了合适的工具
3. 执行效率 (execution_efficiency) — 工具调用是否冗余、顺序是否合理
4. 错误处理 (error_handling) — 遇到错误时的恢复能力
5. 结果质量 (result_quality) — 最终输出是否满足用户需求
6. 综合评分 (overall) — 加权综合分

用法：
    # 评估单个 session
    python scripts/eval_agent_judge.py \\
        --session workspace/xxx/.novare/sessions/1781186968-9aa1fe4b.jsonl \\
        --judge-model qwen-plus

    # 批量评估目录下所有 session
    python scripts/eval_agent_judge.py \\
        --session-dir workspace/xxx/.novare/sessions/ \\
        --judge-model qwen-plus \\
        -o reports/agent_eval.json

    # 从 PostgreSQL 读取 session
    python scripts/eval_agent_judge.py \\
        --session-id 1781186968-9aa1fe4b \\
        --user-id YOUR_UUID \\
        --judge-model qwen-plus

环境变量：
    DASHSCOPE_API_KEY   — 百炼 API key（评审模型）
    JUDGE_MODEL         — 默认评审模型（覆盖 --judge-model）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("eval_agent_judge")


# ── 数据结构 ──────────────────────────────────────────────────────────────

@dataclass
class ToolCallRecord:
    """单次工具调用记录"""
    name: str
    arguments: dict
    result_preview: str = ""
    is_error: bool = False
    duration_sec: float = 0


@dataclass
class AgentExecution:
    """Agent 执行过程的结构化表示"""
    session_id: str
    user_query: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    final_response: str = ""
    total_tool_calls: int = 0
    failed_tool_calls: int = 0
    tool_distribution: dict = field(default_factory=dict)


@dataclass
class JudgeScore:
    """评审维度评分"""
    dimension: str
    score: float  # 1-10
    reasoning: str
    suggestions: list[str] = field(default_factory=list)


@dataclass
class JudgeResult:
    """完整评审结果"""
    session_id: str
    scores: list[JudgeScore] = field(default_factory=list)
    overall_score: float = 0
    summary: str = ""
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    raw_response: str = ""


# ── JSONL 解析 ──────────────────────────────────────────────────────────────

def parse_session_jsonl(jsonl_path: str) -> AgentExecution:
    """从 JSONL 文件解析 Agent 执行过程"""
    messages = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                messages.append(json.loads(line))

    return _build_execution(messages, session_id=Path(jsonl_path).stem)


def parse_session_messages(messages: list[dict], session_id: str = "") -> AgentExecution:
    """从消息列表解析 Agent 执行过程"""
    return _build_execution(messages, session_id=session_id)


def _build_execution(messages: list[dict], session_id: str = "") -> AgentExecution:
    """从消息列表构建 AgentExecution"""
    # 提取用户查询
    user_query = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    # 提取工具调用
    tool_calls = []
    tool_dist = {}
    failed = 0

    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name", "unknown")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                tool_dist[name] = tool_dist.get(name, 0) + 1

                # 查找对应的 tool result
                result_content = ""
                is_error = False
                for r_msg in messages:
                    if (r_msg.get("role") == "tool"
                            and r_msg.get("tool_call_id") == tc.get("id")):
                        result_content = r_msg.get("content", "")[:500]
                        # 检测错误
                        try:
                            result_json = json.loads(r_msg.get("content", "{}"))
                            is_error = not result_json.get("ok", True)
                        except (json.JSONDecodeError, ValueError):
                            is_error = "error" in result_content.lower()[:200]
                        break

                if is_error:
                    failed += 1

                tool_calls.append(ToolCallRecord(
                    name=name,
                    arguments=args,
                    result_preview=result_content[:300],
                    is_error=is_error,
                ))

    # 提取最终回复
    final_response = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            final_response = msg["content"]
            break

    return AgentExecution(
        session_id=session_id,
        user_query=user_query,
        tool_calls=tool_calls,
        final_response=final_response,
        total_tool_calls=len(tool_calls),
        failed_tool_calls=failed,
        tool_distribution=tool_dist,
    )


# ── 评审 Prompt ─────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """你是一个专业的 AI Agent 评审专家。你的任务是评估 Agent 的执行过程质量。

请从以下 6 个维度进行评分（每项 1-10 分）：

1. **任务理解** (task_understanding)：Agent 是否正确理解了用户意图
2. **工具选择** (tool_selection)：是否选择了合适的工具，参数是否正确
3. **执行效率** (execution_efficiency)：工具调用是否冗余，顺序是否合理
4. **错误处理** (error_handling)：遇到错误时的恢复能力（重试、切换方案等）
5. **结果质量** (result_quality)：最终输出是否满足用户需求，内容是否准确
6. **综合评分** (overall)：整体表现的加权评分

评分标准：
- 9-10: 优秀，完美执行
- 7-8: 良好，有小瑕疵
- 5-6: 一般，有明显问题
- 3-4: 较差，存在重大缺陷
- 1-2: 很差，完全失败

请以 JSON 格式输出，包含每个维度的分数、理由和改进建议。"""


JUDGE_USER_PROMPT = """## 用户请求
{user_query}

## Agent 执行过程
{execution_trace}

## 工具调用统计
- 总调用次数: {total_calls}
- 失败次数: {failed_calls}
- 工具分布: {tool_dist}

## 最终回复
{final_response}

---

请对上述 Agent 执行过程进行评审。输出严格 JSON 格式：

```json
{{
  "scores": [
    {{
      "dimension": "task_understanding",
      "score": 8,
      "reasoning": "Agent 正确理解了用户想要搜索和解析论文的意图",
      "suggestions": ["可以更精确地理解用户对论文数量的要求"]
    }},
    {{
      "dimension": "tool_selection",
      "score": 7,
      "reasoning": "选择了正确的工具",
      "suggestions": ["..."]
    }},
    ...
  ],
  "overall_score": 7.5,
  "summary": "一句话总结",
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["缺点1", "缺点2"]
}}
```

只输出 JSON，不要其他文字。"""


# ── 评审执行 ──────────────────────────────────────────────────────────────

def _format_execution_trace(execution: AgentExecution, max_calls: int = 30) -> str:
    """格式化执行过程为可读文本"""
    lines = []
    for i, tc in enumerate(execution.tool_calls[:max_calls]):
        status = "❌" if tc.is_error else "✅"
        args_str = json.dumps(tc.arguments, ensure_ascii=False)[:200]
        lines.append(f"{i+1}. {status} {tc.name}({args_str})")
        if tc.result_preview:
            lines.append(f"   结果: {tc.result_preview[:150]}...")

    if len(execution.tool_calls) > max_calls:
        lines.append(f"... 还有 {len(execution.tool_calls) - max_calls} 次调用")

    return "\n".join(lines)


async def judge_session(
    execution: AgentExecution,
    judge_model: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> JudgeResult:
    """使用 LLM 评审 Agent 执行过程"""
    from novare.llm_client import LLMClient

    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
    base_url = base_url or os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

    if not api_key:
        raise ValueError("需要设置 DASHSCOPE_API_KEY 环境变量")

    proxy = os.environ.get("NOVARE_PROXY") or None
    client = LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=judge_model,
        proxy=proxy,
    )

    # 构建 prompt
    trace = _format_execution_trace(execution)
    user_prompt = JUDGE_USER_PROMPT.format(
        user_query=execution.user_query[:1000],
        execution_trace=trace[:3000],
        total_calls=execution.total_tool_calls,
        failed_calls=execution.failed_tool_calls,
        tool_dist=json.dumps(execution.tool_distribution, ensure_ascii=False),
        final_response=execution.final_response[:1000],
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # 调用评审模型
    t0 = time.monotonic()
    response = await client.collect_stream(messages)
    latency = time.monotonic() - t0
    content = response.content or ""
    logger.info("Judge call completed: %.1fs, %d chars", latency, len(content))

    # 解析结果
    result = _parse_judge_response(content, execution.session_id)
    result.raw_response = content
    return result


def _parse_judge_response(content: str, session_id: str) -> JudgeResult:
    """解析评审模型输出"""
    # 提取 JSON
    json_match = re.search(r'\{[\s\S]*\}', content)
    if not json_match:
        logger.warning("Failed to extract JSON from judge response")
        return JudgeResult(
            session_id=session_id,
            overall_score=0,
            summary="评审模型输出解析失败",
            raw_response=content,
        )

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        logger.warning("Failed to parse judge JSON")
        return JudgeResult(
            session_id=session_id,
            overall_score=0,
            summary="评审模型 JSON 解析失败",
            raw_response=content,
        )

    scores = []
    for s in data.get("scores", []):
        scores.append(JudgeScore(
            dimension=s.get("dimension", "unknown"),
            score=min(10, max(1, float(s.get("score", 5)))),
            reasoning=s.get("reasoning", ""),
            suggestions=s.get("suggestions", []),
        ))

    return JudgeResult(
        session_id=session_id,
        scores=scores,
        overall_score=float(data.get("overall_score", 0)),
        summary=data.get("summary", ""),
        strengths=data.get("strengths", []),
        weaknesses=data.get("weaknesses", []),
    )


import re  # noqa: E402


# ── 报告输出 ──────────────────────────────────────────────────────────────

def print_judge_report(result: JudgeResult):
    """打印评审报告"""
    print("\n" + "=" * 70)
    print(f"Agent 执行评审报告 — Session: {result.session_id}")
    print("=" * 70)

    # 各维度评分
    print("\n📊 各维度评分:")
    print(f"{'维度':<25} {'分数':>6}  理由")
    print("-" * 70)
    for s in result.scores:
        bar = "█" * int(s.score) + "░" * (10 - int(s.score))
        print(f"  {s.dimension:<23} {s.score:>5.1f}  {bar}")
        if s.reasoning:
            print(f"  {'':23}       {s.reasoning[:60]}")
        for sug in s.suggestions[:2]:
            print(f"  {'':23}       💡 {sug}")

    # 综合评分
    print(f"\n{'=' * 70}")
    print(f"🏆 综合评分: {result.overall_score:.1f}/10")
    print(f"{'=' * 70}")

    # 总结
    if result.summary:
        print(f"\n📝 总结: {result.summary}")

    # 优缺点
    if result.strengths:
        print("\n✅ 优点:")
        for s in result.strengths:
            print(f"  - {s}")

    if result.weaknesses:
        print("\n⚠️  不足:")
        for w in result.weaknesses:
            print(f"  - {w}")

    print()


def save_judge_results(results: list[JudgeResult], output_path: str):
    """保存评审结果到 JSON"""
    data = []
    for r in results:
        data.append({
            "session_id": r.session_id,
            "overall_score": r.overall_score,
            "summary": r.summary,
            "scores": [
                {
                    "dimension": s.dimension,
                    "score": s.score,
                    "reasoning": s.reasoning,
                    "suggestions": s.suggestions,
                }
                for s in r.scores
            ],
            "strengths": r.strengths,
            "weaknesses": r.weaknesses,
        })

    Path(output_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Results saved to %s", output_path)


# ── 批量评估 ──────────────────────────────────────────────────────────────

async def batch_evaluate(
    session_paths: list[str],
    judge_model: str,
    concurrency: int = 3,
) -> list[JudgeResult]:
    """批量评估多个 session"""
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def _eval_one(path: str):
        async with semaphore:
            logger.info("Evaluating: %s", Path(path).name)
            execution = parse_session_jsonl(path)
            result = await judge_session(execution, judge_model)
            print_judge_report(result)
            return result

    tasks = [_eval_one(p) for p in session_paths]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


# ── 主流程 ────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Agent 执行过程 LLM-as-Judge 评审")
    parser.add_argument("--session", default=None, help="单个 JSONL session 文件路径")
    parser.add_argument("--session-dir", default=None, help="批量评估：session 目录")
    parser.add_argument("--session-id", default=None, help="从 PostgreSQL 读取（需 --user-id）")
    parser.add_argument("--user-id", default=None, help="用户 ID（配合 --session-id）")
    parser.add_argument("--judge-model", default=None,
                        help="评审模型（默认 DASHSCOPE_JUDGE_MODEL 或 qwen-plus）")
    parser.add_argument("--concurrency", type=int, default=3, help="并发数")
    parser.add_argument("--output", "-o", default=None, help="保存评审结果 JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示原始评审输出")
    args = parser.parse_args()

    judge_model = args.judge_model or os.environ.get("DASHSCOPE_JUDGE_MODEL", "qwen-plus")
    logger.info("Judge model: %s", judge_model)

    results = []

    if args.session:
        # 单个文件评估
        execution = parse_session_jsonl(args.session)
        result = await judge_session(execution, judge_model)
        print_judge_report(result)
        if args.verbose:
            print("\n--- Raw Judge Response ---")
            print(result.raw_response)
        results.append(result)

    elif args.session_dir:
        # 批量评估
        session_dir = Path(args.session_dir)
        jsonl_files = sorted(session_dir.glob("*.jsonl"))
        if not jsonl_files:
            logger.error("No .jsonl files found in %s", args.session_dir)
            sys.exit(1)

        logger.info("Found %d sessions to evaluate", len(jsonl_files))
        results = await batch_evaluate(
            [str(f) for f in jsonl_files],
            judge_model,
            concurrency=args.concurrency,
        )

        # 打印汇总
        if len(results) > 1:
            avg_score = sum(r.overall_score for r in results) / len(results)
            print(f"\n{'=' * 70}")
            print(f"📊 批量评估汇总 ({len(results)} sessions)")
            print(f"{'=' * 70}")
            print(f"  平均综合评分: {avg_score:.1f}/10")
            print(f"  最高: {max(r.overall_score for r in results):.1f}")
            print(f"  最低: {min(r.overall_score for r in results):.1f}")

    elif args.session_id:
        # 从 PostgreSQL 读取
        logger.error("PostgreSQL 模式暂未实现，请使用 --session 指定 JSONL 文件")
        sys.exit(1)

    else:
        logger.error("请指定 --session, --session-dir, 或 --session-id")
        sys.exit(1)

    # 保存结果
    if args.output and results:
        save_judge_results(results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
