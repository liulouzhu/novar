"""novare/tools/reviewer_evaluate.py -- 评审模型对抗评估工具

使用独立的评审模型对候选创新点做交叉验证，发现单一模型的盲区。
需要配置 NOVARE_REVIEWER_API_KEY 等环境变量。"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from novare.llm_client import LLMClient

logger = logging.getLogger("novare.tools.reviewer")

REVIEWER_SYSTEM_PROMPT = """你是一个独立的研究评审专家，负责对候选创新点做交叉验证评估。

你的角色是"外部评审人"，与生成候选的执行者是不同视角。请保持客观、严格。

评审维度：
1. **评估 (assessment)**：一句话判断该候选的潜力
2. **弱点 (weakness)**：指出一个具体的、非模糊的弱点
3. **建议 (suggestion)**：一个具体的改进建议
4. **独立推荐 (recommendation)**：PROCEED / REVISE / DROP
5. **与执行者评分的一致性 (agreement)**：concordant / discordant，以及你的理由

输出 JSON 格式。"""


async def handle_reviewer_evaluate(args: dict, **kwargs) -> str:
    """处理 reviewer_evaluate 工具调用

    参数：
        candidates: 候选创新点列表（JSON 数组），每项包含 title/problem/idea 等字段
        topic: 研究主题
        stage: 评审阶段 ("candidates" | "review")
        executor_review: 执行者的评审结果（stage="review" 时提供）
    """
    # registry.execute 用 kwargs.update(tool_context) 解包，reviewer_llm 在顶层；
    # 兼容旧格式 kwargs["tool_context"]["reviewer_llm"]
    reviewer_llm: LLMClient | None = kwargs.get("reviewer_llm")
    if reviewer_llm is None:
        reviewer_llm = kwargs.get("tool_context", {}).get("reviewer_llm")
    if not reviewer_llm:
        return json.dumps({
            "error": "评审模型未配置。请设置环境变量 NOVARE_REVIEWER_API_KEY、NOVARE_REVIEWER_BASE_URL、NOVARE_REVIEWER_MODEL。",
            "hint": "示例：\nNOVARE_REVIEWER_API_KEY=sk-xxx\nNOVARE_REVIEWER_BASE_URL=https://api.openai.com/v1\nNOVARE_REVIEWER_MODEL=gpt-4o"
        }, ensure_ascii=False)

    topic = args.get("topic", "")
    stage = args.get("stage", "candidates")
    candidates = args.get("candidates", [])
    executor_review = args.get("executor_review", "")

    if not candidates:
        return "Error: candidates 不能为空"

    # 根据阶段构建不同的 prompt
    if stage == "candidates":
        candidates_json = json.dumps(candidates[:10], ensure_ascii=False, indent=2)
        prompt = f"""## 研究主题
{topic}

## 执行者生成的候选创新点
{candidates_json}

## 你的任务
对每个候选，提供：
1. 一句话评估该候选的潜力
2. 一个具体的弱点（直接指出，不要含糊）
3. 一个改进建议
4. 独立推荐：PROCEED / REVISE / DROP

输出 JSON 格式：
{{"reviews": [{{"candidate_title": "...", "assessment": "...", "weakness": "...", "suggestion": "...", "recommendation": "PROCEED|REVISE|DROP"}}]}}

只输出 JSON，不要其他文字。"""

    elif stage == "review":
        candidates_json = json.dumps(candidates[:10], ensure_ascii=False, indent=2)
        prompt = f"""## 研究主题
{topic}

## 执行者的评审结果摘要
{executor_review}

## 候选创新点（供参考）
{candidates_json}

## 你的任务
对每个候选，给出你的独立评估：
1. 你是否同意执行者的评分？标记 concordant（一致）或 discordant（分歧）
2. 你的独立理由（1-2 句）
3. 最终推荐：PROCEED / REVISE / DROP

对于 discordant 的候选，请详细说明分歧原因。

输出 JSON 格式：
{{"external_reviews": [{{"candidate_title": "...", "agreement": "concordant|discordant", "reasoning": "...", "recommendation": "PROCEED|REVISE|DROP"}}]}}

只输出 JSON，不要其他文字。"""

    else:
        return f"Error: 未知的评审阶段 '{stage}'。使用 'candidates' 或 'review'。"

    # 调用评审模型
    messages = [
        {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        response = await reviewer_llm.collect_stream(messages)
        content = response.content or ""

        # 尝试解析 JSON
        import re
        # 提取 JSON 块
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return json.dumps(parsed, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pass

        # 如果解析失败，返回原始内容
        return content

    except Exception as e:
        logger.error("Reviewer evaluation failed: %s", e)
        return f"Error: 评审模型调用失败: {e}"
