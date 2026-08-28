"""Tavily Web 搜索工具 - 通用网页搜索

使用 Tavily API 进行网页搜索，支持基础搜索和高级搜索深度。
可用于搜索最新资讯、技术文档、博客文章等非学术内容。
"""

import logging
import os
from typing import Optional

import httpx

from tools.result import ok, fail

logger = logging.getLogger("research-server.tavily_search")

# Tavily API 配置
TAVILY_API_URL = "https://api.tavily.com/search"
DEFAULT_MAX_RESULTS = 5
MAX_MAX_RESULTS = 20


async def _search_tavily(
    query: str,
    search_depth: str = "basic",
    max_results: int = 5,
    include_answer: bool = True,
    include_raw_content: bool = False,
    topic: str = "general",
) -> tuple[dict, str]:
    """
    调用 Tavily API 进行网页搜索。

    Args:
        query: 搜索查询
        search_depth: 搜索深度 ("basic" 或 "advanced")
        max_results: 最大返回结果数
        include_answer: 是否包含 AI 生成的答案
        include_raw_content: 是否包含原始网页内容
        topic: 搜索主题类别 ("general", "news", "finance" 等)

    Returns:
        (搜索结果字典, 错误信息)
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {}, "Tavily API key not configured (TAVILY_API_KEY)"

    payload = {
        "query": query,
        "search_depth": search_depth,
        "max_results": min(max_results, MAX_MAX_RESULTS),
        "include_answer": include_answer,
        "include_raw_content": include_raw_content,
        "topic": topic,
    }

    headers = {
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30, headers=headers, follow_redirects=True) as client:
        for attempt in range(3):
            try:
                resp = await client.post(
                    TAVILY_API_URL,
                    json=payload,
                    auth=("tavily", api_key),
                )

                if resp.status_code == 429:
                    # Rate limited
                    retry_after = resp.headers.get("Retry-After")
                    wait = min(int(retry_after) if retry_after else 2 ** (attempt + 1), 30)
                    logger.warning("Tavily rate limited, retrying in %ds (attempt %d/3)", wait, attempt + 1)
                    import asyncio
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                return data, ""

            except httpx.HTTPStatusError as e:
                logger.error("Tavily API error (attempt %d): %s", attempt + 1, e)
                if e.response.status_code == 429 and attempt < 2:
                    import asyncio
                    await asyncio.sleep(2 ** (attempt + 1))
                    continue
                return {}, f"Tavily: HTTP {e.response.status_code} - {e.response.text[:200]}"
            except Exception as e:
                logger.error("Tavily search failed (attempt %d): %s", attempt + 1, e)
                if attempt < 2:
                    import asyncio
                    await asyncio.sleep(1)
                    continue
                return {}, f"Tavily: {type(e).__name__} - {str(e)[:200]}"

    return {}, "Tavily: 所有重试均失败"


def _format_results(data: dict, query: str) -> str:
    """格式化搜索结果为可读文本"""
    results = data.get("results", [])
    answer = data.get("answer")

    if not results and not answer:
        return f"搜索 '{query}' 未找到相关网页结果。"

    lines = []

    # AI 生成的答案（如果有）
    if answer:
        lines.append("**AI 摘要：**")
        lines.append(answer)
        lines.append("")

    # 搜索结果列表
    if results:
        lines.append(f"**找到 {len(results)} 条网页结果：**\n")
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            url = r.get("url", "")
            content = r.get("content", "")
            score = r.get("score", 0)

            lines.append(f"**{i}. {title}**")
            lines.append(f"   URL: {url}")
            lines.append(f"   相关度: {score:.2f}")
            if content:
                # 截断过长的内容
                display_content = content[:500] + "..." if len(content) > 500 else content
                lines.append(f"   摘要: {display_content}")
            lines.append("")

    return "\n".join(lines)


def _build_result_json(data: dict, query: str) -> dict:
    """构建结构化的 JSON 结果"""
    results = data.get("results", [])
    answer = data.get("answer")

    formatted_results = []
    for r in results:
        formatted_results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0),
            "published_date": r.get("published_date"),
        })

    return {
        "query": query,
        "answer": answer,
        "results": formatted_results,
        "total": len(formatted_results),
    }


async def handle_tavily_search(args: dict, user_id: str = None) -> str:
    """
    Tavily 网页搜索入口

    Args:
        args: 工具调用参数
        query: 搜索查询（必填）
        search_depth: 搜索深度 - "basic"(快速) 或 "advanced"(深入), 默认 "basic"
        max_results: 最大返回数量, 默认 5, 最大 20
        include_answer: 是否包含 AI 答案, 默认 true
        topic: 搜索主题 - "general"(通用), "news"(新闻), "finance"(金融)

    Returns:
        JSON 格式的搜索结果
    """
    query = args.get("query", "").strip()
    if not query:
        return fail("tavily_search", "请提供搜索查询 (query)。")

    search_depth = args.get("search_depth", "basic")
    if search_depth not in ("basic", "advanced"):
        search_depth = "basic"

    max_results = min(args.get("max_results", DEFAULT_MAX_RESULTS), MAX_MAX_RESULTS)
    include_answer = args.get("include_answer", True)
    topic = args.get("topic", "general")

    logger.info("Tavily search: query=%s, depth=%s, max=%d", query, search_depth, max_results)

    # 执行搜索
    data, error = await _search_tavily(
        query=query,
        search_depth=search_depth,
        max_results=max_results,
        include_answer=include_answer,
        topic=topic,
    )

    if error:
        return fail("tavily_search", error)

    if not data:
        return fail("tavily_search", "搜索返回空结果")

    # 构建结果
    result_data = _build_result_json(data, query)

    # 构建 sources（用于 RAG 证据溯源）
    sources = [
        {"id": r["url"], "title": r["title"]}
        for r in result_data["results"]
        if r.get("url")
    ]

    # 构建 summary
    answer = result_data.get("answer")
    total = result_data["total"]
    if answer:
        summary = f"搜索 '{query}' 找到 {total} 条结果，AI 摘要已生成"
    elif total > 0:
        summary = f"搜索 '{query}' 找到 {total} 条网页结果"
    else:
        summary = f"搜索 '{query}' 未找到相关结果"

    return ok(
        "tavily_search",
        result_data,
        summary=summary,
        sources=sources,
        providers=["tavily"],
    )
