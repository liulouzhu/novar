"""创新点搜索工具 - 多源论文检索 + 文献景观分析

借鉴 researchbot 的 InnovationWorkflowTool 中的数据检索部分，
适配为 Novare MCP 工具格式。"""

import asyncio
import json
import logging
import re
import os
from typing import Any

import httpx

from core.database import get_connection, upsert_paper
from tools.result import ok, fail, truncate, MAX_ABSTRACT

logger = logging.getLogger("research-server.innovation_search")

# ── 多源论文搜索 ─────────────────────────────────────────────────────────

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,authors,abstract,year,externalIds,citationCount,openAccessPdf"
ARXIV_API = "https://export.arxiv.org/api/query"
TITLE_KEY_RE = re.compile(r"[^a-z0-9]")


async def _search_semantic_scholar(query: str, limit: int = 8) -> list[dict]:
    """查询 Semantic Scholar"""
    headers = {"User-Agent": "Novare-ResearchAgent/0.1"}
    s2_key = os.environ.get("S2_API_KEY")
    if s2_key:
        headers["x-api-key"] = s2_key

    params = {"query": query, "limit": min(limit, 20), "fields": S2_FIELDS}
    async with httpx.AsyncClient(timeout=30, headers=headers, follow_redirects=True) as client:
        try:
            resp = await client.get(S2_API, params=params)
            if resp.status_code == 429:
                await asyncio.sleep(5)
                resp = await client.get(S2_API, params=params)
            resp.raise_for_status()
            data = resp.json()
            papers = []
            for item in data.get("data", []):
                papers.append({
                    "paper_id": item.get("paperId", ""),
                    "title": item.get("title", ""),
                    "authors": [a.get("name", "") for a in (item.get("authors") or [])[:5]],
                    "abstract": item.get("abstract", ""),
                    "year": item.get("year"),
                    "citation_count": item.get("citationCount", 0),
                    "source": "semantic_scholar",
                })
            return papers
        except Exception as e:
            logger.warning("Semantic Scholar search failed: %s", e)
            return []


async def _search_arxiv(query: str, limit: int = 8) -> list[dict]:
    """查询 arXiv"""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": min(limit, 20),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        try:
            resp = await client.get(ARXIV_API, params=params)
            resp.raise_for_status()
            # 解析 Atom XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            papers = []
            for entry in root.findall("atom:entry", ns):
                title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
                summary = (entry.findtext("atom:summary", "", ns) or "").strip()
                authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
                published = (entry.findtext("atom:published", "", ns) or "")[:4]
                arxiv_id = ""
                for link in entry.findall("atom:link", ns):
                    href = link.get("href", "")
                    if "/abs/" in href:
                        arxiv_id = href.split("/abs/")[-1]
                        break
                papers.append({
                    "paper_id": f"arxiv:{arxiv_id}" if arxiv_id else "",
                    "title": title,
                    "authors": authors[:5],
                    "abstract": summary,
                    "year": int(published) if published.isdigit() else None,
                    "source": "arxiv",
                })
            return papers
        except Exception as e:
            logger.warning("arXiv search failed: %s", e)
            return []


def _deduplicate_papers(papers: list[dict]) -> list[dict]:
    """按标题去重"""
    seen: set[str] = set()
    unique = []
    for p in papers:
        key = TITLE_KEY_RE.sub("", p.get("title", "").lower())[:80]
        if key and key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


async def _search_multi_source(query: str, max_per_source: int = 8) -> list[dict]:
    """并发搜索多个学术源"""
    s2_task = _search_semantic_scholar(query, max_per_source)
    arxiv_task = _search_arxiv(query, max_per_source)
    results = await asyncio.gather(s2_task, arxiv_task, return_exceptions=True)
    all_papers = []
    for r in results:
        if isinstance(r, list):
            all_papers.extend(r)
    return _deduplicate_papers(all_papers)


def _score_paper_relevance(paper: dict, query: str) -> float:
    """关键词相关性评分（借鉴 researchbot 的评分逻辑）"""
    score = 0.0
    query_terms = query.lower().split()
    title = paper.get("title", "").lower()
    abstract = paper.get("abstract", "").lower()

    for term in query_terms:
        if term in title:
            score += 5.0
        if term in abstract:
            score += 1.0
    return score


# ── MCP 工具入口 ─────────────────────────────────────────────────────────

async def handle_innovation_search(arguments: dict, user_id: str = None) -> str:
    """处理 innovation_search 工具调用

    子命令:
    - landscape: 文献景观扫描（多源搜索 + 相关性排序）
    - novelty_search: 针对特定候选的关键词搜索相关论文
    """
    action = arguments.get("action", "landscape")
    topic = arguments.get("topic", "")
    keywords = arguments.get("keywords", [])
    max_per_source = arguments.get("max_per_source", 8)

    if not topic and not keywords:
        return fail("innovation_search", "topic or keywords required")

    if action == "landscape":
        # Stage 0: 文献景观扫描
        query = topic or " ".join(keywords[:3])
        papers = await _search_multi_source(query, max_per_source)

        # 相关性排序
        for p in papers:
            p["_relevance"] = _score_paper_relevance(p, query)
        papers.sort(key=lambda x: x["_relevance"], reverse=True)

        # 保存到数据库并关联用户（best-effort：失败时带 warning 返回搜索结果）
        db_persist_warning = None
        try:
            async with get_connection() as conn:
                for p in papers:
                    if p.get("paper_id"):
                        await upsert_paper(conn, {
                            "id": p["paper_id"],
                            "title": p.get("title", ""),
                            "abstract": p.get("abstract", ""),
                            "authors": json.dumps(p.get("authors", []), ensure_ascii=False),
                            "year": p.get("year"),
                            "source": p.get("source", ""),
                            "pdf_path": None,
                            "url": None,
                            "citation_count": p.get("citation_count", 0),
                            "visibility": "public",
                        })
                if user_id:
                    from tools.paper_parse import associate_user_paper
                    for p in papers:
                        if p.get("paper_id"):
                            await associate_user_paper(
                                user_id, p["paper_id"],
                                relation_type="searched",
                                has_fulltext_access=False,
                                source="innovation_search",
                            )
        except Exception as e:
            logger.warning("Failed to persist papers to DB: %s", e)
            db_persist_warning = "论文搜索成功，但数据库持久化失败，本次结果可能不会出现在论文库中。"

        papers_json = [
            {
                "paper_id": p.get("paper_id", ""),
                "title": p.get("title", ""),
                "authors": p.get("authors", [])[:10],
                "abstract": truncate(p.get("abstract", "") or "", MAX_ABSTRACT),
                "year": p.get("year"),
                "citation_count": p.get("citation_count", 0),
                "source": p.get("source", ""),
            }
            for p in papers[:30]
        ]
        sources = [{"id": p["paper_id"], "title": p.get("title", "")} for p in papers_json if p.get("paper_id")]
        providers = list(set(p.get("source", "") for p in papers if p.get("source")))

        warnings = [db_persist_warning] if db_persist_warning else []
        return ok(
            "innovation_search",
            {"action": "landscape", "topic": topic, "total_papers": len(papers), "papers": papers_json},
            summary=f"文献景观扫描 '{topic}' 找到 {len(papers)} 篇论文",
            sources=sources,
            providers=providers,
            warnings=warnings,
        )

    elif action == "novelty_search":
        # Stage 2: 针对候选创新点的关键词搜索
        query = " OR ".join(keywords[:3]) if keywords else topic
        papers = await _search_multi_source(query, max_per_source)

        papers_json = [
            {
                "paper_id": p.get("paper_id", ""),
                "title": p.get("title", ""),
                "authors": p.get("authors", [])[:10],
                "abstract": truncate(p.get("abstract", "") or "", MAX_ABSTRACT),
                "year": p.get("year"),
                "citation_count": p.get("citation_count", 0),
                "source": p.get("source", ""),
            }
            for p in papers[:15]
        ]
        sources = [{"id": p["paper_id"], "title": p.get("title", "")} for p in papers_json if p.get("paper_id")]
        providers = list(set(p.get("source", "") for p in papers if p.get("source")))

        return ok(
            "innovation_search",
            {"action": "novelty_search", "query": query, "total_papers": len(papers), "papers": papers_json},
            summary=f"新颖性搜索 '{query}' 找到 {len(papers)} 篇相关论文",
            sources=sources,
            providers=providers,
        )

    else:
        return fail("innovation_search", f"未知的 action '{action}'。使用 'landscape' 或 'novelty_search'。")
