"""论文检索工具 - 并行查询 Semantic Scholar + arXiv

Redis 缓存：相同查询在 TTL 内直接返回缓存结果，按 user_id 隔离。
"""

import asyncio
import logging
import os
import xml.etree.ElementTree as ET
from typing import Optional

import httpx

from core.database import get_connection, upsert_paper
from tools.result import ok, fail, truncate, MAX_ABSTRACT

# 缓存 helper（纯计算，无 Redis 依赖）
from novare.cache import make_cache_key, cacheable_size

# Redis 访问（MCP server 进程可能没有 web.backend，安全降级）
try:
    from web.backend.redis_service import redis_service
except Exception:
    redis_service = None

logger = logging.getLogger("research-server.paper_search")

_CACHE_TTL = 1800  # 30 分钟

# Semantic Scholar API
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,authors,abstract,year,externalIds,citationCount,openAccessPdf"

# arXiv API
ARXIV_API = "https://export.arxiv.org/api/query"


async def _search_semantic_scholar(
    query: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = 10,
) -> tuple[list[dict], str]:
    """查询 Semantic Scholar API。返回 (结果列表, 错误信息)"""
    params = {
        "query": query,
        "limit": min(limit, 20),
        "fields": S2_FIELDS,
    }
    if year_from or year_to:
        year_range = f"{year_from or ''}-{year_to or ''}"
        params["year"] = year_range

    headers = {"User-Agent": "Novare-ResearchAgent/0.1"}
    s2_key = os.environ.get("S2_API_KEY")
    if s2_key:
        headers["x-api-key"] = s2_key

    async with httpx.AsyncClient(timeout=30, headers=headers, follow_redirects=True) as client:
        for attempt in range(3):
            try:
                resp = await client.get(S2_API, params=params)

                if resp.status_code == 429:
                    # 优先使用 Retry-After header，否则指数退避
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        wait = min(int(retry_after), 30)
                    else:
                        wait = 2 ** attempt
                    logger.warning("Semantic Scholar rate limited, retrying in %ds (attempt %d/3)", wait, attempt + 1)
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()

                results = []
                for paper in data.get("data", []):
                    ext_ids = paper.get("externalIds", {}) or {}
                    paper_id = (
                        ext_ids.get("DOI")
                        or ext_ids.get("ArXiv")
                        or paper.get("paperId", "")
                    )
                    if not paper_id:
                        continue

                    if ext_ids.get("DOI"):
                        source_id = f"doi:{ext_ids['DOI']}"
                    elif ext_ids.get("ArXiv"):
                        source_id = f"arxiv:{ext_ids['ArXiv']}"
                    else:
                        source_id = f"s2:{paper.get('paperId', '')}"

                    authors = [a.get("name", "") for a in (paper.get("authors") or [])]
                    pdf_url = None
                    oa = paper.get("openAccessPdf")
                    if oa:
                        pdf_url = oa.get("url")

                    results.append({
                        "id": source_id,
                        "identifiers": [
                            identifier for identifier in (
                                f"doi:{ext_ids['DOI']}" if ext_ids.get("DOI") else None,
                                f"arxiv:{ext_ids['ArXiv']}" if ext_ids.get("ArXiv") else None,
                                f"s2:{paper.get('paperId', '')}" if paper.get("paperId") else None,
                            ) if identifier
                        ],
                        "title": paper.get("title", ""),
                        "authors": authors,
                        "abstract": paper.get("abstract", ""),
                        "year": paper.get("year"),
                        "source": "semantic_scholar",
                        "url": f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}",
                        "pdf_url": pdf_url,
                        "citation_count": paper.get("citationCount", 0),
                    })
                return results, ""

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    if attempt < 2:
                        retry_after = e.response.headers.get("Retry-After")
                        wait = min(int(retry_after), 30) if retry_after else 2 ** attempt
                        await asyncio.sleep(wait)
                        continue
                    return [], "Semantic Scholar: 请求频率超限(429)，请稍后重试"
                logger.error("Semantic Scholar API error: %s", e)
                return [], f"Semantic Scholar: HTTP {e.response.status_code}"
            except Exception as e:
                logger.error("Semantic Scholar search failed: %s", e)
                return [], f"Semantic Scholar: {type(e).__name__}"

    return [], "Semantic Scholar: 请求频率超限(429)，已重试3次"


async def _search_arxiv(
    query: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = 10,
) -> tuple[list[dict], str]:
    """查询 arXiv API。返回 (结果列表, 错误信息)"""
    search_query = f"all:{query}"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": min(limit, 20),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    async with httpx.AsyncClient(timeout=60, headers={"User-Agent": "Novare-ResearchAgent/0.1"}, follow_redirects=True) as client:
        for attempt in range(3):
            try:
                resp = await client.get(ARXIV_API, params=params)

                if resp.status_code in (429, 503):
                    wait = 3 * (attempt + 1)  # arXiv 建议至少 3s 间隔
                    logger.warning("arXiv rate limited (%d), retrying in %ds", resp.status_code, wait)
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                xml_text = resp.text

                # 解析 Atom XML
                ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
                root = ET.fromstring(xml_text)

                results = []
                for entry in root.findall("atom:entry", ns):
                    title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
                    summary = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")

                    entry_id = entry.findtext("atom:id", "", ns)
                    arxiv_id = entry_id.split("/abs/")[-1] if "/abs/" in entry_id else entry_id
                    arxiv_base = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id

                    authors = []
                    for author in entry.findall("atom:author", ns):
                        name = author.findtext("atom:name", "", ns)
                        if name:
                            authors.append(name)

                    published = entry.findtext("atom:published", "")
                    year = int(published[:4]) if published and len(published) >= 4 else None

                    if year_from and year and year < year_from:
                        continue
                    if year_to and year and year > year_to:
                        continue

                    pdf_url = None
                    for link in entry.findall("atom:link", ns):
                        if link.get("title") == "pdf":
                            pdf_url = link.get("href")
                            break

                    results.append({
                        "id": f"arxiv:{arxiv_base}",
                        "identifiers": [f"arxiv:{arxiv_base}"],
                        "title": title,
                        "authors": authors,
                        "abstract": summary[:500],
                        "year": year,
                        "source": "arxiv",
                        "url": f"https://arxiv.org/abs/{arxiv_base}",
                        "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_base}",
                        "citation_count": 0,
                    })
                return results, ""

            except httpx.HTTPStatusError as e:
                logger.error("arXiv API error (attempt %d): %s", attempt + 1, e)
                if e.response.status_code in (429, 503) and attempt < 2:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                return [], f"arXiv: HTTP {e.response.status_code}"
            except Exception as e:
                logger.error("arXiv search failed (attempt %d): %s", attempt + 1, e)
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
                return [], f"arXiv: {type(e).__name__}"

    return [], "arXiv: 已重试3次仍失败"


def _merge_results(s2_results: list[dict], arxiv_results: list[dict], limit: int) -> list[dict]:
    """合并去重两个源的结果"""
    seen_ids = set()
    merged = []

    # 优先 Semantic Scholar（有引用数）
    for paper in s2_results:
        key = paper["id"]
        if key not in seen_ids:
            seen_ids.add(key)
            merged.append(paper)

    # 补充 arXiv 结果
    for paper in arxiv_results:
        key = paper["id"]
        if key not in seen_ids:
            seen_ids.add(key)
            merged.append(paper)

    # 按引用数排序
    merged.sort(key=lambda p: p.get("citation_count", 0), reverse=True)
    return merged[:limit]


def _format_results(papers: list[dict]) -> str:
    """格式化论文列表为可读文本（CLI 兼容保留，主路径走 JSON）"""
    if not papers:
        return "未找到相关论文。请尝试不同的搜索词。"

    lines = [f"找到 {len(papers)} 篇相关论文：\n"]
    for i, p in enumerate(papers, 1):
        authors_str = ", ".join(p["authors"][:3])
        if len(p["authors"]) > 3:
            authors_str += " et al."

        lines.append(f"**{i}. {p['title']}**")
        lines.append(f"   作者: {authors_str}")
        lines.append(f"   年份: {p.get('year', 'N/A')} | 引用: {p.get('citation_count', 0)} | 来源: {p.get('source', 'N/A')}")
        lines.append(f"   ID: {p['id']}")

        if p.get("pdf_url"):
            lines.append(f"   PDF: {p['pdf_url']}")

        abstract = p.get("abstract", "")
        if abstract:
            lines.append(f"   摘要: {abstract[:300]}{'...' if len(abstract) > 300 else ''}")
        lines.append("")

    return "\n".join(lines)


def _build_paper_json(paper: dict) -> dict:
    """将内部 paper dict 转为 JSON 输出格式（含大小截断）"""
    authors = paper.get("authors", [])
    return {
        "paper_id": paper["id"],
        "title": paper.get("title", ""),
        "authors": authors[:10],
        "year": paper.get("year"),
        "abstract": truncate(paper.get("abstract", ""), MAX_ABSTRACT),
        "citation_count": paper.get("citation_count", 0),
        "url": paper.get("url", ""),
        "pdf_url": paper.get("pdf_url"),
    }


async def handle_paper_search(args: dict, user_id: str = None) -> str:
    """论文检索入口 — 返回统一 JSON 格式（带 Redis 缓存）"""
    query = args.get("query", "").strip()
    if not query:
        return fail("paper_search", "请提供搜索关键词。")

    year_from = args.get("year_from")
    year_to = args.get("year_to")
    limit = min(args.get("limit", 10), 20)

    # ── Redis 缓存：尝试命中 ──
    _rs = redis_service
    cache_key = make_cache_key("paper_search", user_id, {
        "query": query, "limit": limit,
        "year_from": year_from, "year_to": year_to,
    })
    if cache_key and _rs and _rs.is_available:
        try:
            cached = await _rs.get_json(cache_key)
            if cached and isinstance(cached, dict) and "result" in cached:
                logger.info("paper_search cache hit: %s", cache_key)
                return cached["result"]
        except Exception:
            logger.debug("cache read failed (non-fatal)", exc_info=True)

    # ── 原始搜索逻辑 ──
    # 并行查询两个源
    s2_task = _search_semantic_scholar(query, year_from, year_to, limit)
    arxiv_task = _search_arxiv(query, year_from, year_to, limit)
    (s2_results, s2_err), (arxiv_results, arxiv_err) = await asyncio.gather(s2_task, arxiv_task)

    # 合并结果
    merged = _merge_results(s2_results, arxiv_results, limit)

    # 写入数据库并关联用户
    if merged:
        try:
            async with get_connection() as conn:
                for paper in merged:
                    paper["visibility"] = "public"
                    await upsert_paper(conn, paper)
                deduplicated = {}
                for paper in merged:
                    deduplicated.setdefault(paper["id"], paper)
                merged = list(deduplicated.values())
                # 搜索到的论文自动关联到当前用户
                if user_id:
                    from tools.paper_parse import associate_user_paper
                    for paper in merged:
                        await associate_user_paper(
                            user_id, paper["id"],
                            relation_type="searched",
                            has_fulltext_access=False,
                            source="paper_search",
                        )
        except Exception as e:
            logger.warning("Failed to save papers to DB: %s", e)

    # ── 收集 providers 和 warnings ──
    providers = []
    warnings = []
    if s2_results:
        providers.append("semantic_scholar")
    if s2_err:
        warnings.append(f"Semantic Scholar: {s2_err}")
    if arxiv_results:
        providers.append("arxiv")
    if arxiv_err:
        warnings.append(f"arXiv: {arxiv_err}")

    # ── 构建 sources（证据来源） ──
    sources = [
        {"id": p["id"], "title": p.get("title", "")}
        for p in merged
    ]

    # ── 返回结果 ──
    result: str
    if merged:
        papers_json = [_build_paper_json(p) for p in merged]
        result = ok(
            "paper_search",
            {
                "query": query,
                "total": len(merged),
                "papers": papers_json,
            },
            summary=f"搜索 '{query}' 找到 {len(merged)} 篇论文",
            sources=sources,
            providers=providers,
            warnings=warnings,
        )
    else:
        # 两个源都失败了
        errors = [e for e in (s2_err, arxiv_err) if e]
        if errors:
            return fail("paper_search", f"搜索失败，所有数据源均不可用：{'; '.join(errors)}")
        result = ok(
            "paper_search",
            {"query": query, "total": 0, "papers": []},
            summary=f"搜索 '{query}' 未找到相关论文",
            providers=providers,
            warnings=warnings or ["请尝试不同的搜索词"],
        )

    # ── Redis 缓存：写入成功结果 ──
    if cache_key and _rs and _rs.is_available:
        try:
            if cacheable_size(result):
                await _rs.set_json(cache_key, {"result": result}, ttl=_CACHE_TTL)
        except Exception:
            logger.debug("cache write failed (non-fatal)", exc_info=True)

    return result
