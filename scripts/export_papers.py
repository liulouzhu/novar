"""从 PostgreSQL 导出已解析论文数据为 Markdown 文件。

用法：
    # 导出所有论文
    python scripts/export_papers.py

    # 指定输出目录
    python scripts/export_papers.py --output ./exported_papers

    # 只导出指定论文
    python scripts/export_papers.py --paper-ids arxiv:2308.11681 doi:10.1234/abc

    # 同时导出评测集（query-ground_truth 对）
    python scripts/export_papers.py --eval-set

环境变量：
    DATABASE_URL — PostgreSQL 连接串（必需）
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# mcp-server 目录
MCP_ROOT = PROJECT_ROOT / "mcp-server"
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from sqlalchemy import select, func
from web.backend.db.base import get_session_factory, get_engine, DATABASE_URL
from web.backend.db.models import Paper, Chunk, Embedding, Citation, UserPaper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("export_papers")


# ── 数据查询 ─────────────────────────────────────────────────────────────

async def fetch_all_papers(db) -> list[dict]:
    """获取所有论文元数据。"""
    result = await db.execute(
        select(Paper).order_by(Paper.created_at.desc())
    )
    rows = result.scalars().all()
    papers = []
    for p in rows:
        papers.append({
            "id": p.id,
            "title": p.title,
            "authors": p.authors if isinstance(p.authors, list) else [],
            "abstract": p.abstract or "",
            "year": p.year,
            "source": p.source,
            "pdf_path": p.pdf_path,
            "url": p.url,
            "citation_count": p.citation_count or 0,
            "visibility": p.visibility,
            "created_at": p.created_at.isoformat() if p.created_at else "",
        })
    return papers


async def fetch_chunks_for_paper(db, paper_id: str) -> list[dict]:
    """获取指定论文的所有 chunks，按 ordinal 排序。"""
    result = await db.execute(
        select(Chunk)
        .where(Chunk.paper_id == paper_id)
        .order_by(Chunk.ordinal)
    )
    rows = result.scalars().all()
    return [
        {
            "id": c.id,
            "section": c.section or "Unknown",
            "ordinal": c.ordinal or 0,
            "text": c.text,
        }
        for c in rows
    ]


async def fetch_embedding_stats(db) -> dict:
    """获取 embedding 统计信息。"""
    result = await db.execute(
        select(Embedding.dim, func.count(Embedding.chunk_id))
        .group_by(Embedding.dim)
    )
    return {row[0]: row[1] for row in result.all()}


async def fetch_citations_for_paper(db, paper_id: str) -> dict:
    """获取指定论文的引用关系。"""
    # 该论文引用的论文
    result_citing = await db.execute(
        select(Citation.target_id).where(Citation.source_id == paper_id)
    )
    citing = [r[0] for r in result_citing.all()]

    # 引用该论文的论文
    result_cited_by = await db.execute(
        select(Citation.source_id).where(Citation.target_id == paper_id)
    )
    cited_by = [r[0] for r in result_cited_by.all()]

    return {"citing": citing, "cited_by": cited_by}


async def fetch_user_paper_stats(db) -> dict:
    """获取用户-论文关联统计。"""
    result = await db.execute(
        select(
            UserPaper.paper_id,
            func.count(UserPaper.user_id).label("user_count"),
        )
        .group_by(UserPaper.paper_id)
    )
    return {row[0]: row[1] for row in result.all()}


# ── Markdown 生成 ────────────────────────────────────────────────────────

def _safe_filename(paper_id: str) -> str:
    """将 paper_id 转为安全文件名。"""
    # arxiv:2308.11681 → arxiv_2308.11681
    name = re.sub(r'[<>:"/\\|?*]', '_', paper_id)
    return name


def paper_to_markdown(paper: dict, chunks: list[dict], citations: dict) -> str:
    """将论文数据转为 Markdown 格式。"""
    lines = []

    # Frontmatter
    lines.append("---")
    lines.append(f"paper_id: \"{paper['id']}\"")
    lines.append(f"title: \"{paper['title']}\"")
    authors_str = ", ".join(paper["authors"][:10]) if paper["authors"] else "Unknown"
    lines.append(f"authors: \"{authors_str}\"")
    if paper["year"]:
        lines.append(f"year: {paper['year']}")
    lines.append(f"source: \"{paper['source'] or 'unknown'}\"")
    lines.append(f"citation_count: {paper['citation_count']}")
    lines.append(f"visibility: \"{paper['visibility']}\"")
    if paper["url"]:
        lines.append(f"url: \"{paper['url']}\"")
    if paper["pdf_path"]:
        lines.append(f"pdf_path: \"{paper['pdf_path']}\"")
    lines.append(f"exported_at: \"{datetime.now().isoformat()}\"")
    lines.append(f"chunk_count: {len(chunks)}")
    lines.append("---")
    lines.append("")

    # 标题
    lines.append(f"# {paper['title']}")
    lines.append("")

    # 元信息
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Paper ID**: `{paper['id']}`")
    lines.append(f"- **Authors**: {authors_str}")
    if paper["year"]:
        lines.append(f"- **Year**: {paper['year']}")
    lines.append(f"- **Source**: {paper['source'] or 'unknown'}")
    lines.append(f"- **Citations**: {paper['citation_count']}")
    if paper["url"]:
        lines.append(f"- **URL**: {paper['url']}")
    lines.append("")

    # 摘要
    if paper["abstract"]:
        lines.append("## Abstract")
        lines.append("")
        lines.append(paper["abstract"])
        lines.append("")

    # 引用关系
    if citations["citing"] or citations["cited_by"]:
        lines.append("## Citations")
        lines.append("")
        if citations["citing"]:
            lines.append(f"**References ({len(citations['citing'])})**:")
            for ref_id in citations["citing"][:20]:
                lines.append(f"- `{ref_id}`")
            if len(citations["citing"]) > 20:
                lines.append(f"- ... and {len(citations['citing']) - 20} more")
        if citations["cited_by"]:
            lines.append(f"**Cited by ({len(citations['cited_by'])})**:")
            for ref_id in citations["cited_by"][:20]:
                lines.append(f"- `{ref_id}`")
            if len(citations["cited_by"]) > 20:
                lines.append(f"- ... and {len(citations['cited_by']) - 20} more")
        lines.append("")

    # 按章节组织 chunks
    if chunks:
        lines.append("## Content")
        lines.append("")

        # 按 section 分组
        sections: dict[str, list[dict]] = {}
        for c in chunks:
            sec = c["section"]
            if sec not in sections:
                sections[sec] = []
            sections[sec].append(c)

        for sec_name, sec_chunks in sections.items():
            lines.append(f"### {sec_name}")
            lines.append("")
            for c in sec_chunks:
                lines.append(c["text"].strip())
                lines.append("")

    return "\n".join(lines)


# ── 评测集生成 ────────────────────────────────────────────────────────────

def generate_eval_set(papers: list[dict], all_chunks: dict[str, list[dict]]) -> list[dict]:
    """基于已有论文数据生成评测集。

    每个评测用例包含：
    - query: 自然语言问题
    - expected_paper_ids: 期望命中的论文 ID 列表
    - expected_keywords: 期望在答案中出现的关键词
    - category: 评测类别
    """
    eval_cases = []

    for paper in papers:
        pid = paper["id"]
        chunks = all_chunks.get(pid, [])
        if not chunks:
            continue

        title = paper["title"]
        abstract = paper.get("abstract", "")

        # 1. 精确论文检索：基于标题
        if title:
            eval_cases.append({
                "query": f"What is the main contribution of the paper \"{title}\"?",
                "expected_paper_ids": [pid],
                "expected_keywords": _extract_keywords_from_text(title),
                "category": "paper_retrieval",
                "source_paper_id": pid,
            })

        # 2. 基于 abstract 的语义检索
        if abstract and len(abstract) > 50:
            # 取 abstract 的第一句作为查询基础
            first_sentence = _extract_first_sentence(abstract)
            if first_sentence and len(first_sentence) > 20:
                eval_cases.append({
                    "query": f"Find papers that discuss: {first_sentence}",
                    "expected_paper_ids": [pid],
                    "expected_keywords": _extract_keywords_from_text(abstract),
                    "category": "semantic_retrieval",
                    "source_paper_id": pid,
                })

        # 3. 基于 chunks 的内容检索
        for chunk in chunks[:3]:  # 每篇论文最多 3 个 chunk
            text = chunk["text"]
            if len(text) < 100:
                continue
            # 取 chunk 中的关键句子
            key_sentence = _extract_key_sentence(text)
            if key_sentence:
                eval_cases.append({
                    "query": f"Which paper mentions: \"{key_sentence[:200]}\"",
                    "expected_paper_ids": [pid],
                    "expected_keywords": _extract_keywords_from_text(text),
                    "category": "chunk_retrieval",
                    "source_paper_id": pid,
                    "source_chunk_id": chunk["id"],
                })

        # 4. paper_id 过滤测试
        if len(chunks) > 0:
            eval_cases.append({
                "query": f"Summarize the content of paper {pid}",
                "expected_paper_ids": [pid],
                "expected_keywords": [],
                "category": "paper_id_filter",
                "source_paper_id": pid,
                "filter_paper_id": pid,
            })

    return eval_cases


def _extract_first_sentence(text: str) -> str:
    """提取第一句话。"""
    # 按句号、问号、感叹号分割
    match = re.match(r'^(.+?[.!?])\s', text)
    if match:
        return match.group(1).strip()
    # fallback: 取前 200 字符
    return text[:200].strip()


def _extract_key_sentence(text: str) -> str:
    """提取包含关键信息的句子（优先选择含数字/百分比的句子）。"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # 优先选择包含数字的句子
    for s in sentences:
        if re.search(r'\d+\.?\d*[%]?', s) and len(s) > 30:
            return s.strip()
    # fallback: 取第一个足够长的句子
    for s in sentences:
        if len(s) > 30:
            return s.strip()
    return ""


def _extract_keywords_from_text(text: str) -> list[str]:
    """从文本中提取关键词（简单的 TF 方法）。"""
    # 停用词
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "under", "again",
        "this", "that", "these", "those", "and", "but", "or", "nor", "not",
        "so", "if", "than", "too", "very", "just", "about", "also", "more",
        "some", "such", "no", "only", "own", "same", "other", "then",
        "which", "who", "whom", "what", "where", "when", "why", "how",
        "all", "each", "every", "both", "few", "most", "other", "some",
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_freq: dict[str, int] = {}
    for w in words:
        if w not in stopwords:
            word_freq[w] = word_freq.get(w, 0) + 1
    # 取频率最高的 10 个词
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:10]]


# ── 主流程 ────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="导出 PostgreSQL 中的论文数据为 Markdown")
    parser.add_argument("--output", "-o", default="./exported_papers",
                        help="输出目录（默认 ./exported_papers）")
    parser.add_argument("--paper-ids", nargs="*", default=None,
                        help="只导出指定的论文 ID（留空则导出全部）")
    parser.add_argument("--eval-set", action="store_true",
                        help="同时生成评测集")
    parser.add_argument("--stats", action="store_true",
                        help="只输出统计信息，不导出文件")
    args = parser.parse_args()

    if not DATABASE_URL:
        logger.error("DATABASE_URL 未设置。请在 .env 文件中配置。")
        sys.exit(1)

    logger.info("Connecting to database...")
    factory = get_session_factory()

    async with factory() as db:
        # 获取论文列表
        if args.paper_ids:
            papers = []
            for pid in args.paper_ids:
                result = await db.execute(select(Paper).where(Paper.id == pid))
                p = result.scalar_one_or_none()
                if p:
                    papers.append({
                        "id": p.id, "title": p.title,
                        "authors": p.authors if isinstance(p.authors, list) else [],
                        "abstract": p.abstract or "", "year": p.year,
                        "source": p.source, "pdf_path": p.pdf_path,
                        "url": p.url, "citation_count": p.citation_count or 0,
                        "visibility": p.visibility,
                        "created_at": p.created_at.isoformat() if p.created_at else "",
                    })
                else:
                    logger.warning("Paper not found: %s", pid)
        else:
            papers = await fetch_all_papers(db)

        logger.info("Found %d papers", len(papers))

        # 统计信息
        embedding_stats = await fetch_embedding_stats(db)
        user_paper_stats = await fetch_user_paper_stats(db)

        logger.info("Embedding stats: %s", embedding_stats)
        logger.info("Total user-paper associations: %d", sum(user_paper_stats.values()))

        if args.stats:
            # 打印统计摘要
            print("\n=== 数据库统计 ===")
            print(f"论文总数: {len(papers)}")
            print(f"Embedding 维度分布: {embedding_stats}")
            print(f"用户-论文关联总数: {sum(user_paper_stats.values())}")

            # 每篇论文的 chunk 数量
            total_chunks = 0
            for p in papers:
                result = await db.execute(
                    select(func.count(Chunk.id)).where(Chunk.paper_id == p["id"])
                )
                cnt = result.scalar() or 0
                total_chunks += cnt
            print(f"Chunk 总数: {total_chunks}")
            return

        # 创建输出目录
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 导出每篇论文
        all_chunks: dict[str, list[dict]] = {}
        for i, paper in enumerate(papers, 1):
            pid = paper["id"]
            logger.info("[%d/%d] Exporting: %s — %s", i, len(papers), pid, paper["title"][:60])

            chunks = await fetch_chunks_for_paper(db, pid)
            all_chunks[pid] = chunks
            citations = await fetch_citations_for_paper(db, pid)

            md_content = paper_to_markdown(paper, chunks, citations)

            filename = _safe_filename(pid) + ".md"
            filepath = output_dir / filename
            filepath.write_text(md_content, encoding="utf-8")

        logger.info("Exported %d papers to %s", len(papers), output_dir)

        # 生成评测集
        if args.eval_set:
            eval_cases = generate_eval_set(papers, all_chunks)
            eval_path = output_dir / "eval_set.json"
            eval_path.write_text(
                json.dumps(eval_cases, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Generated %d eval cases → %s", len(eval_cases), eval_path)

            # 打印评测集摘要
            categories = {}
            for ec in eval_cases:
                cat = ec["category"]
                categories[cat] = categories.get(cat, 0) + 1
            print("\n=== 评测集摘要 ===")
            for cat, cnt in sorted(categories.items()):
                print(f"  {cat}: {cnt}")
            print(f"  总计: {len(eval_cases)}")

    # 清理引擎
    await get_engine().dispose()


if __name__ == "__main__":
    asyncio.run(main())
