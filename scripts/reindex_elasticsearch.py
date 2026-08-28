"""Elasticsearch 索引重建脚本

从 PostgreSQL 读取所有 papers/chunks，批量写入 Elasticsearch。
默认 dry-run，需要 --apply 显式执行写入。

用法：
    # Dry-run：只统计，不写入
    python scripts/reindex_elasticsearch.py

    # 执行写入
    python scripts/reindex_elasticsearch.py --apply

    # 局部重建指定论文
    python scripts/reindex_elasticsearch.py --apply --paper-id arxiv:2308.11681

环境变量：
    DATABASE_URL — PostgreSQL 连接串（必需）
    ELASTICSEARCH_URL — ES 地址（默认 http://localhost:9200）
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MCP_ROOT = PROJECT_ROOT / "mcp-server"
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from sqlalchemy import select, func
from web.backend.db.base import get_session_factory, get_engine, DATABASE_URL
from web.backend.db.models import Paper, Chunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("reindex_es")


async def fetch_papers_with_chunks(db, paper_id: str | None = None) -> list[dict]:
    """获取论文及其 chunks。"""
    if paper_id:
        result = await db.execute(select(Paper).where(Paper.id == paper_id))
    else:
        result = await db.execute(select(Paper).order_by(Paper.created_at))
    papers = result.scalars().all()

    data = []
    for p in papers:
        chunk_result = await db.execute(
            select(Chunk)
            .where(Chunk.paper_id == p.id)
            .order_by(Chunk.ordinal)
        )
        chunks = chunk_result.scalars().all()
        data.append({
            "paper_id": p.id,
            "title": p.title,
            "chunks": [
                {
                    "chunk_id": c.id,
                    "section": c.section or "",
                    "text": c.text,
                }
                for c in chunks
            ],
        })
    return data


async def main():
    parser = argparse.ArgumentParser(description="Elasticsearch 索引重建")
    parser.add_argument("--apply", action="store_true", help="执行写入（默认 dry-run）")
    parser.add_argument("--paper-id", default=None, help="只重建指定论文")
    parser.add_argument("--batch-size", type=int, default=100, help="批量写入大小")
    args = parser.parse_args()

    if not DATABASE_URL:
        logger.error("DATABASE_URL 未设置")
        sys.exit(1)

    factory = get_session_factory()
    async with factory() as db:
        data = await fetch_papers_with_chunks(db, args.paper_id)

    total_papers = len(data)
    total_chunks = sum(len(p["chunks"]) for p in data)

    print(f"\n{'=' * 50}")
    print(f"Elasticsearch 索引重建{'（dry-run）' if not args.apply else ''}")
    print(f"{'=' * 50}")
    print(f"论文数: {total_papers}")
    print(f"Chunk 总数: {total_chunks}")

    if not args.apply:
        print(f"\n使用 --apply 执行写入")
        await get_engine().dispose()
        return

    # 执行写入
    from core.elasticsearch_store import ensure_index, bulk_upsert_chunks, close_client

    if not await ensure_index():
        logger.error("Failed to ensure ES index")
        await get_engine().dispose()
        return

    success_total = 0
    fail_total = 0
    t0 = time.monotonic()

    for paper in data:
        paper_id = paper["paper_id"]
        title = paper["title"]
        chunks = paper["chunks"]

        if not chunks:
            continue

        # 分批写入
        for i in range(0, len(chunks), args.batch_size):
            batch = chunks[i:i + args.batch_size]
            docs = [
                {
                    "chunk_id": c["chunk_id"],
                    "paper_id": paper_id,
                    "title": title,
                    "section": c["section"],
                    "text": c["text"],
                }
                for c in batch
            ]
            result = await bulk_upsert_chunks(docs)
            count = result["success"]
            success_total += count
            fail_total += len(docs) - count
            for error in result["errors"]:
                logger.warning("ES bulk error: %s", error)

        logger.info("Indexed paper %s (%d chunks)", paper_id, len(chunks))

    elapsed = time.monotonic() - t0
    print(f"\n{'=' * 50}")
    print(f"完成: {success_total} 成功, {fail_total} 失败, {elapsed:.1f}s")
    print(f"{'=' * 50}")

    await close_client()
    await get_engine().dispose()


if __name__ == "__main__":
    asyncio.run(main())
