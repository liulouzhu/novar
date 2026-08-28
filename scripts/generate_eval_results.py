"""生成离线评测结果

对 eval_set.json 中的每个 query 执行检索，保存为 results.json。

两种模式：
  1. --user-id   按用户隔离检索（与生产环境一致）
  2. --all-papers 全库检索，不限用户（用于评测检索质量基线）

用法：
    # 全库评测（不需要 user_id）
    python scripts/generate_eval_results.py \
        --eval-set exported_papers/eval_set.json \
        --all-papers \
        -o exported_papers/results.json

    # 用户隔离评测
    python scripts/generate_eval_results.py \
        --eval-set exported_papers/eval_set.json \
        --user-id YOUR_UUID \
        -o exported_papers/results.json
"""

import argparse
import asyncio
import json
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("gen_results")


async def run_one(
    query: str,
    filter_paper_id: str | None,
    user_id: str | None,
    all_papers: bool,
) -> dict:
    """执行单条检索。

    all_papers=True 时：
    - 不传 user_id，允许无作用域全库检索
    - paper_id filter 仍然生效（用于 paper_id_filter 类别测试）
    """
    from tools import rag_query as rq
    from tools.rag_query import handle_rag_query

    args = {"question": query, "top_k": 10}
    if filter_paper_id:
        args["paper_id"] = filter_paper_id

    t0 = time.monotonic()
    try:
        if all_papers:
            # 临时开启无作用域检索
            old_val = rq.ALLOW_UNSCOPED
            rq.ALLOW_UNSCOPED = True
            try:
                raw = await handle_rag_query(args, user_id=None)
            finally:
                rq.ALLOW_UNSCOPED = old_val
        else:
            raw = await handle_rag_query(args, user_id=user_id)

        latency = (time.monotonic() - t0) * 1000
        parsed = json.loads(raw)
        if parsed.get("ok"):
            results = parsed["data"]["results"]
            return {
                "query": query,
                "retrieved_paper_ids": [r["paper_id"] for r in results],
                "retrieved_texts": [r.get("text", "") for r in results],
                "search_method": parsed["data"].get("search_method", ""),
                "latency_ms": latency,
            }
        else:
            return {
                "query": query,
                "retrieved_paper_ids": [],
                "retrieved_texts": [],
                "error": parsed.get("error", "unknown"),
                "latency_ms": latency,
            }
    except Exception as e:
        return {
            "query": query,
            "retrieved_paper_ids": [],
            "error": str(e),
            "latency_ms": (time.monotonic() - t0) * 1000,
        }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--user-id", default=None, help="用户 ID（user-scoped 模式）")
    parser.add_argument("--all-papers", action="store_true",
                        help="全库检索模式，不限用户")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    if not args.all_papers and not args.user_id:
        args.user_id = os.environ.get("NOVARE_USER_ID")
        if not args.user_id:
            logger.error("需要 --user-id 或 --all-papers")
            sys.exit(1)

    eval_data = json.load(open(args.eval_set, encoding="utf-8"))
    logger.info("Loaded %d cases, mode=%s",
                len(eval_data), "all-papers" if args.all_papers else f"user={args.user_id[:8]}...")

    # paper_id_filter 类别在全库模式下也传 paper_id（验证过滤逻辑）
    # 其他类别不传 paper_id（验证纯语义检索）

    sem = asyncio.Semaphore(args.concurrency)
    results = []

    async def _run(idx, case):
        async with sem:
            logger.info("[%d/%d] %s | %s",
                        idx + 1, len(eval_data), case["category"], case["query"][:60])
            r = await run_one(
                case["query"],
                case.get("filter_paper_id"),
                args.user_id,
                args.all_papers,
            )
            if r.get("error"):
                logger.warning("  ERROR: %s", r["error"][:100])
            else:
                logger.info("  → %d hits, method=%s, %.0fms",
                            len(r["retrieved_paper_ids"]),
                            r.get("search_method", ""),
                            r.get("latency_ms", 0))
            results.append(r)

    await asyncio.gather(*[_run(i, c) for i, c in enumerate(eval_data)])

    # 按原始顺序排列
    query_order = {c["query"]: i for i, c in enumerate(eval_data)}
    ordered = sorted(results, key=lambda r: query_order.get(r["query"], 999))

    Path(args.output).write_text(
        json.dumps(ordered, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    hits = sum(1 for r in ordered if r.get("retrieved_paper_ids"))
    errors = sum(1 for r in ordered if r.get("error"))
    logger.info("Done: %d results, %d with hits, %d errors → %s",
                len(ordered), hits, errors, args.output)


if __name__ == "__main__":
    asyncio.run(main())
