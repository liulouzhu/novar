"""RAG 检索层评测脚本

基于 eval_set.json 评测检索质量，覆盖：
- Recall@K (K=1,3,5,10)
- MRR (Mean Reciprocal Rank)
- Hit Rate@K
- 按类别分组统计
- paper_id 过滤正确性

用法：
    # 实时评测（需要数据库 + embedding provider）
    python scripts/eval_rag.py --eval-set exported_papers/eval_set.json

    # 离线评测（使用预存的检索结果）
    python scripts/eval_rag.py --eval-set exported_papers/eval_set.json --results results.json

    # 只评测 paper_id 过滤
    python scripts/eval_rag.py --eval-set exported_papers/eval_set.json --category paper_id_filter

    # 输出详细报告
    python scripts/eval_rag.py --eval-set exported_papers/eval_set.json --verbose
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MCP_ROOT = PROJECT_ROOT / "mcp-server"
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("eval_rag")
logger.setLevel(logging.INFO)


# ── 数据结构 ──────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    query: str
    expected_paper_ids: list[str]
    expected_keywords: list[str] = field(default_factory=list)
    category: str = ""
    source_paper_id: str = ""
    filter_paper_id: str | None = None
    source_chunk_id: int | None = None


@dataclass
class RetrievalResult:
    query: str
    retrieved_paper_ids: list[str]
    retrieved_texts: list[str] = field(default_factory=list)
    search_method: str = ""
    latency_ms: float = 0
    error: str | None = None


@dataclass
class EvalMetrics:
    recall_at_1: float = 0
    recall_at_3: float = 0
    recall_at_5: float = 0
    recall_at_10: float = 0
    mrr: float = 0
    hit_rate: float = 0
    keyword_coverage: float = 0
    total: int = 0
    errors: int = 0
    avg_latency_ms: float = 0


# ── 评测指标计算 ──────────────────────────────────────────────────────────

def compute_metrics(results: list[tuple[EvalCase, RetrievalResult]]) -> EvalMetrics:
    """计算检索评测指标。"""
    metrics = EvalMetrics()
    if not results:
        return metrics

    recalls = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr_sum = 0
    hits = 0
    kw_scores = []
    total_latency = 0
    errors = 0

    for case, result in results:
        if result.error:
            errors += 1
            continue

        expected = set(case.expected_paper_ids)
        retrieved = result.retrieved_paper_ids

        # Recall@K
        for k in (1, 3, 5, 10):
            retrieved_at_k = set(retrieved[:k])
            if expected & retrieved_at_k:
                recalls[k] += 1

        # MRR — 第一个相关结果的排名倒数
        rr = 0
        for rank, pid in enumerate(retrieved, 1):
            if pid in expected:
                rr = 1.0 / rank
                break
        mrr_sum += rr

        # Hit Rate — 至少有一个相关结果
        if rr > 0:
            hits += 1

        # Keyword coverage
        if case.expected_keywords and result.retrieved_texts:
            combined_text = " ".join(result.retrieved_texts).lower()
            matched = sum(1 for kw in case.expected_keywords if kw.lower() in combined_text)
            kw_scores.append(matched / len(case.expected_keywords))

        total_latency += result.latency_ms

    n = len(results) - errors
    if n > 0:
        metrics.recall_at_1 = recalls[1] / n
        metrics.recall_at_3 = recalls[3] / n
        metrics.recall_at_5 = recalls[5] / n
        metrics.recall_at_10 = recalls[10] / n
        metrics.mrr = mrr_sum / n
        metrics.hit_rate = hits / n
        metrics.keyword_coverage = sum(kw_scores) / len(kw_scores) if kw_scores else 0
        metrics.avg_latency_ms = total_latency / n

    metrics.total = len(results)
    metrics.errors = errors
    return metrics


def merge_metrics(all_metrics: dict[str, EvalMetrics]) -> EvalMetrics:
    """合并多个类别的指标为总体指标。"""
    total = sum(m.total for m in all_metrics.values())
    errors = sum(m.errors for m in all_metrics.values())
    if total == 0:
        return EvalMetrics()

    # 加权平均（按 total 数量加权）
    merged = EvalMetrics()
    for attr in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
                  "mrr", "hit_rate", "keyword_coverage", "avg_latency_ms"):
        weighted_sum = sum(getattr(m, attr) * m.total for m in all_metrics.values())
        setattr(merged, attr, weighted_sum / total if total else 0)

    merged.total = total
    merged.errors = errors
    return merged


# ── 实时检索 ──────────────────────────────────────────────────────────────

async def run_live_retrieval(case: EvalCase, user_id: str) -> RetrievalResult:
    """通过 rag_query 执行实时检索。"""
    t0 = time.monotonic()

    try:
        from tools.rag_query import handle_rag_query

        args = {"question": case.query, "top_k": 10}
        if case.filter_paper_id:
            args["paper_id"] = case.filter_paper_id

        raw_result = await handle_rag_query(args, user_id=user_id)
        latency = (time.monotonic() - t0) * 1000

        result_json = json.loads(raw_result)
        if not result_json.get("ok"):
            return RetrievalResult(
                query=case.query,
                retrieved_paper_ids=[],
                latency_ms=latency,
                error=result_json.get("error", "unknown"),
            )

        results_data = result_json.get("data", {}).get("results", [])
        return RetrievalResult(
            query=case.query,
            retrieved_paper_ids=[r["paper_id"] for r in results_data],
            retrieved_texts=[r.get("text", "") for r in results_data],
            search_method=result_json.get("data", {}).get("search_method", ""),
            latency_ms=latency,
        )
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        return RetrievalResult(
            query=case.query,
            retrieved_paper_ids=[],
            latency_ms=latency,
            error=str(e),
        )


async def run_live_evaluation(
    eval_cases: list[EvalCase],
    user_id: str,
    concurrency: int = 5,
) -> dict[str, tuple[EvalCase, RetrievalResult]]:
    """并发执行所有评测用例的实时检索。"""
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[str, tuple[EvalCase, RetrievalResult]] = {}

    async def _run_one(idx: int, case: EvalCase):
        async with semaphore:
            logger.info("[%d/%d] %s: %s", idx + 1, len(eval_cases), case.category, case.query[:80])
            result = await run_live_retrieval(case, user_id)
            if result.error:
                logger.warning("  ERROR: %s", result.error[:100])
            else:
                logger.info("  → %d results, method=%s, %.0fms",
                           len(result.retrieved_paper_ids), result.search_method, result.latency_ms)
            results[idx] = (case, result)

    tasks = [_run_one(i, c) for i, c in enumerate(eval_cases)]
    await asyncio.gather(*tasks)
    return results


# ── 离线评测 ──────────────────────────────────────────────────────────────

def load_offline_results(path: str) -> dict[str, RetrievalResult]:
    """加载预存的检索结果。"""
    data = json.load(open(path, encoding="utf-8"))
    results = {}
    for item in data:
        results[item["query"]] = RetrievalResult(
            query=item["query"],
            retrieved_paper_ids=item.get("retrieved_paper_ids", []),
            retrieved_texts=item.get("retrieved_texts", []),
            search_method=item.get("search_method", ""),
            latency_ms=item.get("latency_ms", 0),
            error=item.get("error"),
        )
    return results


# ── 报告输出 ──────────────────────────────────────────────────────────────

def print_report(
    category_metrics: dict[str, EvalMetrics],
    all_results: list[tuple[EvalCase, RetrievalResult]],
    verbose: bool = False,
):
    """打印评测报告。"""
    print("\n" + "=" * 70)
    print("RAG 检索层评测报告")
    print("=" * 70)

    # 按类别输出
    for cat, metrics in sorted(category_metrics.items()):
        print(f"\n--- {cat} (n={metrics.total}, errors={metrics.errors}) ---")
        print(f"  Recall@1:   {metrics.recall_at_1:.1%}")
        print(f"  Recall@3:   {metrics.recall_at_3:.1%}")
        print(f"  Recall@5:   {metrics.recall_at_5:.1%}")
        print(f"  Recall@10:  {metrics.recall_at_10:.1%}")
        print(f"  MRR:        {metrics.mrr:.3f}")
        print(f"  Hit Rate:   {metrics.hit_rate:.1%}")
        if metrics.keyword_coverage > 0:
            print(f"  KW Cover:   {metrics.keyword_coverage:.1%}")
        print(f"  Avg Latency: {metrics.avg_latency_ms:.0f}ms")

    # 总体指标
    merged = merge_metrics(category_metrics)
    print(f"\n{'=' * 70}")
    print(f"总体 (n={merged.total}, errors={merged.errors})")
    print(f"{'=' * 70}")
    print(f"  Recall@1:   {merged.recall_at_1:.1%}")
    print(f"  Recall@3:   {merged.recall_at_3:.1%}")
    print(f"  Recall@5:   {merged.recall_at_5:.1%}")
    print(f"  Recall@10:  {merged.recall_at_10:.1%}")
    print(f"  MRR:        {merged.mrr:.3f}")
    print(f"  Hit Rate:   {merged.hit_rate:.1%}")
    print(f"  Avg Latency: {merged.avg_latency_ms:.0f}ms")

    # 失败用例
    failures = [(c, r) for c, r in all_results if r.error]
    if failures:
        print(f"\n--- Errors ({len(failures)}) ---")
        for case, result in failures[:10]:
            print(f"  [{case.category}] {case.query[:60]}...")
            print(f"    Error: {result.error[:120]}")

    # 详细输出：每个失败的检索
    if verbose:
        misses = [(c, r) for c, r in all_results
                  if not r.error and not (set(c.expected_paper_ids) & set(r.retrieved_paper_ids))]
        if misses:
            print(f"\n--- Misses ({len(misses)}) ---")
            for case, result in misses[:20]:
                print(f"  [{case.category}] {case.query[:70]}...")
                print(f"    Expected: {case.expected_paper_ids}")
                print(f"    Got:      {result.retrieved_paper_ids[:5]}")
                if result.retrieved_texts:
                    print(f"    Top text: {result.retrieved_texts[0][:100]}...")
                print()


def save_results(
    all_results: list[tuple[EvalCase, RetrievalResult]],
    output_path: str,
):
    """保存评测结果到 JSON。"""
    data = []
    for case, result in all_results:
        data.append({
            "query": case.query,
            "category": case.category,
            "expected_paper_ids": case.expected_paper_ids,
            "retrieved_paper_ids": result.retrieved_paper_ids,
            "hit": bool(set(case.expected_paper_ids) & set(result.retrieved_paper_ids)),
            "search_method": result.search_method,
            "latency_ms": result.latency_ms,
            "error": result.error,
        })
    Path(output_path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Results saved to %s", output_path)


# ── 主流程 ────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="RAG 检索层评测")
    parser.add_argument("--eval-set", required=True, help="评测集 JSON 文件路径")
    parser.add_argument("--results", default=None, help="预存检索结果 JSON（离线模式）")
    parser.add_argument("--user-id", default=None, help="用户 ID（实时模式必需）")
    parser.add_argument("--category", default=None, help="只评测指定类别")
    parser.add_argument("--top-k", type=int, default=10, help="检索 top_k（默认 10）")
    parser.add_argument("--concurrency", type=int, default=5, help="并发数（默认 5）")
    parser.add_argument("--output", "-o", default=None, help="保存评测结果到 JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出详细报告")
    args = parser.parse_args()

    # 加载评测集
    eval_data = json.load(open(args.eval_set, encoding="utf-8"))
    eval_cases = [EvalCase(**item) for item in eval_data]

    if args.category:
        eval_cases = [c for c in eval_cases if c.category == args.category]

    logger.info("Loaded %d eval cases", len(eval_cases))

    # 执行检索
    all_results: list[tuple[EvalCase, RetrievalResult]] = []

    if args.results:
        # 离线模式
        logger.info("Offline mode: loading results from %s", args.results)
        offline = load_offline_results(args.results)
        for case in eval_cases:
            result = offline.get(case.query)
            if result is None:
                result = RetrievalResult(query=case.query, retrieved_paper_ids=[], error="not in results file")
            all_results.append((case, result))
    else:
        # 实时模式
        user_id = args.user_id
        if not user_id:
            user_id = os.environ.get("NOVARE_USER_ID")
        if not user_id:
            logger.error("实时模式需要 --user-id 或 NOVARE_USER_ID 环境变量")
            sys.exit(1)

        # 设置 top_k（通过环境变量传递给 rag_query）
        from tools import rag_query as rq_mod
        original_validate = rq_mod._validate_top_k

        raw_results = await run_live_evaluation(eval_cases, user_id, concurrency=args.concurrency)
        all_results = [(raw_results[i][0], raw_results[i][1]) for i in sorted(raw_results.keys())]

    # 按类别分组
    category_results: dict[str, list[tuple[EvalCase, RetrievalResult]]] = {}
    for case, result in all_results:
        cat = case.category or "unknown"
        if cat not in category_results:
            category_results[cat] = []
        category_results[cat].append((case, result))

    # 计算指标
    category_metrics = {}
    for cat, pairs in category_results.items():
        category_metrics[cat] = compute_metrics(pairs)

    # 输出报告
    print_report(category_metrics, all_results, verbose=args.verbose)

    # 保存结果
    if args.output:
        save_results(all_results, args.output)


if __name__ == "__main__":
    asyncio.run(main())
