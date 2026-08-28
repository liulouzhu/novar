"""混合检索评测脚本 — 向量 + BM25 + RRF 对比评测

支持三种检索模式对比：
  1. vector-only  — 纯向量检索（Milvus cosine similarity）
  2. keyword-only — 纯 BM25 关键词检索（Elasticsearch）
  3. hybrid-rrf   — 向量 + BM25 混合检索（RRF 融合）

用法：
    # 三模式对比评测（需要数据库 + embedding + ES）
    python scripts/eval_hybrid_rag.py \
        --eval-set exported_papers/eval_set.json \
        --user-id YOUR_UUID

    # 只评测 hybrid 模式
    python scripts/eval_hybrid_rag.py \
        --eval-set exported_papers/eval_set.json \
        --user-id YOUR_UUID \
        --mode hybrid-rrf

    # 使用离线结果对比
    python scripts/eval_hybrid_rag.py \
        --eval-set exported_papers/eval_set.json \
        --vector-results exported_papers/vector_results.json \
        --keyword-results exported_papers/keyword_results.json \
        --hybrid-results exported_papers/hybrid_results.json

    # 输出详细报告 + 保存结果
    python scripts/eval_hybrid_rag.py \
        --eval-set exported_papers/eval_set.json \
        --user-id YOUR_UUID \
        --verbose \
        -o exported_papers/hybrid_eval_report.json

    # 指定 top_k
    python scripts/eval_hybrid_rag.py \
        --eval-set exported_papers/eval_set.json \
        --user-id YOUR_UUID \
        --top-k 20

环境变量：
    DATABASE_URL        — PostgreSQL 连接串
    DASHSCOPE_API_KEY   — 百炼 API key（embedding 必需）
    ELASTICSEARCH_URL   — ES 地址（BM25 必需）
    NOVARE_USER_ID      — 默认用户 ID
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
from unittest.mock import patch

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
logger = logging.getLogger("eval_hybrid")
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
    vector_candidates: int = 0
    keyword_candidates: int = 0
    fused_candidates: int = 0
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


# ── 指标计算 ──────────────────────────────────────────────────────────────

def compute_metrics(results: list[tuple[EvalCase, RetrievalResult]]) -> EvalMetrics:
    metrics = EvalMetrics()
    if not results:
        return metrics

    recalls = {1: 0, 3: 0, 5: 0, 10: 0}
    mrr_sum = 0.0
    hits = 0
    kw_scores = []
    total_latency = 0.0
    errors = 0

    for case, result in results:
        if result.error:
            errors += 1
            continue

        expected = set(case.expected_paper_ids)
        retrieved = result.retrieved_paper_ids

        for k in (1, 3, 5, 10):
            if expected & set(retrieved[:k]):
                recalls[k] += 1

        rr = 0.0
        for rank, pid in enumerate(retrieved, 1):
            if pid in expected:
                rr = 1.0 / rank
                break
        mrr_sum += rr
        if rr > 0:
            hits += 1

        if case.expected_keywords and result.retrieved_texts:
            combined = " ".join(result.retrieved_texts).lower()
            matched = sum(1 for kw in case.expected_keywords if kw.lower() in combined)
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
    total = sum(m.total for m in all_metrics.values())
    errors = sum(m.errors for m in all_metrics.values())
    if total == 0:
        return EvalMetrics()

    merged = EvalMetrics()
    for attr in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
                  "mrr", "hit_rate", "keyword_coverage", "avg_latency_ms"):
        weighted = sum(getattr(m, attr) * m.total for m in all_metrics.values())
        setattr(merged, attr, weighted / total if total else 0)
    merged.total = total
    merged.errors = errors
    return merged


# ── 检索执行器 ────────────────────────────────────────────────────────────

async def _run_vector_only(case: EvalCase, user_id: str, top_k: int, all_papers: bool = False) -> RetrievalResult:
    """纯向量检索：禁用 ES，只走 Milvus。"""
    from tools import rag_query as rq
    from tools.rag_query import handle_rag_query

    t0 = time.monotonic()
    try:
        args = {"question": case.query, "top_k": top_k}
        if case.filter_paper_id:
            args["paper_id"] = case.filter_paper_id

        # 禁用 ES 搜索
        with patch.object(rq, "_es_search", return_value=([], True, None)):
            raw = await handle_rag_query(args, user_id=None if all_papers else user_id,
                                         allow_unscoped=True if all_papers else None)

        latency = (time.monotonic() - t0) * 1000
        parsed = json.loads(raw)
        if not parsed.get("ok"):
            return RetrievalResult(query=case.query, retrieved_paper_ids=[],
                                   latency_ms=latency, error=parsed.get("error"))

        data = parsed["data"]
        return RetrievalResult(
            query=case.query,
            retrieved_paper_ids=[r["paper_id"] for r in data.get("results", [])],
            retrieved_texts=[r.get("text", "") for r in data.get("results", [])],
            search_method=data.get("search_method", "vector-only"),
            vector_candidates=data.get("vector_candidates", 0),
            latency_ms=latency,
        )
    except Exception as e:
        return RetrievalResult(query=case.query, retrieved_paper_ids=[],
                               latency_ms=(time.monotonic() - t0) * 1000, error=str(e))


async def _run_keyword_only(case: EvalCase, user_id: str, top_k: int, all_papers: bool = False) -> RetrievalResult:
    """纯 BM25 检索：禁用 Milvus，只走 ES。"""
    from tools import rag_query as rq
    from tools.rag_query import handle_rag_query, _resolve_paper_ids

    t0 = time.monotonic()
    try:
        args = {"question": case.query, "top_k": top_k}
        if case.filter_paper_id:
            args["paper_id"] = case.filter_paper_id

        from core.elasticsearch_store import search_chunks

        if all_papers:
            # 全库模式：获取所有 paper_ids
            from core.database import get_connection
            from web.backend.db.models import Paper
            from sqlalchemy import select
            async with get_connection() as conn:
                result = await conn.execute(select(Paper.id))
                stored_ids = [r[0] for r in result.all()]
            scoped = _resolve_paper_ids(args)
            if scoped:
                # 用 canonical 映射过滤
                from tools.rag_query import _build_canonical_to_stored_map, _resolve_to_stored_ids
                canonical_to_stored = _build_canonical_to_stored_map(set(stored_ids))
                stored_ids = sorted(_resolve_to_stored_ids(scoped, canonical_to_stored))
        else:
            from tools.rag_query import _get_user_paper_ids, _build_canonical_to_stored_map, _resolve_to_stored_ids
            allowed = await _get_user_paper_ids(user_id)
            scoped = _resolve_paper_ids(args)
            canonical_to_stored = _build_canonical_to_stored_map(allowed)
            stored_ids = sorted(_resolve_to_stored_ids(scoped, canonical_to_stored)) if scoped else sorted(allowed)

        es_result = await search_chunks(case.query, stored_ids, top_n=top_k)
        latency = (time.monotonic() - t0) * 1000

        return RetrievalResult(
            query=case.query,
            retrieved_paper_ids=[r["paper_id"] for r in es_result.hits],
            retrieved_texts=[r.get("text", "") for r in es_result.hits],
            search_method="keyword-only",
            keyword_candidates=len(es_result.hits),
            latency_ms=latency,
            error=None if es_result.available else es_result.error,
        )
    except Exception as e:
        return RetrievalResult(query=case.query, retrieved_paper_ids=[],
                               latency_ms=(time.monotonic() - t0) * 1000, error=str(e))


async def _run_hybrid_rrf(case: EvalCase, user_id: str, top_k: int, all_papers: bool = False) -> RetrievalResult:
    """混合检索：Milvus + ES + RRF。"""
    from tools.rag_query import handle_rag_query

    t0 = time.monotonic()
    try:
        args = {"question": case.query, "top_k": top_k}
        if case.filter_paper_id:
            args["paper_id"] = case.filter_paper_id

        raw = await handle_rag_query(args, user_id=None if all_papers else user_id,
                                     allow_unscoped=True if all_papers else None)

        latency = (time.monotonic() - t0) * 1000
        parsed = json.loads(raw)
        if not parsed.get("ok"):
            return RetrievalResult(query=case.query, retrieved_paper_ids=[],
                                   latency_ms=latency, error=parsed.get("error"))

        data = parsed["data"]
        return RetrievalResult(
            query=case.query,
            retrieved_paper_ids=[r["paper_id"] for r in data.get("results", [])],
            retrieved_texts=[r.get("text", "") for r in data.get("results", [])],
            search_method=data.get("search_method", "hybrid-rrf"),
            vector_candidates=data.get("vector_candidates", 0),
            keyword_candidates=data.get("keyword_candidates", 0),
            fused_candidates=data.get("fused_candidates", 0),
            latency_ms=latency,
        )
    except Exception as e:
        return RetrievalResult(query=case.query, retrieved_paper_ids=[],
                               latency_ms=(time.monotonic() - t0) * 1000, error=str(e))


_MODE_RUNNERS = {
    "vector-only": _run_vector_only,
    "keyword-only": _run_keyword_only,
    "hybrid-rrf": _run_hybrid_rrf,
}


# ── 并发执行 ──────────────────────────────────────────────────────────────

async def run_evaluation(
    eval_cases: list[EvalCase],
    user_id: str,
    mode: str,
    top_k: int,
    concurrency: int = 5,
    all_papers: bool = False,
) -> list[tuple[EvalCase, RetrievalResult]]:
    runner = _MODE_RUNNERS[mode]
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[int, tuple[EvalCase, RetrievalResult]] = {}

    async def _run_one(idx: int, case: EvalCase):
        async with semaphore:
            logger.info("[%d/%d] [%s] %s", idx + 1, len(eval_cases), mode, case.query[:70])
            result = await runner(case, user_id, top_k, all_papers=all_papers)
            if result.error:
                logger.warning("  ERROR: %s", result.error[:100])
            else:
                logger.info("  → %d hits, method=%s, %.0fms",
                           len(result.retrieved_paper_ids), result.search_method, result.latency_ms)
            results[idx] = (case, result)

    tasks = [_run_one(i, c) for i, c in enumerate(eval_cases)]
    await asyncio.gather(*tasks)
    return [(results[i][0], results[i][1]) for i in sorted(results.keys())]


# ── 报告输出 ──────────────────────────────────────────────────────────────

def print_comparison_table(mode_metrics: dict[str, EvalMetrics]):
    """打印三模式对比表。"""
    print("\n" + "=" * 90)
    print("RAG 混合检索评测对比报告")
    print("=" * 90)

    modes = list(mode_metrics.keys())
    header = f"{'指标':<20}" + "".join(f"{m:>18}" for m in modes)
    print(header)
    print("-" * 90)

    rows = [
        ("Recall@1",   "recall_at_1",   "{:.1%}"),
        ("Recall@3",   "recall_at_3",   "{:.1%}"),
        ("Recall@5",   "recall_at_5",   "{:.1%}"),
        ("Recall@10",  "recall_at_10",  "{:.1%}"),
        ("MRR",        "mrr",           "{:.3f}"),
        ("Hit Rate",   "hit_rate",      "{:.1%}"),
        ("KW Cover",   "keyword_coverage", "{:.1%}"),
        ("Avg Latency", "avg_latency_ms", "{:.0f}ms"),
        ("Errors",     "errors",        "{:d}"),
    ]

    for label, attr, fmt in rows:
        vals = []
        for m in modes:
            met = mode_metrics[m]
            v = getattr(met, attr)
            if "ms" in fmt:
                vals.append(fmt.format(v))
            elif "d" in fmt:
                vals.append(fmt.format(v))
            else:
                vals.append(fmt.format(v))
        print(f"{label:<20}" + "".join(f"{v:>18}" for v in vals))

    print("-" * 90)
    for m in modes:
        met = mode_metrics[m]
        print(f"  {m}: n={met.total}")


def print_category_detail(
    mode: str,
    category_results: dict[str, list[tuple[EvalCase, RetrievalResult]]],
):
    """打印单个模式的按类别详情。"""
    print(f"\n--- {mode} 按类别详情 ---")
    for cat, pairs in sorted(category_results.items()):
        metrics = compute_metrics(pairs)
        print(f"  {cat} (n={metrics.total}): "
              f"R@1={metrics.recall_at_1:.0%} R@5={metrics.recall_at_5:.0%} "
              f"MRR={metrics.mrr:.3f} Hit={metrics.hit_rate:.0%}")


def print_miss_analysis(
    mode: str,
    results: list[tuple[EvalCase, RetrievalResult]],
):
    """打印未命中分析。"""
    misses = [(c, r) for c, r in results
              if not r.error and not (set(c.expected_paper_ids) & set(r.retrieved_paper_ids))]
    if not misses:
        print(f"\n  {mode}: 无未命中 ✓")
        return

    print(f"\n  {mode}: {len(misses)} 个未命中")
    for case, result in misses[:5]:
        print(f"    [{case.category}] {case.query[:60]}...")
        print(f"      Expected: {case.expected_paper_ids}")
        print(f"      Got:      {result.retrieved_paper_ids[:3]}")


# ── 离线结果加载 ──────────────────────────────────────────────────────────

def load_offline_results(path: str) -> dict[str, RetrievalResult]:
    data = json.load(open(path, encoding="utf-8"))
    results = {}
    for item in data:
        results[item["query"]] = RetrievalResult(
            query=item["query"],
            retrieved_paper_ids=item.get("retrieved_paper_ids", []),
            retrieved_texts=item.get("retrieved_texts", []),
            search_method=item.get("search_method", ""),
            vector_candidates=item.get("vector_candidates", 0),
            keyword_candidates=item.get("keyword_candidates", 0),
            fused_candidates=item.get("fused_candidates", 0),
            latency_ms=item.get("latency_ms", 0),
            error=item.get("error"),
        )
    return results


def save_results(results: list[tuple[EvalCase, RetrievalResult]], path: str):
    data = []
    for case, r in results:
        data.append({
            "query": r.query,
            "category": case.category,
            "expected_paper_ids": case.expected_paper_ids,
            "retrieved_paper_ids": r.retrieved_paper_ids,
            "hit": bool(set(case.expected_paper_ids) & set(r.retrieved_paper_ids)),
            "search_method": r.search_method,
            "vector_candidates": r.vector_candidates,
            "keyword_candidates": r.keyword_candidates,
            "fused_candidates": r.fused_candidates,
            "latency_ms": r.latency_ms,
            "error": r.error,
        })
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", path)


# ── 主流程 ────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="RAG 混合检索评测")
    parser.add_argument("--eval-set", required=True, help="评测集 JSON")
    parser.add_argument("--user-id", default=None, help="用户 ID")
    parser.add_argument("--all-papers", action="store_true",
                        help="全库评测模式，不限用户（不需要 --user-id）")
    parser.add_argument("--mode", default="all",
                        choices=["all", "vector-only", "keyword-only", "hybrid-rrf"],
                        help="评测模式（默认 all = 三模式对比）")
    parser.add_argument("--top-k", type=int, default=10, help="检索 top_k（默认 10）")
    parser.add_argument("--concurrency", type=int, default=3, help="并发数")
    parser.add_argument("--category", default=None, help="只评测指定类别")
    parser.add_argument("--vector-results", default=None, help="离线向量结果 JSON")
    parser.add_argument("--keyword-results", default=None, help="离线 BM25 结果 JSON")
    parser.add_argument("--hybrid-results", default=None, help="离线 hybrid 结果 JSON")
    parser.add_argument("--output", "-o", default=None, help="保存对比报告 JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    args = parser.parse_args()

    # 加载评测集
    eval_data = json.load(open(args.eval_set, encoding="utf-8"))
    eval_cases = [EvalCase(**item) for item in eval_data]
    if args.category:
        eval_cases = [c for c in eval_cases if c.category == args.category]
    logger.info("Loaded %d eval cases", len(eval_cases))

    # 确定用户 ID 和全库模式
    all_papers = args.all_papers
    user_id = args.user_id or os.environ.get("NOVARE_USER_ID")
    if not all_papers and not user_id:
        logger.error("需要 --user-id 或 --all-papers")
        sys.exit(1)

    # 确定要运行的模式
    modes = ["vector-only", "keyword-only", "hybrid-rrf"] if args.mode == "all" else [args.mode]

    # ── 执行评测 ──
    mode_results: dict[str, list[tuple[EvalCase, RetrievalResult]]] = {}
    mode_metrics: dict[str, EvalMetrics] = {}

    for mode in modes:
        logger.info(f"\n{'='*50} Evaluating: {mode} {'='*50}")

        if mode == "vector-only" and args.vector_results:
            offline = load_offline_results(args.vector_results)
            pairs = [(c, offline.get(c.query, RetrievalResult(query=c.query, retrieved_paper_ids=[], error="not found")))
                     for c in eval_cases]
        elif mode == "keyword-only" and args.keyword_results:
            offline = load_offline_results(args.keyword_results)
            pairs = [(c, offline.get(c.query, RetrievalResult(query=c.query, retrieved_paper_ids=[], error="not found")))
                     for c in eval_cases]
        elif mode == "hybrid-rrf" and args.hybrid_results:
            offline = load_offline_results(args.hybrid_results)
            pairs = [(c, offline.get(c.query, RetrievalResult(query=c.query, retrieved_paper_ids=[], error="not found")))
                     for c in eval_cases]
        else:
            pairs = await run_evaluation(eval_cases, user_id or "", mode, args.top_k,
                                         args.concurrency, all_papers=all_papers)

        mode_results[mode] = pairs

        # 计算按类别指标
        cat_results: dict[str, list] = {}
        for c, r in pairs:
            cat = c.category or "unknown"
            cat_results.setdefault(cat, []).append((c, r))
        cat_metrics = {cat: compute_metrics(ps) for cat, ps in cat_results.items()}
        mode_metrics[mode] = merge_metrics(cat_metrics)

        # 打印按类别详情
        if args.verbose:
            print_category_detail(mode, cat_results)
            print_miss_analysis(mode, pairs)

    # ── 输出对比报告 ──
    print_comparison_table(mode_metrics)

    # ── 保存结果 ──
    if args.output:
        report = {
            "eval_set": args.eval_set,
            "top_k": args.top_k,
            "modes": {},
        }
        for mode in modes:
            pairs = mode_results[mode]
            report["modes"][mode] = {
                "metrics": {
                    "recall_at_1": mode_metrics[mode].recall_at_1,
                    "recall_at_3": mode_metrics[mode].recall_at_3,
                    "recall_at_5": mode_metrics[mode].recall_at_5,
                    "recall_at_10": mode_metrics[mode].recall_at_10,
                    "mrr": mode_metrics[mode].mrr,
                    "hit_rate": mode_metrics[mode].hit_rate,
                    "keyword_coverage": mode_metrics[mode].keyword_coverage,
                    "avg_latency_ms": mode_metrics[mode].avg_latency_ms,
                    "total": mode_metrics[mode].total,
                    "errors": mode_metrics[mode].errors,
                },
                "results": [
                    {
                        "query": c.query,
                        "category": c.category,
                        "expected_paper_ids": c.expected_paper_ids,
                        "retrieved_paper_ids": r.retrieved_paper_ids,
                        "hit": bool(set(c.expected_paper_ids) & set(r.retrieved_paper_ids)),
                        "search_method": r.search_method,
                        "latency_ms": r.latency_ms,
                        "error": r.error,
                    }
                    for c, r in pairs
                ],
            }
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Report saved to %s", args.output)

    # ── 保存各模式独立结果（供后续离线对比）──
    for mode in modes:
        out_path = Path(args.output).parent / f"{mode.replace('-', '_')}_results.json" if args.output else None
        if out_path:
            save_results(mode_results[mode], str(out_path))


if __name__ == "__main__":
    asyncio.run(main())
