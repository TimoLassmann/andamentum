"""End-to-end accuracy benchmark for the semantic provider router.

Runs all 200 labeled queries through :func:`rank_providers` using a real
Ollama embedding backend, computes top-1 accuracy, top-3 recall, MRR, and
per-category breakdowns, and writes a timestamped JSON report to
``benchmarks/epistemic/routing_runs/``.

This is the guard-rail for tracking routing quality over time. When new
providers are added or descriptions are tuned, rerun and diff against
prior runs.

Gating
------
The benchmark requires a live Ollama backend and is therefore **not** part
of default CI. It is marked with both ``@pytest.mark.ollama`` and
``@pytest.mark.benchmark``. To run:

.. code-block:: console

    uv run pytest -m benchmark src/andamentum/epistemic/tests/test_provider_routing_benchmark.py -s

Metrics
-------
- **top_1_strict** — fraction of queries whose exact ``primary`` provider
  is ranked first. Pessimistic for biomedical questions where several
  providers are legitimately correct, because the ``primary`` label is
  arbitrary among co-equal answers.
- **top_1_permissive** — fraction of queries whose rank-1 provider is in
  the ``acceptable`` set. The honest top-1 for multi-provider domains.
- **top_3_recall** — fraction of queries whose ``acceptable`` set has at
  least one member in the top 3 ranked providers. This is the headline
  metric: it mirrors how :func:`select_providers` is actually consumed by
  ``PlanTaskOperation`` (which takes top-K).
- **mrr** — mean reciprocal rank of the ``primary`` provider across all
  queries. Rewards rankings where the primary is close to the top even
  when not #1.
- **per_category_*** — same metrics computed per category so quality
  regressions can be traced to a specific provider's description.
- **confusion_pairs** — when ``primary`` is missed at rank 1 AND rank-1
  is not in the acceptable set, which provider took its place. Tells you
  which description pairs the embedder genuinely conflates (not cases
  where any acceptable answer was still chosen).
- **score_distribution** — mean / stdev / median / min / max of the
  winning provider's cosine similarity per category. Helps you calibrate
  ``min_score``.

Pass threshold
--------------
The pytest assertion gate is set conservatively at ``top_3_recall >= 0.85``.
The real target is higher; you can tighten this after the first calibration
run tells you what the baseline actually is.
"""

from __future__ import annotations

import json
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from andamentum.epistemic.provider_routing import (
    ProviderScore,
    _clear_cache,
    rank_providers,
)

from .routing_benchmark_queries import QUERIES, BenchmarkQuery


# ── Config ───────────────────────────────────────────────────────────────────

DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "ANDAMENTUM_EMBEDDING_MODEL", "embeddinggemma:latest"
)

REPO_ROOT = Path(__file__).resolve().parents[5]
BENCHMARK_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "epistemic" / "routing_runs"

PASS_THRESHOLD_TOP_3_RECALL = 0.95


# ── Result structures ───────────────────────────────────────────────────────


@dataclass
class QueryResult:
    """Per-query evaluation record."""

    query: str
    primary: str
    acceptable: list[str]
    category: str
    ranking: list[tuple[str, float]]  # (provider, score) ordered desc
    primary_rank: int | None  # 1-indexed position of primary in ranking
    top_3_hit: bool  # any acceptable provider in top 3
    top_1_strict: bool  # primary is rank 1 (exact match)
    top_1_permissive: bool  # rank 1 is in the acceptable set


@dataclass
class BenchmarkMetrics:
    """Aggregate metrics for one benchmark run."""

    total_queries: int
    top_1_strict: float
    top_1_permissive: float
    top_3_recall: float
    mrr: float
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_pairs: list[tuple[str, str, int]] = field(default_factory=list)
    score_distribution: dict[str, dict[str, float]] = field(default_factory=dict)
    embedding_model: str = ""
    timestamp: str = ""


# ── Evaluation ───────────────────────────────────────────────────────────────


def _evaluate_query(
    scores: list[ProviderScore],
    entry: BenchmarkQuery,
) -> QueryResult:
    """Score one query's ranking against its ground truth labels."""
    ordered = [(s.name, s.score) for s in scores]
    name_to_rank = {name: i + 1 for i, (name, _) in enumerate(ordered)}
    primary_rank = name_to_rank.get(entry["primary"])

    top_3_names = {name for name, _ in ordered[:3]}
    top_3_hit = bool(entry["acceptable"] & top_3_names)

    rank_1 = ordered[0][0] if ordered else None
    top_1_strict = rank_1 == entry["primary"]
    top_1_permissive = rank_1 in entry["acceptable"] if rank_1 else False

    return QueryResult(
        query=entry["query"],
        primary=entry["primary"],
        acceptable=sorted(entry["acceptable"]),
        category=entry["category"],
        ranking=ordered,
        primary_rank=primary_rank,
        top_3_hit=top_3_hit,
        top_1_strict=top_1_strict,
        top_1_permissive=top_1_permissive,
    )


def _aggregate_metrics(
    results: list[QueryResult],
    embedding_model: str,
) -> BenchmarkMetrics:
    """Reduce per-query results into overall + per-category metrics."""
    n = len(results)
    top_1_strict = sum(1 for r in results if r.top_1_strict) / n
    top_1_permissive = sum(1 for r in results if r.top_1_permissive) / n
    top_3 = sum(1 for r in results if r.top_3_hit) / n

    # MRR: mean of 1 / rank_of_primary, counting misses as 0
    reciprocals = []
    for r in results:
        if r.primary_rank is None:
            reciprocals.append(0.0)
        else:
            reciprocals.append(1.0 / r.primary_rank)
    mrr = sum(reciprocals) / n

    # Per-category breakdown
    by_category: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)

    per_category: dict[str, dict[str, float]] = {}
    for cat, items in sorted(by_category.items()):
        cat_n = len(items)
        per_category[cat] = {
            "count": cat_n,
            "top_1_strict": sum(1 for x in items if x.top_1_strict) / cat_n,
            "top_1_permissive": sum(1 for x in items if x.top_1_permissive) / cat_n,
            "top_3_recall": sum(1 for x in items if x.top_3_hit) / cat_n,
            "mrr": sum(
                (1.0 / x.primary_rank) if x.primary_rank else 0.0 for x in items
            )
            / cat_n,
        }

    # Confusion: only count cases where rank-1 is not in the acceptable set.
    # When rank-1 is different from primary but still acceptable, the router
    # did the right thing for a multi-provider question.
    confusion_counter: Counter[tuple[str, str]] = Counter()
    for r in results:
        if not r.top_1_permissive and r.ranking:
            actual_top = r.ranking[0][0]
            confusion_counter[(r.primary, actual_top)] += 1
    confusion_pairs = [
        (primary, actual, count)
        for (primary, actual), count in confusion_counter.most_common()
    ]

    # Score distribution per category
    score_distribution: dict[str, dict[str, float]] = {}
    for cat, items in sorted(by_category.items()):
        winning_scores = [x.ranking[0][1] for x in items if x.ranking]
        if winning_scores:
            score_distribution[cat] = {
                "mean": statistics.mean(winning_scores),
                "stdev": statistics.stdev(winning_scores)
                if len(winning_scores) > 1
                else 0.0,
                "median": statistics.median(winning_scores),
                "min": min(winning_scores),
                "max": max(winning_scores),
            }

    return BenchmarkMetrics(
        total_queries=n,
        top_1_strict=top_1_strict,
        top_1_permissive=top_1_permissive,
        top_3_recall=top_3,
        mrr=mrr,
        per_category=per_category,
        confusion_pairs=confusion_pairs,
        score_distribution=score_distribution,
        embedding_model=embedding_model,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _save_report(
    metrics: BenchmarkMetrics,
    results: list[QueryResult],
    output_dir: Path,
) -> Path:
    """Write a timestamped JSON report alongside the full per-query trace."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = output_dir / f"routing_benchmark_{stamp}.json"

    payload: dict[str, Any] = {
        "metrics": asdict(metrics),
        "per_query": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return path


def _print_report(metrics: BenchmarkMetrics) -> None:
    """Print a human-readable summary to stdout."""
    print()
    print("=" * 78)
    print(
        f"Provider routing benchmark  ({metrics.total_queries} queries, "
        f"model={metrics.embedding_model})"
    )
    print("=" * 78)
    print(f"Top-3 recall:         {metrics.top_3_recall:.1%}  (HEADLINE — mirrors PlanTaskOperation)")
    print(f"Top-1 permissive:     {metrics.top_1_permissive:.1%}  (rank-1 in acceptable set)")
    print(f"Top-1 strict:         {metrics.top_1_strict:.1%}  (rank-1 == primary label)")
    print(f"MRR (primary):        {metrics.mrr:.3f}")
    print()
    print("Per-category:")
    print(
        f"  {'category':<16} {'n':>4} {'top1s':>7} {'top1p':>7} "
        f"{'top3':>7} {'MRR':>7} {'score μ':>8} {'score σ':>8}"
    )
    for cat, stats in metrics.per_category.items():
        sd = metrics.score_distribution.get(cat, {})
        print(
            f"  {cat:<16} "
            f"{int(stats['count']):>4} "
            f"{stats['top_1_strict']:>7.1%} "
            f"{stats['top_1_permissive']:>7.1%} "
            f"{stats['top_3_recall']:>7.1%} "
            f"{stats['mrr']:>7.3f} "
            f"{sd.get('mean', 0):>8.3f} "
            f"{sd.get('stdev', 0):>8.3f}"
        )
    print()
    if metrics.confusion_pairs:
        print("Top genuine confusions (rank-1 outside acceptable set, expected → actual, count):")
        for primary, actual, count in metrics.confusion_pairs[:10]:
            print(f"  {primary:<16} → {actual:<16} {count}")
    else:
        print("No genuine confusions — every query's rank-1 is in its acceptable set.")
    print("=" * 78)
    print()


# ── The actual test ─────────────────────────────────────────────────────────


@pytest.mark.ollama
@pytest.mark.benchmark
async def test_routing_accuracy_benchmark():
    """Run all 200 queries through the router and assert minimum accuracy.

    Saves a detailed JSON report regardless of pass/fail so regressions
    can be inspected.
    """
    _clear_cache()  # start fresh so cache behavior is comparable across runs

    results: list[QueryResult] = []
    for entry in QUERIES:
        scores = await rank_providers(
            entry["query"],
            embedding_model=DEFAULT_EMBEDDING_MODEL,
        )
        results.append(_evaluate_query(scores, entry))

    metrics = _aggregate_metrics(results, DEFAULT_EMBEDDING_MODEL)
    report_path = _save_report(metrics, results, BENCHMARK_OUTPUT_DIR)
    _print_report(metrics)
    print(f"Full report written to: {report_path}")

    # Hard gate for CI / regression detection
    assert metrics.top_3_recall >= PASS_THRESHOLD_TOP_3_RECALL, (
        f"top_3_recall {metrics.top_3_recall:.1%} below threshold "
        f"{PASS_THRESHOLD_TOP_3_RECALL:.1%} — provider routing has regressed. "
        f"See {report_path} for details."
    )
