"""Run the chunker against a benchmark case and produce a CaseRun."""

from __future__ import annotations

import time

from andamentum.chunker.extractor import ExecutorFn, extract_units
from andamentum.chunker.types import ChunkingFailedError

from .metrics import boundary_f1, granularity_ratio
from .types import BenchmarkCase, CaseRun, Metrics


async def run_case(
    case: BenchmarkCase,
    *,
    primary_executor: ExecutorFn,
    backup_executors: list[ExecutorFn] | None = None,
    model_label: str = "unknown",
    window_size: int = 10_000,
    lookahead: int = 4_000,
) -> CaseRun:
    """Run the chunker on a case and compute metrics.

    On `ChunkingFailedError`, returns a CaseRun with `error` set and `metrics=None`.
    """
    start = time.monotonic()
    try:
        result = await extract_units(
            case.source,
            primary_executor=primary_executor,
            backup_executors=backup_executors or [],
            window_size=window_size,
            lookahead=lookahead,
            domain=case.domain,
        )
    except ChunkingFailedError as exc:
        return CaseRun(case=case, metrics=None, model=model_label, error=str(exc))

    elapsed = time.monotonic() - start

    # Boundary positions = unique starts and ends of truth + predicted units
    truth_boundaries = sorted(
        {u.start_offset for u in case.truth.units}
        | {u.end_offset for u in case.truth.units}
    )
    predicted_boundaries = sorted(
        {u.source_start for u in result.units} | {u.source_end for u in result.units}
    )

    p, r, f = boundary_f1(
        predicted_boundaries,
        truth_boundaries,
        tolerance=case.boundary_tolerance_chars,
    )

    method_counts = {"exact": 0, "whitespace_normalised": 0, "fuzzy": 0}
    for u in result.units:
        if u.anchor_match_method in method_counts:
            method_counts[u.anchor_match_method] += 1

    fragmented = sum(1 for u in result.units if not u.complete)
    frag_rate = fragmented / max(len(result.units), 1)

    metrics = Metrics(
        boundary_f1=f,
        boundary_precision=p,
        boundary_recall=r,
        coverage=result.coverage,
        gap_fraction=result.gap_fraction,
        granularity_ratio=granularity_ratio(
            predicted_count=len(result.units),
            truth_count=len(case.truth.units),
        ),
        unit_count_predicted=len(result.units),
        unit_count_truth=len(case.truth.units),
        fragmentation_rate=frag_rate,
        anchor_method_exact=method_counts["exact"],
        anchor_method_normalised=method_counts["whitespace_normalised"],
        anchor_method_fuzzy=method_counts["fuzzy"],
        wall_clock_seconds=elapsed,
        model_calls=result.model_calls,
    )
    return CaseRun(case=case, metrics=metrics, model=model_label, error=None)
