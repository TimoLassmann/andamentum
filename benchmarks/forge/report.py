"""Render a list of :class:`CaseScore` as a plain markdown report.

``render`` returns a markdown table (one row per case: brief, expected verdict, pass rate
as ``"2/3"``, and a sample outcome) plus an overall pass-rate line. ``print_report`` is the
console-friendly wrapper.
"""

from __future__ import annotations

from .types import CaseScore


def _sample_outcome(score: CaseScore) -> str:
    """A short, representative description of what the runs produced."""
    if not score.runs:
        return "—"
    first = score.runs[0]
    # Tier 1
    if first.kind == "built":
        feats = ",".join(sorted(first.features)) or "sequence"
        return f"built ({feats})"
    if first.kind == "design_failed":
        return f"design_failed: {first.error[:60]}"
    # Tier 2
    if first.kind == "works":
        return f"works ({first.holes_filled}/{first.holes_total} holes)"
    if first.kind == "incomplete":
        holes = f"{first.holes_filled}/{first.holes_total} holes"
        miss = f", unfilled: {','.join(first.remaining_holes)}" if first.remaining_holes else ""
        return f"incomplete ({holes}, tests {first.tests_passed}✓/{first.tests_failed}✗{miss})"
    if first.kind == "build_failed":
        return f"build_failed: {first.error[:60]}"
    if first.kind == "refused":
        return "refused"
    return first.kind


def render(scores: list[CaseScore], *, model: str) -> str:
    """Return a markdown table summarising the benchmark run for one model."""
    if not scores:
        return "_no scores_\n"

    lines: list[str] = []
    lines.append("# Forge benchmark report\n")
    lines.append(f"Model: `{model}`\n")
    lines.append("| case | expected | pass rate | sample outcome |")
    lines.append("|---|---|---|---|")

    total_passes = 0
    total_runs = 0
    for s in scores:
        total_passes += s.passes
        total_runs += s.total
        brief = s.case.brief.replace("|", "\\|")
        lines.append(
            f"| {brief} | {s.case.expected} | {s.passes}/{s.total} | "
            f"{_sample_outcome(s)} |"
        )

    rate = total_passes / total_runs if total_runs else 0.0
    lines.append(
        f"\n**Overall: {total_passes}/{total_runs} runs passed ({rate:.0%}).**\n"
    )
    return "\n".join(lines)


def print_report(scores: list[CaseScore], *, model: str) -> None:
    """Print the markdown report to stdout."""
    print(render(scores, model=model))
