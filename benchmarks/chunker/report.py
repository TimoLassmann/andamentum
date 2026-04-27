"""Markdown report generation from a list of CaseRun."""

from __future__ import annotations

from .types import CaseRun


def to_markdown_table(runs: list[CaseRun]) -> str:
    """Render a markdown table summarising one model's run across cases."""
    if not runs:
        return "_no runs_\n"

    model = runs[0].model
    lines: list[str] = []
    lines.append(f"# Chunker benchmark report\n\nModel: `{model}`\n")
    lines.append(
        "| case | F1 | precision | recall | coverage | granularity | "
        "calls | time (s) | floor | result |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    passed = 0
    for r in runs:
        if r.error is not None:
            lines.append(
                f"| {r.case.name} | — | — | — | — | — | — | — | "
                f"{r.case.expected_f1_floor:.2f} | ❌ ERROR: {r.error[:80]} |"
            )
            continue
        m = r.metrics
        assert m is not None
        result = "✅ pass" if r.passed_floor else "❌ below floor"
        if r.passed_floor:
            passed += 1
        lines.append(
            f"| {r.case.name} | {m.boundary_f1:.2f} | {m.boundary_precision:.2f} | "
            f"{m.boundary_recall:.2f} | {m.coverage:.2f} | {m.granularity_ratio:.2f} | "
            f"{m.model_calls} | {m.wall_clock_seconds:.1f} | "
            f"{r.case.expected_f1_floor:.2f} | {result} |"
        )

    lines.append(f"\n**{passed}/{len(runs)} cases passed their F1 floor.**\n")
    return "\n".join(lines)
