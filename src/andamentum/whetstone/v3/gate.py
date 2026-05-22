"""Deterministic gating + dedup of findings before synthesis.

The flood is structurally smaller in v3 (criterion stages emit bounded findings,
no per-instance proofread), so gating is light: drop near-duplicates (findings
whose quotes overlap the same span) keeping the most severe, and order by
importance (severity, then criterion). No LLM.
"""

from __future__ import annotations

from .review import Finding

_SEVERITY_RANK = {"major": 3, "moderate": 2, "minor": 1}


def _overlap(a: Finding, b: Finding) -> bool:
    if a.span is None or b.span is None:
        return False
    if a.span.section_id != b.span.section_id:
        return False
    return a.span.start < b.span.end and b.span.start < a.span.end


def gate_and_aggregate(findings: list[Finding]) -> list[Finding]:
    """Dedup overlapping findings (keep most severe) and order by importance."""
    ordered = sorted(findings, key=lambda f: -_SEVERITY_RANK.get(f.severity, 0))
    kept: list[Finding] = []
    for f in ordered:
        if any(_overlap(f, k) for k in kept):
            continue
        kept.append(f)
    # Final order: severity desc, then criterion name for stable grouping.
    kept.sort(key=lambda f: (-_SEVERITY_RANK.get(f.severity, 0), f.criterion))
    return kept
