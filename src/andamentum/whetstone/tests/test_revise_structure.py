"""Tests for the structure/flow check of `revise`.

It reuses ``ReviewResult.findings`` unchanged — nothing here calls a model.
"""

from __future__ import annotations

from andamentum.whetstone.revise.structure import structure_clean
from andamentum.whetstone.schemas import Finding, ReviewResult


def _finding(**overrides) -> Finding:
    defaults = dict(
        title="unmotivated transition",
        severity="minor",
        confidence="medium",
        rationale="Section 3 jumps to the results with no bridge.",
    )
    defaults.update(overrides)
    return Finding(**defaults)  # type: ignore[arg-type]


def _result(findings: list[Finding]) -> ReviewResult:
    return ReviewResult(summary="", findings=findings)


def test_no_findings_is_satisfied() -> None:
    report = structure_clean(_result([]))
    assert report.satisfied
    assert report.blocking_count == 0


def test_only_minor_findings_is_satisfied() -> None:
    report = structure_clean(_result([_finding(severity="minor")]))
    assert report.satisfied


def test_moderate_finding_blocks() -> None:
    report = structure_clean(_result([_finding(severity="moderate")]))
    assert not report.satisfied
    assert report.blocking_count == 1


def test_major_finding_blocks() -> None:
    report = structure_clean(_result([_finding(severity="major")]))
    assert not report.satisfied
    assert report.blocking_count == 1


def test_mixed_severities_counts_only_blocking() -> None:
    findings = [
        _finding(severity="minor"),
        _finding(severity="moderate", title="a"),
        _finding(severity="major", title="b"),
    ]
    report = structure_clean(_result(findings))
    assert report.blocking_count == 2
    assert report.blocking_titles == ["a", "b"]


def test_blocking_titles_capped_at_five() -> None:
    findings = [_finding(severity="major", title=f"issue {i}") for i in range(8)]
    report = structure_clean(_result(findings))
    assert report.blocking_count == 8
    assert len(report.blocking_titles) == 5
