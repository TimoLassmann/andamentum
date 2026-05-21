"""Pure-logic tests for the whetstone benchmark: aggregation + blinding.

Network and model calls are out of scope here — these exercise the
report aggregation and the worksheet's blinding/de-blinding only.
"""

from __future__ import annotations

from benchmarks.whetstone.adjudicate import build_worksheet
from benchmarks.whetstone.report import aggregate, summarise_paper
from benchmarks.whetstone.types import (
    AdjudicatedFinding,
    ArmOutput,
    PaperRef,
    PaperResult,
)


def _result(
    adj: list[AdjudicatedFinding], *, slug_id="10.1/x", match=None
) -> PaperResult:
    return PaperResult(
        paper=PaperRef(source="biorxiv", id=slug_id, version=1),
        arm_a=ArmOutput(arm="A"),
        arm_b=ArmOutput(arm="B"),
        adjudications=adj,
        verdict_match=match,
    )


def _f(bucket, sev="minor", loc="local", text="x") -> AdjudicatedFinding:
    return AdjudicatedFinding(text=text, bucket=bucket, severity=sev, locality=loc)


# ── report aggregation ────────────────────────────────────────────────────


def test_headline_counts_only_b_only_critical_crosssection() -> None:
    adj = [
        _f("b_only", "critical", "cross_section", "arch gap"),
        _f("b_only", "critical", "local", "lens bug"),  # not headline
        _f("b_only", "minor", "cross_section", "minor xsec"),  # not headline
        _f("both", "critical", "cross_section", "shared"),  # not b_only
        _f("a_only", "critical", "cross_section", "whetstone only"),
    ]
    readout = aggregate([_result(adj)])
    assert readout.headline_arch_gaps == 1
    assert readout.total_local_misses == 1
    assert readout.total_both == 1
    assert readout.total_a_only == 1


def test_per_paper_summary_buckets() -> None:
    adj = [_f("both"), _f("a_only"), _f("b_only"), _f("b_only")]
    p = summarise_paper(_result(adj))
    assert (p.both, p.a_only, p.b_only) == (1, 1, 2)


def test_verdict_match_rate() -> None:
    r1 = _result([], match=True)
    r2 = _result([], match=False)
    r3 = _result([], match=None)  # excluded from the rate
    readout = aggregate([r1, r2, r3])
    assert readout.verdict_match_rate == 0.5


# ── worksheet blinding ────────────────────────────────────────────────────


def test_worksheet_includes_all_b_only_critical() -> None:
    adj = [
        _f("b_only", "critical", "cross_section", "must-include-1"),
        _f("b_only", "critical", "local", "must-include-2"),
        _f("both", "minor", "local", "maybe"),
    ]
    sheet, key = build_worksheet(_result(adj), seed=7, sample_minor=0)
    # Both b_only-critical items appear; system labels are anonymised.
    assert "must-include-1" in sheet and "must-include-2" in sheet
    assert "Arm" not in sheet  # no raw arm labels leak
    assert set(key.keys()) == {"System 1", "System 2"}
    assert set(key.values()) == {"A", "B"}


def test_worksheet_blinding_is_deterministic_per_seed() -> None:
    adj = [_f("b_only", "critical", "cross_section", "issue")]
    _, key1 = build_worksheet(_result(adj), seed=42)
    _, key2 = build_worksheet(_result(adj), seed=42)
    assert key1 == key2  # reproducible de-blinding


def test_worksheet_b_only_attribution_maps_through_key() -> None:
    adj = [_f("b_only", "critical", "cross_section", "B caught this")]
    sheet, key = build_worksheet(_result(adj), seed=3, sample_minor=0)
    # The finding is attributed to whichever System maps to arm B.
    b_system = "System 1" if key["System 1"] == "B" else "System 2"
    assert f"{b_system} only" in sheet
