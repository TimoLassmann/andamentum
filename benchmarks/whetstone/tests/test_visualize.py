"""Pure-logic tests for the side-by-side HTML visualiser (no network/LLM)."""

from __future__ import annotations

from benchmarks.whetstone.types import (
    AdjudicatedFinding,
    ArmFinding,
    ArmOutput,
    Comparison,
    PaperRef,
    PaperResult,
)
from benchmarks.whetstone.visualize import build_html


def _result() -> PaperResult:
    return PaperResult(
        paper=PaperRef(source="biorxiv", id="10.1/x", version=1, title="A Study"),
        arm_a=ArmOutput(
            arm="A",
            findings=[ArmFinding(title="passive voice", detail="here")],
            verdict="whetstone says X",
        ),
        arm_b=ArmOutput(
            arm="B",
            findings=[
                ArmFinding(title="claim unsupported", detail="<b>bad & risky</b>")
            ],
            verdict="whole-doc says Y",
        ),
        adjudications=[
            AdjudicatedFinding(
                text="the central claim is unsupported",
                bucket="b_only",
                severity="critical",
                locality="cross_section",
            )
        ],
        comparison=Comparison(
            more_useful="whole-doc",
            reasoning="B caught the unsupported central claim that A missed.",
        ),
    )


def test_build_html_is_self_contained_and_shows_both_arms() -> None:
    html = build_html([_result()], css="/* css */")
    assert html.startswith("<!DOCTYPE html>")
    assert "<style>" in html  # css inlined → portable
    # Both systems' findings appear.
    assert "passive voice" in html
    assert "claim unsupported" in html
    # Both verdicts appear.
    assert "whetstone says X" in html and "whole-doc says Y" in html
    # The architecture-gap (b_only critical cross_section) is surfaced.
    assert "architecture gap" in html.lower()
    assert "the central claim is unsupported" in html
    # The top comparison section: grounded verdict + scorecard.
    assert "comparative verdict" in html.lower()
    assert "More useful" in html
    assert "B caught the unsupported central claim" in html  # the reasoning prose
    assert "whetstone-only minor (noise)" in html  # a scorecard axis
    # The judge's per-issue adjudication is shown with bucket/severity badges.
    assert "Judge adjudication" in html
    assert "whole-doc only" in html  # bucket label for the b_only issue
    assert "am-badge--danger" in html  # critical severity badge


def test_build_html_escapes_finding_text() -> None:
    html = build_html([_result()], css="")
    # Raw markup from a finding must be escaped, not injected.
    assert "<b>bad & risky</b>" not in html
    assert "&lt;b&gt;bad &amp; risky&lt;/b&gt;" in html


def test_build_html_lists_every_paper_in_sidebar() -> None:
    r1 = _result()
    r2 = _result()
    r2.paper.title = "Second Study"
    html = build_html([r1, r2], css="")
    assert "A Study" in html and "Second Study" in html
    assert html.count('class="paper"') == 2
    # Only the first paper is visible initially.
    assert 'id="paper-1" hidden' in html or 'id="paper-1"hidden' in html
