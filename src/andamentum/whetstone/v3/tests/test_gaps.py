"""Tests for the gap re-examination loop (mocked agents)."""

from __future__ import annotations

import types
from unittest.mock import patch

from andamentum.whetstone.v3.gaps import Demand, coverage_summary, gap_loop
from andamentum.whetstone.v3.model import DocumentModel, Section, Span
from andamentum.whetstone.v3.review import _CriterionFindings, _RawFinding, Finding
from andamentum.whetstone.v3.gaps import _DemandList, _Holds


def _model(src: str) -> DocumentModel:
    return DocumentModel(
        source=src,
        sections=[Section(id="s1", title="S", text=src, start=0, end=len(src))],
    )


def _route(*, demands, reexamine=None, holds=True):
    """Patch gaps' agent builder to route by definition name."""

    def factory(defn, model):
        name = defn.name

        class A:
            async def run(self, _p):
                if name == "v3_gap_analysis":
                    out = _DemandList(demands=demands)
                elif name == "v3_reexamine":
                    out = _CriterionFindings(findings=reexamine or [])
                else:  # v3_recheck
                    out = _Holds(holds=holds)
                return types.SimpleNamespace(output=out)

        return A()

    return (
        patch("andamentum.whetstone.v3.gaps.build_pydantic_ai_agent", new=factory),
        patch("andamentum.whetstone.v3.gaps.resolve_model", new=lambda m: None),
    )


def test_coverage_summary_counts_and_untouched() -> None:
    model = _model("body text here")
    findings = [
        Finding(
            criterion="Story",
            issue="x",
            quote="body",
            span=Span(section_id="s1", start=0, end=4),
        )
    ]
    summary = coverage_summary(findings, model)
    assert "Story:1" in summary
    assert "(none)" in summary  # no untouched sections (s1 is touched)


async def test_empty_demands_leaves_findings_unchanged() -> None:
    model = _model("the method is fast")
    findings = [Finding(criterion="Story", issue="x", quote="the method is fast")]
    p1, p2 = _route(demands=[])
    with p1, p2:
        out = await gap_loop(model, findings, agent_model="stub", cap=2)
    assert out == findings


async def test_reexamine_adds_a_verified_finding() -> None:
    src = "The evaluation omits a baseline comparison entirely."
    model = _model(src)
    demand = Demand(kind="reexamine", detail="check evaluation", target_section_id="s1")
    new = [
        _RawFinding(
            issue="no baseline", quote="omits a baseline comparison", severity="major"
        )
    ]
    p1, p2 = _route(demands=[demand], reexamine=new)
    with p1, p2:
        out = await gap_loop(model, [], agent_model="stub", cap=2)
    assert len(out) == 1
    assert out[0].issue == "no baseline"
    assert out[0].span is not None  # verified/located


async def test_recheck_drops_a_finding_that_does_not_hold() -> None:
    src = "The method is fast."
    model = _model(src)
    findings = [
        Finding(
            criterion="Story",
            issue="dubious",
            quote="The method is fast.",
            span=Span(section_id="s1", start=0, end=19),
        )
    ]
    demand = Demand(kind="recheck", finding_index=0)
    p1, p2 = _route(demands=[demand], holds=False)
    with p1, p2:
        out = await gap_loop(model, findings, agent_model="stub", cap=2)
    assert out == []  # finding rechecked, didn't hold → dropped


async def test_loop_terminates_via_demand_memory() -> None:
    # The same demand is returned every round; memory must filter it so the
    # loop exits at round 2 rather than running forever.
    src = "Some text with a baseline mentioned."
    model = _model(src)
    demand = Demand(kind="reexamine", detail="x", target_section_id="s1")
    new = [_RawFinding(issue="i", quote="a baseline mentioned", severity="minor")]
    p1, p2 = _route(demands=[demand], reexamine=new)
    with p1, p2:
        out = await gap_loop(model, [], agent_model="stub", cap=5)
    # Demand satisfied once (round 1); filtered as prior in round 2 → exit.
    assert len(out) == 1
