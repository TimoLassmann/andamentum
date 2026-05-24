"""Tests for the gap re-examination loop (mocked agents)."""

from __future__ import annotations

import types
from unittest.mock import patch

from andamentum.whetstone.v3.gaps import Demand, coverage_summary, gap_loop
from andamentum.whetstone.v3.model import DocumentModel, Section, Span
from andamentum.whetstone.v3.review import Finding
from andamentum.whetstone.v3.gaps import (
    _DemandList,
    _Holds,
    _ReexamineFinding,
    _ReexamineFindings,
    _snap_criterion,
)


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
            def output_validator(self, fn):
                # _satisfy_reexamine registers an anchor validator on the
                # real agent; the mock just swallows it.
                return fn

            async def run(self, _p):
                if name == "v3_gap_analysis":
                    out = _DemandList(demands=demands)
                elif name == "v3_reexamine":
                    out = _ReexamineFindings(findings=reexamine or [])
                else:  # v3_recheck
                    out = _Holds(holds=holds)
                return types.SimpleNamespace(output=out)

        return A()

    return (
        patch("andamentum.whetstone.v3.gaps.build_pydantic_ai_agent", new=factory),
        patch("andamentum.whetstone.v3.gaps.resolve_model", new=lambda m: None),
    )


def test_snap_criterion_matches_and_falls_back() -> None:
    names = ["Story", "Presentation", "Correctness"]
    assert _snap_criterion("correctness", names) == "Correctness"  # case-insensitive
    assert _snap_criterion("the Presentation criterion", names) == "Presentation"
    assert _snap_criterion("nonsense", names) == "Story"  # fallback to first


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
        _ReexamineFinding(
            issue="no baseline",
            quote="omits a baseline comparison",
            severity="major",
            criterion="Evaluations",
        )
    ]
    p1, p2 = _route(demands=[demand], reexamine=new)
    with p1, p2:
        out = await gap_loop(
            model,
            [],
            agent_model="stub",
            cap=2,
            criterion_names=["Evaluations", "Story"],
        )
    assert len(out) == 1
    assert out[0].issue == "no baseline"
    assert out[0].span is not None  # verified/located
    assert out[0].criterion == "Evaluations"  # classified, not "re-examination"


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


async def test_reexamine_receives_prior_findings_block() -> None:
    """Issue 8: _satisfy_reexamine must surface accumulated findings to the
    agent so it doesn't re-flag known issues. Audit 8A found strong
    cross-run topical overlap on the same sections (s7 regret, s4
    efficiency, s5 stepsize bound); the demand-signature dedup only
    catches identical demands, not duplicate findings."""
    received_prompts: list[tuple[str, str]] = []
    src = "Some text describing the method."
    model = _model(src)

    def factory(defn, _model):
        name = defn.name

        class A:
            def output_validator(self, fn):
                return fn

            async def run(self, p):
                received_prompts.append((name, p))
                if name == "v3_gap_analysis":
                    return types.SimpleNamespace(
                        output=_DemandList(
                            demands=[
                                Demand(
                                    kind="reexamine",
                                    detail="look here",
                                    target_section_id="s1",
                                )
                            ]
                        )
                    )
                # v3_reexamine: return no new findings
                return types.SimpleNamespace(output=_ReexamineFindings(findings=[]))

        return A()

    prior = [
        Finding(
            criterion="Story",
            issue="overclaim about Adam",
            quote="Some text",
            severity="major",
            span=Span(section_id="s1", start=0, end=9),
        )
    ]
    with (
        patch("andamentum.whetstone.v3.gaps.build_pydantic_ai_agent", new=factory),
        patch("andamentum.whetstone.v3.gaps.resolve_model", new=lambda m: None),
    ):
        await gap_loop(model, prior, agent_model="stub", cap=1)

    # The reexamine call should have received a PRIOR FINDINGS block
    reexamine_prompts = [p for name, p in received_prompts if name == "v3_reexamine"]
    assert len(reexamine_prompts) == 1
    prompt = reexamine_prompts[0]
    assert "PRIOR FINDINGS" in prompt
    assert "Story/major" in prompt
    assert "overclaim about Adam" in prompt


async def test_per_round_demand_cap_truncates_chatty_round() -> None:
    """Issue 7: per-round demand cap. analyze_gaps may emit any number of
    demands; gap_loop must truncate to at most ``per_round_demand_cap``
    before satisfying them. Otherwise a chatty round dominates wall-clock
    and structural ceiling on LLM calls is lost."""
    src = "Some text with the method described."
    model = _model(src)
    # 5 distinct reexamine demands all pointing at s1. With cap=2 rounds and
    # per_round_demand_cap=3, the loop should satisfy 3 in round 1 then exit
    # in round 2 because the same demands are filtered by prior memory.
    demands = [
        Demand(kind="reexamine", detail=f"angle {i}", target_section_id="s1")
        for i in range(5)
    ]
    reexamine = [
        _ReexamineFinding(
            issue="point", quote="the method described", severity="minor", criterion="Story"
        )
    ]
    p1, p2 = _route(demands=demands, reexamine=reexamine)
    with p1, p2:
        # cap=1 isolates the per-round cap from cross-round prior-dedup —
        # only one round runs, so the only thing that can drop demands is
        # the truncation step.
        out = await gap_loop(
            model, [], agent_model="stub", cap=1, per_round_demand_cap=3
        )
    # 5 demands emitted, 3 retained after truncation. Each satisfies to one
    # finding (same quote, anchors via verify_findings) → 3 findings total.
    assert len(out) == 3


async def test_loop_terminates_via_demand_memory() -> None:
    # The same demand is returned every round; memory must filter it so the
    # loop exits at round 2 rather than running forever.
    src = "Some text with a baseline mentioned."
    model = _model(src)
    demand = Demand(kind="reexamine", detail="x", target_section_id="s1")
    new = [
        _ReexamineFinding(
            issue="i", quote="a baseline mentioned", severity="minor", criterion="Story"
        )
    ]
    p1, p2 = _route(demands=[demand], reexamine=new)
    with p1, p2:
        out = await gap_loop(model, [], agent_model="stub", cap=5)
    # Demand satisfied once (round 1); filtered as prior in round 2 → exit.
    assert len(out) == 1
