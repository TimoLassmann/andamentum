"""Plan-manager grounding (Tier 1) — deterministic coverage, semantic dedup, the
Review-node loopback, and the clean-plan pass-through.

``plan_coverage`` is a pure function; ``review_plan`` is one stubbed agent call plus a
deterministic rapidfuzz dedup; the Review node is exercised end-to-end through
``run_forge`` with a scripted sink (no live model, no container).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from andamentum.core import AgentDefinition
from andamentum.forge import run_forge
from andamentum.forge.review import plan_coverage, review_plan
from andamentum.forge.schemas import (
    FindingKind,
    ForgeWhy,
    NodeDraft,
    NodeTyping,
    PlanVerdict,
)
from andamentum.forge.spec import NodeKind

from .conftest import ScriptedSink

_WHY = ForgeWhy(
    purpose="Help the user manage a personal reading list.",
    boundary_in="a natural-language request",
    boundary_out="a text answer",
)


def _draft(node_id: str, area: str, job: str) -> NodeDraft:
    return NodeDraft(id=node_id, area=area, job=job, kind=NodeKind.SPINE)


# --- Tier 1a: deterministic coverage --------------------------------------------


def test_plan_coverage_flags_uncovered_area() -> None:
    drafts = [_draft("n1", "core", "Parse the request.")]
    findings = plan_coverage(_WHY, ["core", "persistence"], drafts)
    assert len(findings) == 1
    f = findings[0]
    assert f.kind is FindingKind.UNCOVERED_AREA
    assert "persistence" in f.detail


def test_plan_coverage_clean_when_every_area_owns_a_step() -> None:
    drafts = [
        _draft("n1", "core", "Parse the request."),
        _draft("n2", "persistence", "Save the entry."),
    ]
    assert plan_coverage(_WHY, ["core", "persistence"], drafts) == []


# --- Tier 1b: the semantic plan review + dedup ----------------------------------


class _PlanSink:
    """A minimal ``AgentSink`` that answers only ``plan_manager`` with a scripted verdict."""

    def __init__(self, verdict: PlanVerdict) -> None:
        self._verdict = verdict

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        assert defn.name == "plan_manager"
        return self._verdict


async def test_review_plan_drops_concern_already_covered_by_a_job() -> None:
    # A concern that fuzzy-matches an existing job is redundant — drop it.
    sink = _PlanSink(
        PlanVerdict(serves_goal=False, uncovered_concerns=["answer the request"])
    )
    verdict = await review_plan(
        _WHY, "- n1: Answer the request.", ["Answer the request."], sink=sink
    )
    assert verdict.uncovered_concerns == []


async def test_review_plan_keeps_a_genuinely_new_concern() -> None:
    sink = _PlanSink(
        PlanVerdict(
            serves_goal=False, uncovered_concerns=["persist the list between sessions"]
        )
    )
    verdict = await review_plan(
        _WHY, "- n1: Answer the request.", ["Answer the request."], sink=sink
    )
    assert verdict.uncovered_concerns == ["persist the list between sessions"]


async def test_review_plan_passes_serves_goal_through() -> None:
    sink = _PlanSink(PlanVerdict(serves_goal=True, uncovered_concerns=[]))
    verdict = await review_plan(_WHY, "- n1: Answer.", ["Answer."], sink=sink)
    assert verdict.serves_goal is True
    assert verdict.uncovered_concerns == []


# --- the Review node: loopback + fail-loud at the cap, and clean pass-through ----


class _RejectingSink(ScriptedSink):
    """A scripted sink whose plan manager always rejects with a fresh concern that does
    NOT fuzzy-match the canned jobs, so it survives dedup and the loop runs to the cap."""

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        if defn.name == "plan_manager":
            return PlanVerdict(
                serves_goal=False,
                uncovered_concerns=["persist entries to durable storage between runs"],
            )
        return await super().run(defn, **kwargs)


def _rejecting_sink() -> _RejectingSink:
    # Supply coherent typings so the board assembles clean and Review actually runs each
    # round — the rejection (not a structural flaw) is what must drive the loop.
    return _RejectingSink(
        why=_WHY,
        areas=["core"],
        jobs_by_area={"core": ["Parse the request.", "Answer the request."]},
        typings={
            "n1": NodeTyping(
                kind=NodeKind.SPINE, consumes=["input"], produces=["parsed_request"]
            ),
            "n2": NodeTyping(
                kind=NodeKind.HEAD, consumes=["parsed_request"], produces=["answer"]
            ),
        },
    )


async def test_review_loops_back_and_raises_at_the_cap() -> None:
    # serves_goal=False with a surviving concern → loop back to Frame, redesign, re-review;
    # the stateless sink keeps rejecting, so the pipeline fails loud at MAX_PLAN_REVIEW_ROUNDS.
    with pytest.raises(ValueError) as exc:
        await run_forge("Manage my reading list.", model="test", sink=_rejecting_sink())
    msg = str(exc.value)
    assert "plan review did not converge" in msg
    assert "persist entries to durable storage between runs" in msg


async def test_clean_plan_records_the_verdict_and_passes_through(
    reading_list_sink: ScriptedSink,
) -> None:
    result = await run_forge(
        "Manage my reading list.", model="test", sink=reading_list_sink
    )
    assert result.design_only
    assert result.plan_review is not None
    assert result.plan_review.serves_goal is True
    assert result.plan_review.uncovered_concerns == []
