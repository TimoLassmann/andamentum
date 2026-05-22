"""Tests for the light consolidation pass (mocked agent)."""

from __future__ import annotations

import types
from unittest.mock import patch

from andamentum.whetstone.v3.consolidate import _Consolidation, _Group, consolidate
from andamentum.whetstone.v3.model import Span
from andamentum.whetstone.v3.review import Finding


def _route(consolidation: _Consolidation | None, *, crash: bool = False):
    def factory(_defn, _model):
        class A:
            async def run(self, _p):
                if crash:
                    raise RuntimeError("boom")
                return types.SimpleNamespace(output=consolidation)

        return A()

    return (
        patch(
            "andamentum.whetstone.v3.consolidate.build_pydantic_ai_agent", new=factory
        ),
        patch("andamentum.whetstone.v3.consolidate.resolve_model", new=lambda m: None),
    )


def _f(criterion: str, issue: str, quote: str, severity: str = "moderate") -> Finding:
    return Finding(
        criterion=criterion,
        issue=issue,
        quote=quote,
        severity=severity,  # type: ignore[arg-type]
        span=Span(section_id="s1", start=0, end=len(quote)),
    )


async def test_merges_a_group_keeping_most_severe_anchor() -> None:
    findings = [
        _f("Correctness", "step A under-specified", "qa", severity="minor"),
        _f("Story", "distinct point", "qb", severity="major"),
        _f("Correctness", "step C under-specified", "qc", severity="major"),
    ]
    grouped = _Consolidation(
        groups=[_Group(member_indices=[0, 2], merged_issue="methods under-specified")]
    )
    p1, p2 = _route(grouped)
    with p1, p2:
        out = await consolidate(findings, agent_model="stub")

    assert len(out) == 2  # group(0,2) → 1 merged + finding[1] passthrough
    merged = next(f for f in out if f.issue == "methods under-specified")
    assert merged.severity == "major"  # most-severe member
    assert merged.quote == "qc"  # anchored on the most-severe member
    assert any(f.issue == "distinct point" for f in out)  # untouched


async def test_singletons_and_oob_indices_ignored() -> None:
    findings = [_f("Story", "x", "qx"), _f("Story", "y", "qy")]
    grouped = _Consolidation(
        groups=[_Group(member_indices=[0], merged_issue="solo")]  # <2 → ignored
    )
    p1, p2 = _route(grouped)
    with p1, p2:
        out = await consolidate(findings, agent_model="stub")
    assert len(out) == 2
    assert {f.issue for f in out} == {"x", "y"}


async def test_agent_crash_returns_findings_unchanged() -> None:
    findings = [_f("Story", "x", "qx"), _f("Story", "y", "qy")]
    p1, p2 = _route(None, crash=True)
    with p1, p2:
        out = await consolidate(findings, agent_model="stub")
    assert out == findings


async def test_short_list_skips_agent() -> None:
    findings = [_f("Story", "only one", "q")]
    # No patch needed — should return before building an agent.
    out = await consolidate(findings, agent_model="stub")
    assert out == findings
