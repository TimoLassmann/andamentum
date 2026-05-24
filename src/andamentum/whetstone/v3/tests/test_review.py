"""Tests for the criterion set, generic review (mocked), and verify-findings."""

from __future__ import annotations

import types
from unittest.mock import patch

from andamentum.whetstone.v3.criteria import SPECS, criterion_set_for
from andamentum.whetstone.v3.model import DocumentModel, Section
from andamentum.whetstone.v3.review import (
    Finding,
    _CriterionFindings,
    _RawFinding,
    run_criteria,
    verify_findings,
)


def test_specs_is_the_academic_default() -> None:
    assert criterion_set_for("academic") is SPECS
    assert criterion_set_for("anything-unknown") is SPECS  # falls back
    assert [c.name for c in SPECS] == [
        "Story",
        "Presentation",
        "Evaluations",
        "Correctness",
        "Significance",
    ]
    assert all(c.questions for c in SPECS)


def _model(src: str) -> DocumentModel:
    return DocumentModel(
        source=src,
        sections=[Section(id="s1", title="S", text=src, start=0, end=len(src))],
    )


async def test_run_criteria_tags_findings_by_criterion() -> None:
    out = _CriterionFindings(
        findings=[
            _RawFinding(issue="x", quote="the claim is unsupported", severity="major")
        ]
    )

    class _Agent:
        async def run(self, _prompt, **_kwargs):
            # Accept deps/usage_limits kwargs from review_criterion silently.
            return types.SimpleNamespace(output=out)

    def _build(_criterion, _agent_model):
        return _Agent()

    with patch("andamentum.whetstone.v3.review._build_agent", new=_build):
        findings = await run_criteria(SPECS, _model("x"), agent_model="stub")
    # one finding per criterion, each tagged with its criterion name
    assert {f.criterion for f in findings} == {c.name for c in SPECS}


def test_verify_findings_drops_hallucinations_and_locates_real_ones() -> None:
    src = "The method is fast. The evaluation lacks a baseline comparison."
    model = _model(src)
    findings = [
        Finding(
            criterion="Evaluations",
            issue="no baseline",
            quote="lacks a baseline comparison",
            severity="major",
        ),
        Finding(
            criterion="Story",
            issue="invented",
            quote="we cured cancer",
            severity="major",
        ),
    ]
    kept = verify_findings(findings, model)
    assert len(kept) == 1
    assert kept[0].issue == "no baseline"
    assert kept[0].span is not None
    assert kept[0].span.section_id == "s1"
    assert src[kept[0].span.start : kept[0].span.end] == "lacks a baseline comparison"
