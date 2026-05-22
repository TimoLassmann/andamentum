"""Tests for gating (deterministic) + synthesis adapter."""

from __future__ import annotations

import types
from unittest.mock import patch

from andamentum.whetstone.v3.gate import gate_and_aggregate
from andamentum.whetstone.v3.model import DocumentModel, Section, Span
from andamentum.whetstone.v3.review import Finding
from andamentum.whetstone.v3.synth import (
    StructuredReview,
    _flatten,
    synthesise,
    to_review_result,
)


def _f(criterion, sev, sec, start, end, *, issue="i", quote="q") -> Finding:
    return Finding(
        criterion=criterion,
        issue=issue,
        quote=quote,
        severity=sev,
        span=Span(section_id=sec, start=start, end=end),
    )


def test_gate_drops_overlapping_keeps_most_severe() -> None:
    a = _f("Story", "minor", "s1", 0, 10)
    b = _f("Story", "major", "s1", 5, 15)  # overlaps a; more severe
    c = _f("Evaluations", "moderate", "s2", 0, 10)  # disjoint
    kept = gate_and_aggregate([a, b, c])
    assert b in kept and a not in kept  # most-severe overlapper kept
    assert c in kept
    assert kept[0].severity == "major"  # ordered by severity desc


def test_flatten_review_markdown() -> None:
    r = StructuredReview(
        synopsis="A study.", strengths=["clear"], weaknesses=["no baseline"]
    )
    md = _flatten(r)
    assert "## Summary" in md and "A study." in md
    assert "## Strengths" in md and "- clear" in md
    assert "## Weaknesses" in md and "- no baseline" in md


def test_to_review_result_maps_quotes_to_section_relative_offsets() -> None:
    src = "0123456789ABCDEFGHIJ"
    model = DocumentModel(
        source=src,
        sections=[Section(id="s1", title="S", text=src[10:], start=10, end=20)],
    )
    f = _f("Evaluations", "major", "s1", 12, 16, issue="bad", quote="2345")
    rr = to_review_result(model, [f], StructuredReview(synopsis="x"))
    assert len(rr.findings) == 1
    q = rr.findings[0].quotes[0]
    # source-absolute (12,16) → section-relative (2,6) since section.start=10
    assert (q.char_start, q.char_end) == (2, 6)
    assert q.section_id == "s1" and q.text == "2345"
    assert rr.findings[0].category == "evaluations"
    assert rr.summary.startswith("## Summary")
    assert rr.document_map and rr.document_map[0].section_id == "s1"


async def test_synthesise_returns_structured_review() -> None:
    out = StructuredReview(synopsis="ok", strengths=["s"], weaknesses=["w"])

    def factory(defn, model):
        class A:
            async def run(self, _p):
                return types.SimpleNamespace(output=out)

        return A()

    model = DocumentModel(source="x", sections=[])
    with (
        patch("andamentum.whetstone.v3.synth.build_pydantic_ai_agent", new=factory),
        patch("andamentum.whetstone.v3.synth.resolve_model", new=lambda m: None),
    ):
        r = await synthesise(model, [], agent_model="stub")
    assert r.synopsis == "ok"
