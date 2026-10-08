"""Tests for gating (deterministic) + synthesis adapter."""

from __future__ import annotations

import types
from unittest.mock import patch

from andamentum.whetstone.pipeline.gate import gate_and_aggregate
from andamentum.whetstone.pipeline.model import DocumentModel, Section, Span
from andamentum.whetstone.pipeline.review import Finding
from andamentum.whetstone.pipeline.synth import (
    StructuredReview,
    _flatten,
    _synopsis_length_band,
    critique_and_revise,
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


def test_no_finding_is_discarded_for_overlapping_a_more_severe_one() -> None:
    """Both points survive, each on its own passage.

    Dropping the less severe one lost about a fifth of the review on two real
    documents, and every drop was cross-criterion — consolidate has already
    merged the same-point findings, so what reaches the gate is a DIFFERENT
    observation about the same sentence.
    """
    a = _f("Presentation", "minor", "s1", 0, 10, issue="signposting is weak")
    b = _f("Story", "major", "s1", 5, 15, issue="the novelty claim is unsupported")
    c = _f("Evaluations", "moderate", "s2", 0, 10, issue="define the metric")

    kept = gate_and_aggregate([a, b, c])

    assert sorted(kept, key=id) == sorted([a, b, c], key=id)
    # Each keeps its own anchor, severity and criterion — nothing is reworded.
    assert kept == [b, c, a]  # severity desc, then criterion


def test_findings_are_ordered_most_severe_first() -> None:
    minor = _f("Story", "minor", "s1", 0, 10)
    major = _f("Story", "major", "s2", 0, 10)
    moderate = _f("Story", "moderate", "s3", 0, 10)
    assert gate_and_aggregate([minor, major, moderate]) == [major, moderate, minor]


def test_one_perspective_s_points_sit_together() -> None:
    """Criterion is the secondary key, so a reader reads one angle at a time."""
    story = _f("Story", "moderate", "s1", 0, 10)
    evaluations = _f("Evaluations", "moderate", "s2", 0, 10)
    story_two = _f("Story", "moderate", "s3", 0, 10)
    ordered = gate_and_aggregate([story, evaluations, story_two])
    assert [f.criterion for f in ordered] == ["Evaluations", "Story", "Story"]


def test_flatten_review_markdown() -> None:
    r = StructuredReview(
        synopsis="A study.", strengths=["clear"], weaknesses=["no baseline"]
    )
    md = _flatten(r)
    assert "## Summary" in md and "A study." in md
    assert "## Strengths" in md and "- clear" in md
    assert "## Weaknesses" in md and "- no baseline" in md


def test_to_review_result_keeps_quotes_in_document_coordinates() -> None:
    src = "0123456789ABCDEFGHIJ"
    model = DocumentModel(
        source=src,
        sections=[Section(id="s1", title="S", text=src[10:], start=10, end=20)],
    )
    f = _f("Evaluations", "major", "s1", 12, 16, issue="bad", quote="2345")
    rr = to_review_result(model, [f], StructuredReview(synopsis="x"))
    assert len(rr.findings) == 1
    q = rr.findings[0].quotes[0]
    # The span is copied across, not rebased. Both are addresses in the whole
    # reviewed document, which is what the docx renderer hands to the surgery
    # layer; subtracting section.start=10 would place the annotation ten
    # characters early.
    assert (q.char_start, q.char_end) == (12, 16)
    assert q.section_id == "s1" and q.text == "2345"
    assert rr.findings[0].category == "evaluations"
    assert rr.summary.startswith("## Summary")
    assert rr.document_map and rr.document_map[0].section_id == "s1"


def test_synopsis_length_band_short_doc() -> None:
    """Issue 9: ≤1000 words → 1-sentence band."""
    source = "word " * 500  # 500 words
    assert _synopsis_length_band(source) == "1 sentence"


def test_synopsis_length_band_typical_paper() -> None:
    """Issue 9: 1000-5000 words → 2-4 sentences band."""
    source = "word " * 3000  # 3000 words
    assert _synopsis_length_band(source) == "2-4 sentences"


def test_synopsis_length_band_long_manuscript() -> None:
    """Issue 9: >5000 words → 4-8 sentences band."""
    source = "word " * 8000  # 8000 words
    assert _synopsis_length_band(source) == "4-8 sentences"


def test_synopsis_length_band_boundary_at_1000() -> None:
    """Boundary case: exactly 1000 words → still in 1-sentence band (≤)."""
    source = "word " * 1000
    assert _synopsis_length_band(source) == "1 sentence"


def test_synopsis_length_band_boundary_at_5000() -> None:
    """Boundary case: exactly 5000 words → still in 2-4 sentences band (≤)."""
    source = "word " * 5000
    assert _synopsis_length_band(source) == "2-4 sentences"


async def test_critique_receives_three_ground_truth_blocks() -> None:
    """Issue 1: critique_and_revise must surface SECTION GISTS, AUTHOR
    CLAIMS, and SUPPORTED FINDINGS as separate labeled blocks. The
    previous version validated against claims alone, which structurally
    cannot support absence-based weaknesses."""
    from andamentum.whetstone.pipeline.model import Claim, SectionGist

    received_prompts: list[str] = []
    revised = StructuredReview(synopsis="ok", strengths=["s"], weaknesses=["w"])

    def factory(_defn, _model):
        class A:
            async def run(self, p):
                received_prompts.append(p)
                return types.SimpleNamespace(output=revised)

        return A()

    model = DocumentModel(
        source="some doc text",
        sections=[
            Section(id="s1", title="Intro", text="some doc text", start=0, end=13)
        ],
        claims=[
            Claim(
                id="c1",
                quote="some doc text",
                span=Span(section_id="s1", start=0, end=13),
            )
        ],
        gists=[SectionGist(section_id="s1", title="Intro", gist="the intro")],
    )
    draft = StructuredReview(
        synopsis="draft", strengths=["a"], weaknesses=["no baseline"]
    )
    findings = [
        Finding(
            criterion="Evaluations",
            issue="lacks baseline comparison",
            quote="some doc",
            severity="major",
            span=Span(section_id="s1", start=0, end=8),
        )
    ]
    with (
        patch("andamentum.whetstone.pipeline.synth.build_pydantic_ai_agent", new=factory),
        patch("andamentum.whetstone.pipeline.synth.resolve_model", new=lambda m: None),
    ):
        out = await critique_and_revise(model, draft, findings, agent_model="stub")

    assert out is revised
    assert len(received_prompts) == 1
    prompt = received_prompts[0]
    # All three ground-truth blocks must appear
    assert "SECTION GISTS" in prompt
    assert "AUTHOR CLAIMS" in prompt
    assert "SUPPORTED FINDINGS" in prompt
    # Finding text should be in the SUPPORTED FINDINGS block so the agent
    # knows to preserve weaknesses matching it
    assert "lacks baseline comparison" in prompt
    # Gist + claim still shown
    assert "the intro" in prompt
    assert "some doc text" in prompt
    # Draft itself shown (model_dump_json includes the keys)
    assert "no baseline" in prompt


async def test_synthesise_returns_structured_review() -> None:
    out = StructuredReview(synopsis="ok", strengths=["s"], weaknesses=["w"])

    def factory(defn, model):
        class A:
            async def run(self, _p):
                return types.SimpleNamespace(output=out)

        return A()

    model = DocumentModel(source="x", sections=[])
    with (
        patch("andamentum.whetstone.pipeline.synth.build_pydantic_ai_agent", new=factory),
        patch("andamentum.whetstone.pipeline.synth.resolve_model", new=lambda m: None),
    ):
        r = await synthesise(model, [], agent_model="stub")
    assert r.synopsis == "ok"


def test_an_exact_repeat_is_dropped() -> None:
    """Same span, same words — one observation reported twice. Deciding this
    needs no judgement, unlike dropping a differently-worded point."""
    a = _f("Story", "major", "s1", 0, 10, issue="the claim is unsupported")
    b = _f("Evaluations", "minor", "s1", 0, 10, issue="the claim is unsupported")
    kept = gate_and_aggregate([a, b])
    assert kept == [a]  # the more severe copy survives


def test_the_same_words_on_a_different_passage_are_two_findings() -> None:
    """A repeated problem in two places is two problems to fix."""
    a = _f("Story", "major", "s1", 0, 10, issue="unsupported claim")
    b = _f("Story", "major", "s1", 40, 50, issue="unsupported claim")
    assert len(gate_and_aggregate([a, b])) == 2


def test_differently_worded_points_on_one_passage_both_survive() -> None:
    """Similarity is not identity. Judging whether two wordings mean the same
    thing is consolidate's job, with an LLM and a conservative bias — this
    stage only removes literal repeats."""
    a = _f("Story", "major", "s1", 0, 10, issue="the claim is unsupported")
    b = _f("Story", "major", "s1", 0, 10, issue="this claim lacks support")
    assert len(gate_and_aggregate([a, b])) == 2


def test_a_consolidate_failure_cannot_flood_one_sentence() -> None:
    """Consolidate returns its input unchanged on any agent error. Without a
    deterministic backstop that would put nine identical comments in a margin."""
    repeats = [
        _f(c, "moderate", "s1", 0, 10, issue="operationally under-specified")
        for c in ("Story", "Presentation", "Evaluations", "Correctness", "Significance")
    ]
    assert len(gate_and_aggregate(repeats)) == 1
