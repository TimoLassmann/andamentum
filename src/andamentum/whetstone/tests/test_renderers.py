"""Tests for the markdown / HTML / docx renderers.

Markdown and HTML are exercised on a sample ReviewResult — they're pure
functions, no external dependencies. The docx renderer test is lighter:
it only verifies the v2→v1 DocumentPatch adapter (since the actual
docx-writing machinery is v1's, well-tested elsewhere).
"""

from pathlib import Path


from andamentum.whetstone import (
    AuthorQuestion,
    Edit,
    Finding,
    Quote,
    ReviewMetrics,
    ReviewResult,
    SectionCard,
    render_html,
    render_markdown,
)


def _sample_result() -> ReviewResult:
    """A small ReviewResult exercising every section a renderer might emit."""
    return ReviewResult(
        summary="## Executive Summary\n\nThe paper has solid methodology but two unresolved citation issues.\n\n## Major Findings\n\nOne unresolved citation reference.",
        findings=[
            Finding(
                title="Unsupported claim about novelty",
                severity="major",
                confidence="high",
                rationale="The introduction claims the method is novel, but [12] reports an equivalent approach.",
                quotes=[
                    Quote(
                        section_id="sec_001",
                        char_start=10,
                        char_end=40,
                        text="our novel approach to alignment",
                    )
                ],
                sections_involved=["sec_001"],
                source="challenged",
                perspective="rigorous",
            ),
            Finding(
                title="Style: long sentences in §2",
                severity="minor",
                confidence="medium",
                rationale="Several sentences exceed 40 words in the methods section.",
                quotes=[],
                sections_involved=["sec_002"],
                source="investigate",
            ),
        ],
        deterministic_findings=[
            Finding(
                title="Citation [42] used but not in references",
                severity="major",
                confidence="high",
                rationale="[42] is cited 3 times but has no entry in the references section.",
                quotes=[
                    Quote(
                        section_id="sec_001",
                        char_start=100,
                        char_end=104,
                        text="[42]",
                    )
                ],
                sections_involved=["sec_001"],
                source="deterministic",
            ),
        ],
        edits=[
            Edit(
                title="Tighten opening sentence",
                severity="minor",
                confidence="high",
                rationale="The opening sentence is wordier than it needs to be.",
                section_id="sec_001",
                char_start=0,
                char_end=80,
                original_text="It is generally the case that approaches to this problem have been varied",
                new_text="Approaches to this problem vary widely",
            ),
        ],
        author_questions=[
            AuthorQuestion(
                question="Is the sample size in §3 really N=50, or N=48 as §5 implies?",
                why="The two values appear in different sections without reconciliation.",
                sections_involved=["sec_003", "sec_005"],
            )
        ],
        document_map=[
            SectionCard(section_id="sec_001", title="Introduction", one_line_gist="Frames the problem"),
            SectionCard(section_id="sec_002", title="Methods", one_line_gist="Describes the algorithm"),
        ],
        metrics=ReviewMetrics(
            llm_calls=12, deterministic_findings_count=1, edits_count=1
        ),
    )


# ── Markdown ───────────────────────────────────────────────────────────


def test_markdown_renders_every_section():
    md = render_markdown(_sample_result())
    assert "# Whetstone Review" in md
    assert "Executive Summary" in md
    assert "Author questions" in md
    assert "Edits (1)" in md
    assert "Findings (LLM-investigated)" in md
    assert "Deterministic findings" in md
    assert "Document map" in md
    # Edit shows as diff block
    assert "```diff" in md
    assert "- It is generally the case" in md
    assert "+ Approaches to this problem vary" in md


def test_markdown_writes_to_file_when_path_given(tmp_path: Path):
    out = tmp_path / "review.md"
    md = render_markdown(_sample_result(), output_path=out)
    assert out.exists()
    assert out.read_text() == md


def test_markdown_clean_document_says_so():
    """An empty result emits a clean message, not a wall of empty headings."""
    md = render_markdown(ReviewResult())
    assert "looks clean" in md.lower()


def test_markdown_findings_grouped_by_priority():
    md = render_markdown(_sample_result())
    # MUST FIX appears before CONSIDER in the LLM findings section
    findings_section = md.split("## Findings (LLM-investigated)")[1].split("---")[0]
    assert findings_section.index("### MUST FIX") < findings_section.index(
        "### CONSIDER"
    )


def test_markdown_persona_shown_for_panel_findings():
    md = render_markdown(_sample_result())
    assert "_rigorous_" in md  # the perspective tag


# ── HTML ───────────────────────────────────────────────────────────────


def test_html_renders_to_self_contained_document():
    html = render_html(_sample_result())
    assert "<html" in html.lower()
    assert "Whetstone Review" in html
    # Edits section present
    assert "Tighten opening sentence" in html
    # Findings present
    assert "Unsupported claim" in html
    # Citation deterministic finding present
    assert "[42]" in html
    # Author questions present
    assert "N=50" in html or "N&#x3D;50" in html  # may be HTML-escaped


def test_html_writes_to_file_when_path_given(tmp_path: Path):
    out = tmp_path / "review.html"
    html = render_html(_sample_result(), output_path=out)
    assert out.exists()
    assert out.read_text() == html


def test_html_clean_document_emits_success_callout():
    html = render_html(ReviewResult())
    assert "looks clean" in html.lower()


# ── DOCX adapter (v2 → v1 DocumentPatch) ───────────────────────────────


def test_docx_adapter_converts_edits_to_text_edit_patches():
    from andamentum.whetstone.models import DocumentPatch
    from andamentum.whetstone.renderers.docx import _to_document_patches

    patches = _to_document_patches(_sample_result(), DocumentPatch)
    text_edits = [p for p in patches if p.patch_type == "text_edit"]
    assert len(text_edits) == 1
    assert text_edits[0].text_pattern == "It is generally the case that approaches to this problem have been varied"
    assert text_edits[0].new_text == "Approaches to this problem vary widely"
    assert "wordier" in text_edits[0].explanation.lower()


def test_docx_adapter_converts_findings_with_quotes_to_comments():
    from andamentum.whetstone.models import DocumentPatch
    from andamentum.whetstone.renderers.docx import _to_document_patches

    patches = _to_document_patches(_sample_result(), DocumentPatch)
    comments = [p for p in patches if p.patch_type == "comment"]
    # Two findings have quotes (one LLM, one deterministic). The other LLM
    # finding has no quotes → must be skipped (no anchor).
    assert len(comments) == 2
    titles_in_comments = " ".join(p.comment_text for p in comments)
    assert "Unsupported claim" in titles_in_comments
    assert "[42]" in titles_in_comments


def test_docx_adapter_skips_findings_without_quotes():
    """A finding without quotes can't be anchored to a Word comment."""
    from andamentum.whetstone.models import DocumentPatch
    from andamentum.whetstone.renderers.docx import _to_document_patches

    result = ReviewResult(
        findings=[
            Finding(
                title="No quote",
                severity="moderate",
                confidence="high",
                rationale="floating finding",
                quotes=[],  # no anchor
                sections_involved=["sec_001"],
                source="investigate",
            ),
        ],
    )
    patches = _to_document_patches(result, DocumentPatch)
    assert patches == []


def test_docx_confidence_levels_map_to_floats():
    """v2 uses low/medium/high; v1's DocumentPatch wants a 0..1 float."""
    from andamentum.whetstone.renderers.docx import _confidence_to_float

    assert 0.0 < _confidence_to_float("low") < _confidence_to_float("medium")
    assert _confidence_to_float("medium") < _confidence_to_float("high") <= 1.0


def test_docx_strips_duplicate_executive_summary_heading():
    """The report header supplies its own Executive Summary heading; the
    summary's leading one is dropped so it isn't shown twice."""
    from andamentum.whetstone.renderers.docx import _strip_leading_exec_heading

    out = _strip_leading_exec_heading("## Executive Summary\n\nProse here.\n\n## MUST FIX\n\nx")
    assert not out.lstrip().startswith("## Executive Summary")
    assert out.startswith("Prose here.")
    assert "## MUST FIX" in out  # other headings preserved


def test_docx_leaves_non_exec_heading_summary_untouched():
    from andamentum.whetstone.renderers.docx import _strip_leading_exec_heading

    s = "Some summary with no leading heading."
    assert _strip_leading_exec_heading(s) == s


# ── DOCX adapter — panel mode ──────────────────────────────────────────


def _panel_result() -> ReviewResult:
    """A minimal ReviewResult with panel-mode payload populated."""
    from andamentum.whetstone import (
        ExpertProfile,
        ExpertReview,
        PanelSynthesis,
    )

    return ReviewResult(
        summary="",  # synthesise didn't run; only panel synthesis present
        expert_profiles=[
            ExpertProfile(
                name="Dr. Jane Doe",
                position="Professor of Test Science, Test U.",
                education="PhD, Test University, 1995",
                contributions="Pioneered methodology X; co-authored Y.",
                research="Studies Z in real and simulated systems.",
                discipline="Testology",
            ),
        ],
        expert_reviews=[
            ExpertReview(
                expert_name="Dr. Jane Doe",
                discipline="Testology",
                overall_score=8,
                overall_assessment="Solid contribution.",
                scientific_rigor_score=7,
                scientific_rigor_justification="Mostly rigorous.",
                methodology_score=8,
                methodology_justification="Sound design.",
                novelty_score=6,
                novelty_justification="Some novelty.",
                clarity_score=9,
                clarity_justification="Very clear.",
                strengths=["clear", "rigorous", "well-cited"],
                weaknesses=["incremental"],
                recommendation="Minor Revisions",
                recommendation_justification="Tighten the discussion.",
            ),
        ],
        panel_synthesis=PanelSynthesis(
            average_overall_score=8.0,
            score_range="7-9",
            number_of_experts=1,
            consensus_strengths=["clear writing", "sound methodology"],
            consensus_weaknesses=["incremental contribution"],
            divergent_opinions=[],
            scientific_rigor_summary="High rigor across reviewers.",
            methodology_summary="Sound design across reviewers.",
            novelty_summary="Moderate novelty; one expert flagged it as incremental.",
            clarity_summary="Excellent clarity.",
            overall_recommendation="Minor Revisions",
            recommendation_justification=(
                "The contribution is sound and the writing is excellent; "
                "the novelty framing needs tightening."
            ),
            confidence_level="high",
            key_decision_factors=["sound rigor", "clarity"],
            review_summary="Strong submission with minor revision needs.",
            critical_issues=[],
            novelty_findings="",
        ),
    )


def test_docx_panel_synthesis_in_review_summary():
    """Panel synthesis prose is folded into the prepended report."""
    from unittest import mock

    from andamentum.whetstone.renderers import docx as docx_mod

    captured: dict = {}

    def fake_finalize(**kwargs):
        captured.update(kwargs)
        # Return shape matches finalize_reviewed_document (path, result)
        return ("fake/output.docx", object())

    with mock.patch(
        "andamentum.whetstone.docx.finalization.finalize_reviewed_document",
        fake_finalize,
    ):
        docx_mod.render_docx(
            _panel_result(),
            source_docx_path="fake.docx",
            output_path="out.docx",
        )

    summary = captured.get("review_summary", "")
    # Synthesis content slots into ``## Executive Summary`` upstream, so the
    # body itself opens with the headline line — no nested ``## Panel Synthesis``.
    assert "Panel Synthesis" not in summary
    assert "Recommendation: Minor Revisions" in summary
    assert "average score" in summary.lower()
    assert "8.0/10" in summary
    # Reviewer scores section appears with per-criterion bullet rows.
    assert "Reviewer scores" in summary
    assert "Dr. Jane Doe 8" in summary
    assert "Consensus strengths" in summary
    assert "clear writing" in summary


def test_docx_panel_passes_expert_payload_through():
    """expert_reviews + generated_experts are forwarded to the v1 finaliser."""
    from unittest import mock

    from andamentum.whetstone.renderers import docx as docx_mod

    captured: dict = {}

    def fake_finalize(**kwargs):
        captured.update(kwargs)
        return ("fake/output.docx", object())

    with mock.patch(
        "andamentum.whetstone.docx.finalization.finalize_reviewed_document",
        fake_finalize,
    ):
        docx_mod.render_docx(
            _panel_result(),
            source_docx_path="fake.docx",
            output_path="out.docx",
        )

    reviews = captured.get("expert_reviews")
    experts = captured.get("generated_experts")
    assert reviews is not None and len(reviews) == 1
    assert experts is not None and len(experts) == 1
    # The objects passed through are still the v2 pydantic models;
    # finalize_reviewed_document calls model_dump() on them via
    # normalize_to_dict.
    assert reviews[0].expert_name == "Dr. Jane Doe"
    assert experts[0].discipline == "Testology"


def test_docx_review_mode_omits_panel_payload():
    """When the result has no panel data, expert_reviews/generated_experts
    are passed as None so the v1 finaliser skips that section."""
    from unittest import mock

    from andamentum.whetstone.renderers import docx as docx_mod

    captured: dict = {}

    def fake_finalize(**kwargs):
        captured.update(kwargs)
        return ("fake/output.docx", object())

    with mock.patch(
        "andamentum.whetstone.docx.finalization.finalize_reviewed_document",
        fake_finalize,
    ):
        docx_mod.render_docx(
            _sample_result(),
            source_docx_path="fake.docx",
            output_path="out.docx",
        )

    assert captured.get("expert_reviews") is None
    assert captured.get("generated_experts") is None
    # And no Panel Synthesis heading in the summary either
    assert "Panel Synthesis" not in captured.get("review_summary", "")


def test_docx_novelty_findings_routed_to_dedicated_field():
    """Findings with category="novelty" go to novelty_findings, NOT as
    anchored comments — they have no quotes to anchor to."""
    from unittest import mock

    from andamentum.whetstone.models import DocumentPatch
    from andamentum.whetstone.renderers import docx as docx_mod
    from andamentum.whetstone.renderers.docx import _to_document_patches

    novelty_finding = Finding(
        title="Novelty claim contradicted by prior work",
        severity="major",
        confidence="high",
        rationale="Prior work X established this in 2018.",
        quotes=[],
        sections_involved=[],
        source="investigate",
        category="novelty",
    )
    result = ReviewResult(findings=[novelty_finding])

    # Adapter must NOT emit a comment for the novelty finding
    patches = _to_document_patches(result, DocumentPatch)
    assert patches == []

    # render_docx should pass it through novelty_findings instead
    captured: dict = {}

    def fake_finalize(**kwargs):
        captured.update(kwargs)
        return ("fake/output.docx", object())

    with mock.patch(
        "andamentum.whetstone.docx.finalization.finalize_reviewed_document",
        fake_finalize,
    ):
        docx_mod.render_docx(
            result,
            source_docx_path="fake.docx",
            output_path="out.docx",
        )

    novelty_text = captured.get("novelty_findings", "")
    assert "Novelty claim contradicted by prior work" in novelty_text
    assert "2018" in novelty_text
