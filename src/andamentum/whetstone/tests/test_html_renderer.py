"""Tests for the typeset-based HTML renderer.

The renderer transforms a ReviewResult into andamentum.typeset atoms and
delegates styling to typeset.render(). Tests assert on structural markers
(headers, section wrappers, severity classes) rather than exact bytes so
that typeset styling changes don't break them.
"""

from typing import Literal

from andamentum.whetstone.issues import DocumentIssue
from andamentum.whetstone.models import DocumentPatch
from andamentum.whetstone.orchestrator import ReviewResult

IssueSeverity = Literal["major", "minor", "suggestion", "strength"]


def _edit(pattern: str, new: str) -> DocumentPatch:
    return DocumentPatch(
        patch_type="text_edit",
        text_pattern=pattern,
        new_text=new,
        explanation="test",
        confidence=0.9,
    )


def _issue(severity: IssueSeverity, title: str) -> DocumentIssue:
    return DocumentIssue(
        issue_type=severity,
        category="test",
        title=title,
        description="test description",
        agent_type="test",
    )


def test_html_contains_disclaimer():
    from andamentum.whetstone.renderers.html import render_html

    result = ReviewResult(task="edit", patches=[_edit("a", "b")])
    html = render_html(result=result, original_content="a")

    assert "<!DOCTYPE html>" in html
    assert "own drafts" in html.lower() or "not for peer review" in html.lower()


def test_html_edit_task_structure():
    from andamentum.whetstone.renderers.html import render_html

    result = ReviewResult(
        task="edit",
        patches=[_edit("teh", "the"), _edit("data is", "data are")],
    )
    html = render_html(result=result, original_content="teh data is")

    assert "Whetstone" in html
    assert "teh" in html
    assert "data is" in html
    assert "data are" in html


def test_html_review_task_with_issues():
    from andamentum.whetstone.renderers.html import render_html

    result = ReviewResult(
        task="review",
        issues=[
            _issue("major", "Missing control"),
            _issue("minor", "Typo in abstract"),
            _issue("strength", "Clear writing"),
        ],
    )
    html = render_html(result=result, original_content="x")

    assert "Missing control" in html
    assert "Typo in abstract" in html
    assert "Clear writing" in html


def test_html_empty_result():
    from andamentum.whetstone.renderers.html import render_html

    result = ReviewResult(task="review")
    html = render_html(result=result, original_content="x")

    # Should still produce a valid HTML document, not an error.
    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html


def test_html_panel_task_renders_experts():
    from andamentum.whetstone.agents.output_models import (
        ExpertProfile,
        ExpertReviewOutput,
    )
    from andamentum.whetstone.renderers.html import render_html

    profile = ExpertProfile(
        name="Dr. Jane Doe",
        position="Professor of Test Science",
        education="PhD, Test University",
        contributions="Pioneered test-driven development of tests.",
        research="Studies how tests test tests.",
        discipline="Testology",
    )
    review = ExpertReviewOutput(
        expert_name="Dr. Jane Doe",
        discipline="Testology",
        overall_score=8,
        overall_assessment="Solid work.",
        scientific_rigor_score=7,
        scientific_rigor_justification="Decent rigor.",
        methodology_score=8,
        methodology_justification="Good methods.",
        novelty_score=6,
        novelty_justification="Somewhat novel.",
        clarity_score=9,
        clarity_justification="Very clear.",
        strengths=["clear", "novel"],
        weaknesses=["too short"],
        recommendation="Minor Revisions",
        recommendation_justification="Tighten the discussion.",
    )

    result = ReviewResult(
        task="panel",
        disciplines=["Testology"],
        expert_profiles=[profile],
        expert_reviews=[review],
    )
    html = render_html(result=result, original_content="x")

    assert "Dr. Jane Doe" in html
    assert "Testology" in html
    assert "Minor Revisions" in html


def test_html_renders_checklist_items():
    from andamentum.whetstone import ChecklistItem, ReviewResult
    from andamentum.whetstone.renderers import render_html

    result = ReviewResult(
        task="checklist",
        checklist=[
            ChecklistItem(
                name="Abstract wordcount",
                status="pass",
                notes="240 words",
                category="abstract",
            ),
            ChecklistItem(
                name="Ethics statement",
                status="fail",
                notes="Missing",
                category="statements",
            ),
        ],
    )
    html = render_html(result=result, original_content="")
    assert "Abstract wordcount" in html
    assert "Ethics statement" in html
    # Some fail indicator somewhere in the output
    assert "fail" in html.lower() or "✗" in html
