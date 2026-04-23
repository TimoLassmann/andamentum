"""End-to-end test: construct a ReviewResult by hand, run every renderer.

Does not exercise the LLM — that's covered by hand-run smoke examples in
the README. This test proves the shape of ReviewResult flows through all
renderers without error.
"""

import tempfile
from pathlib import Path

from andamentum.whetstone import (
    DocumentIssue,
    DocumentPatch,
    ReviewResult,
    apply_patches,
    render_diff,
    render_html,
)


def _result_review() -> ReviewResult:
    return ReviewResult(
        task="review",
        issues=[
            DocumentIssue(
                issue_type="major",
                category="methodology",
                title="Missing control",
                description="No control group described.",
                recommendation="Add a control arm.",
                agent_type="methodology",
            ),
            DocumentIssue(
                issue_type="strength",
                category="clarity",
                title="Clear abstract",
                description="The abstract is tight and specific.",
                agent_type="clarity",
            ),
        ],
    )


def _result_edit() -> ReviewResult:
    return ReviewResult(
        task="edit",
        patches=[
            DocumentPatch(
                patch_type="text_edit",
                text_pattern="The data shows",
                new_text="The data show",
                explanation="Subject-verb agreement",
                confidence=0.95,
            ),
        ],
    )


def test_diff_renders_review():
    out = render_diff(
        patches=[],
        issues=_result_review().issues,
        original_content="",
    )
    assert "Missing control" in out


def test_html_renders_review():
    html = render_html(result=_result_review(), original_content="")
    assert html.startswith("<!DOCTYPE html>")
    assert "Missing control" in html


def test_html_renders_edit():
    html = render_html(result=_result_edit(), original_content="The data shows trends")
    assert "The data shows" in html
    assert "The data show" in html


def test_apply_patches_round_trip():
    text = "The data shows trends"
    revised = apply_patches(text, _result_edit().patches)
    assert revised == "The data show trends"


def test_write_html_to_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.html"
        path.write_text(render_html(result=_result_review(), original_content=""))
        assert path.exists()
        assert path.stat().st_size > 1000  # non-trivial HTML
