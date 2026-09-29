"""The editor's refusals must reach the reader.

``declined_findings`` was collected into graph state and then dead-ended — it
reached no result type and no renderer. Every run with the editor on produced
five to seven reasoned refusals that nobody ever saw, and a finding with no
edit against it was indistinguishable from one the editor never looked at.
"""

from __future__ import annotations

from andamentum.whetstone.renderers.html import render_html
from andamentum.whetstone.renderers.markdown import render_markdown
from andamentum.whetstone.schemas import DeclinedFinding, ReviewResult


def _declined(n: int = 1) -> list[DeclinedFinding]:
    return [
        DeclinedFinding(
            criterion=f"Criterion {i}",
            issue=f"issue {i} that a rewrite cannot answer",
            why=f"needs a restructure, reason {i}",
            section_id="s1",
        )
        for i in range(1, n + 1)
    ]


def test_markdown_reports_what_the_editor_handed_back() -> None:
    out = render_markdown(ReviewResult(summary="s", declined_findings=_declined(2)))
    assert "Left to the author (2)" in out
    assert "issue 1 that a rewrite cannot answer" in out
    assert "needs a restructure, reason 1" in out  # the REASON, not just the issue
    assert "Left to you:" in out


def test_html_reports_what_the_editor_handed_back() -> None:
    out = render_html(ReviewResult(summary="s", declined_findings=_declined(1)))
    assert "Left to the author (1)" in out
    assert "needs a restructure, reason 1" in out


def test_no_declined_section_when_the_editor_did_not_run() -> None:
    """An empty list means the editor did not run, NOT that all was fixed."""
    out = render_markdown(ReviewResult(summary="s"))
    assert "Left to the author" not in out


def test_to_review_result_carries_declined_through() -> None:
    """The graph collects these; `to_review_result` is where they were lost.

    This is the join the plumbing exists for: a caller can now ask, per
    finding, whether it was fixed, refused with a reason, or never reached.
    """
    from andamentum.whetstone.v3.model import DocumentModel, Section
    from andamentum.whetstone.v3.synth import StructuredReview, to_review_result

    model = DocumentModel(
        source="hello world",
        sections=[Section(id="s1", title="T", text="hello world", start=0, end=11)],
    )
    result = to_review_result(
        model,
        [],
        StructuredReview(synopsis="fine"),
        [],
        declined=_declined(3),
    )
    assert len(result.declined_findings) == 3
    assert result.declined_findings[0].why == "needs a restructure, reason 1"
    assert "Left to the author (3)" in render_markdown(result)


def test_the_docx_block_uses_markers_the_word_renderer_understands() -> None:
    """`review_report.py` renders *italic*, and leaves _italic_ as literal
    underscores in the document. Caught in a real run, where the .docx read
    `_The editor read these…_` with the markers visible."""
    from andamentum.whetstone.renderers.docx import _collect_declined

    block = _collect_declined(ReviewResult(summary="s", declined_findings=_declined(1)))
    assert block.startswith("*The editor read these")
    assert "_The editor" not in block
