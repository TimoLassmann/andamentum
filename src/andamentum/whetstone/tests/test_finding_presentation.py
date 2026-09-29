"""How one finding reaches the author, in every format.

A finding carries a short headline, the problem, and one fix. The editor's
answer to it — a rewrite or a refusal — is linked by id, and every renderer
shows the finding and its answer as ONE item: in a Word comment, in the report
entry, and in the markdown and HTML reports. The same item used to appear as a
comment, a tracked change with no reason attached, an entry in "Critical
Issues" and again under "Left to the Author".
"""

from __future__ import annotations

from typing import Any

from andamentum.whetstone.renderers import render_html, render_markdown
from andamentum.whetstone.renderers._finding_text import reading_order
from andamentum.whetstone.renderers.docx import _findings_as_issues, _to_annotations
from andamentum.whetstone.schemas import (
    DeclinedFinding,
    Edit,
    Finding,
    Quote,
    ReviewResult,
    SectionCard,
)
from andamentum.whetstone.v3.model import Span
from andamentum.whetstone.v3.review import Finding as V3Finding
from andamentum.whetstone.v3.synth import _to_wfinding, synopsis_of


def _finding(
    *,
    fid: str,
    start: int,
    severity: str = "moderate",
    title: str = "The claim outruns the data",
    fix: str = "Report the n per group.",
) -> Finding:
    return Finding(
        id=fid,
        title=title,
        severity=severity,  # type: ignore[arg-type]
        confidence="medium",
        rationale="You say it holds for everyone, but only 12 were tested.",
        fix=fix,
        quotes=[
            Quote(
                section_id="s1", char_start=start, char_end=start + 10, text="it holds"
            )
        ],
        sections_involved=["s1"],
    )


def _edit(*, start: int, finding_id: str | None, severity: str = "moderate") -> Edit:
    return Edit(
        title="Hedge",
        severity=severity,  # type: ignore[arg-type]
        rationale="'Always' claimed more than the data showed.",
        section_id="s1",
        char_start=start,
        char_end=start + 10,
        original_text="it always holds",
        new_text="it held in our sample",
        finding_id=finding_id,
    )


def _result(**kw: Any) -> ReviewResult:
    kw.setdefault("summary", "## Summary\n\nA short study.")
    kw.setdefault("document_map", [SectionCard(section_id="s1", title="Methods")])
    return ReviewResult(**kw)


# ── the Word comment ────────────────────────────────────────────────────────


def test_a_comment_says_how_urgent_what_is_wrong_and_what_to_do() -> None:
    _, notes = _to_annotations(
        _result(findings=[_finding(fid="f1", start=0)]), author="t"
    )
    body = notes[0].body
    assert body.startswith("Should fix: The claim outruns the data")
    assert "only 12 were tested" in body
    assert "Fix: Report the n per group." in body


def test_the_rewrite_that_answers_a_finding_is_announced_in_its_comment() -> None:
    """One thread per issue: the comment says the tracked change beside it is
    its fix, instead of the change arriving with no reason attached."""
    result = _result(
        findings=[_finding(fid="f1", start=0)],
        edits=[_edit(start=2, finding_id="f1")],
    )
    rewrites, notes = _to_annotations(result, author="t")
    assert len(rewrites) == 1
    assert len(notes) == 1  # no second comment for the linked edit
    assert "Suggested rewrite: the tracked change here." in notes[0].body


def test_a_rewrite_elsewhere_is_pointed_to_by_its_text() -> None:
    result = _result(
        findings=[_finding(fid="f1", start=0)],
        edits=[_edit(start=40, finding_id="f1")],
    )
    _, notes = _to_annotations(result, author="t")
    assert "tracked change at “it always holds”" in notes[0].body


def test_a_refusal_is_stated_in_the_comment_it_refuses() -> None:
    result = _result(
        findings=[_finding(fid="f1", start=0)],
        declined_findings=[
            DeclinedFinding(
                criterion="c",
                issue="i",
                why="the value is not in the text",
                finding_id="f1",
            )
        ],
    )
    _, notes = _to_annotations(result, author="t")
    body = notes[0].body
    assert (
        "Left to you: no rewrite can fix this in place (the value is not in the text)."
        in body
    )
    # The fix is given once — the refusal is a reason, not a second fix.
    assert body.count("Report the n per group.") == 1


def test_the_editors_own_edits_get_no_comment_and_are_listed_once_with_a_reason() -> (
    None
):
    """A comment per unlinked edit was 51 of 81 comments on a real manuscript,
    and the findings were lost among them. The reason is listed once in the
    report instead — Rewrite.label reaches only the logs."""
    from andamentum.whetstone.renderers.docx import _collect_own_edits

    result = _result(
        edits=[
            _edit(start=30, finding_id=None, severity="minor"),
            _edit(start=0, finding_id=None, severity="moderate"),
        ]
    )
    rewrites, notes = _to_annotations(result, author="t")
    assert len(rewrites) == 2
    assert notes == []
    listed = _collect_own_edits(result)
    assert listed.count("'Always' claimed more than the data showed.") == 2
    assert "*Methods*" in listed


# ── the Word report ─────────────────────────────────────────────────────────


def test_the_report_lists_findings_urgent_first_then_down_the_document() -> None:
    result = _result(
        findings=[
            _finding(fid="late", start=50, title="Late"),
            _finding(fid="minor", start=0, severity="minor", title="Minor"),
            _finding(fid="early", start=10, title="Early"),
            _finding(fid="major", start=90, severity="major", title="Major"),
        ]
    )
    issues = _findings_as_issues(result)
    assert [i["title"] for i in issues] == ["Major", "Early", "Late", "Minor"]
    assert issues[0]["severity"] == "Must fix"
    assert issues[0]["recommendation"] == "Report the n per group."
    # Section by title, not by the pipeline's id.
    assert issues[0]["location"].startswith("Methods")


def test_a_linked_refusal_is_not_listed_twice_in_the_report() -> None:
    from andamentum.whetstone.renderers.docx import _collect_declined

    result = _result(
        findings=[_finding(fid="f1", start=0)],
        declined_findings=[
            DeclinedFinding(criterion="c", issue="i", why="needs data", finding_id="f1")
        ],
    )
    assert _collect_declined(result) == ""
    assert "(needs data)" in _findings_as_issues(result)[0]["description"]


# ── markdown and HTML ───────────────────────────────────────────────────────


def test_markdown_shows_the_rewrite_under_its_finding_and_nowhere_else() -> None:
    result = _result(
        findings=[_finding(fid="f1", start=0)],
        edits=[_edit(start=2, finding_id="f1")],
    )
    md = render_markdown(result)
    assert "Other edits" not in md
    finding_at = md.index("The claim outruns the data")
    assert md.index("**Fix:** Report the n per group.") > finding_at
    assert md.index("+ it held in our sample") > finding_at


def test_html_shows_the_fix_and_the_rewrite_with_the_finding() -> None:
    result = _result(
        findings=[_finding(fid="f1", start=0)],
        edits=[_edit(start=2, finding_id="f1")],
    )
    html = render_html(result)
    assert "Other edits" not in html
    assert "Report the n per group." in html
    assert "it held in our sample" in html


def test_the_summary_heading_appears_once() -> None:
    html = render_html(_result())
    assert html.count(">Summary<") == 1
    md = render_markdown(_result())
    assert md.count("## Summary") == 1


def test_a_partial_review_is_flagged_in_every_format() -> None:
    result = _result(failed_criteria=["Evaluations"])
    assert "Partial review." in render_markdown(result)
    assert "Partial review." in render_html(result)


# ── the pipeline's side ─────────────────────────────────────────────────────


def test_the_headline_heads_the_finding_and_the_criterion_moves_to_category() -> None:
    f = V3Finding(
        criterion="Evidence support",
        issue="You give no source.",
        quote="q",
        headline="Improvement figure has no source",
        fix="Cite Table 2.",
        span=Span(section_id="s1", start=0, end=1),
    )
    public = _to_wfinding(f, model=None)  # type: ignore[arg-type]
    assert public.title == "Improvement figure has no source"
    assert public.category == "evidence support"
    assert public.fix == "Cite Table 2."
    assert public.id == f.id


def test_a_finding_without_a_headline_falls_back_to_its_criterion() -> None:
    f = V3Finding(criterion="Story", issue="x", quote="q")
    assert _to_wfinding(f, model=None).title == "Story"  # type: ignore[arg-type]


def test_reading_order_is_stable_for_quote_less_findings() -> None:
    a = Finding(title="a", severity="moderate", confidence="medium", rationale="")
    b = Finding(title="b", severity="moderate", confidence="medium", rationale="")
    assert [f.title for f in reading_order([a, b])] == ["a", "b"]


def test_synopsis_of_takes_the_first_part_only() -> None:
    assert synopsis_of("## Summary\n\nOne.\n\n## Strengths\n\n- s") == "One."
    assert synopsis_of("Plain text.") == "Plain text."


def test_a_fix_written_into_the_issue_is_cut_out_of_it() -> None:
    """gpt-5.4-nano on a real manuscript ended issues with "Fix: …" and then
    filled the fix field too, so the comment gave the fix twice."""
    from andamentum.whetstone.v3.prompt_blocks import split_fix

    assert split_fix("The metric is unclear. Fix: State it.", "State the metric.") == (
        "The metric is unclear.",
        "State the metric.",
    )
    # An empty fix field takes the text after the marker.
    assert split_fix("Unclear. fix: State it.", "") == ("Unclear.", "State it.")
    assert split_fix("No marker.", "x") == ("No marker.", "x")
