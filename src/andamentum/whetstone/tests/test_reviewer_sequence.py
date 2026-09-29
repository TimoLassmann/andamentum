"""Several reviewers in one invocation.

``--reviewer claims,methods`` runs them IN SEQUENCE over one unchanged text;
no round edits. With ``--editor`` (or ``editor`` in the list) one editor pass
follows the last reviewer and rewrites against all of their findings at once —
see ``test_final_editor.py``.

A run with more than one reviewer ends with a reconciliation, which is likewise
not a flag. It groups the findings that make the same point, and each group is
merged into ONE finding naming every reviewer that raised it — the author reads
a point once, however many reviewers raised it.
"""

from __future__ import annotations

from typing import Any

import pytest

from andamentum.whetstone import cli as cli_mod
from andamentum.whetstone.schemas import Finding, Quote, ReviewerHeader, ReviewResult
from andamentum.whetstone.v3.reconcile import (
    AgreedPoint,
    Disagreement,
    ReviewerReconciliation,
    combine_reviews,
)
from andamentum.whetstone.v3.reviewers import PROFILES


# ── parsing ─────────────────────────────────────────────────────────────────


def test_one_name_is_a_list_of_one() -> None:
    assert cli_mod._parse_reviewers("claims") == ["claims"]


def test_order_is_preserved_because_order_is_significant() -> None:
    """The list is a SEQUENCE: the reports stack in the order the reviewers
    ran, so claims,methods and methods,claims do not produce the same file."""
    assert cli_mod._parse_reviewers("methods,claims") == ["methods", "claims"]
    assert cli_mod._parse_reviewers("claims,methods") == ["claims", "methods"]


def test_whitespace_around_names_is_tolerated() -> None:
    assert cli_mod._parse_reviewers(" claims , methods ") == ["claims", "methods"]


def test_an_unknown_name_names_itself_and_the_alternatives(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_mod._parse_reviewers("claims,statistician")
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "statistician" in err
    for name in PROFILES:
        assert name in err


def test_a_repeated_reviewer_is_refused_not_collapsed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each reviewer is a full review, so a repeat costs a full review. That
    makes it a typo every time rather than something to silently de-duplicate."""
    with pytest.raises(SystemExit) as exc:
        cli_mod._parse_reviewers("claims,methods,claims")
    assert exc.value.code == 1
    assert "more than once" in capsys.readouterr().err


def test_an_empty_value_is_refused() -> None:
    with pytest.raises(SystemExit):
        cli_mod._parse_reviewers("  ,  ")


# ── the reconciliation, merged and rendered ─────────────────────────────────


def _finding(text: str, *, severity: Any = "moderate", start: int = 0) -> Finding:
    return Finding(
        title=text,
        severity=severity,
        confidence="medium",
        rationale=f"{text}, explained.",
        quotes=[Quote(section_id="s1", char_start=start, char_end=start + 5, text="x")],
        sections_involved=["s1"],
    )


def _reviews() -> list[tuple[str, ReviewResult]]:
    return [
        (
            "Claims reviewer",
            ReviewResult(
                summary="## Summary\n\nClaims read.\n\n## Weaknesses\n\n- a weakness bullet",
                findings=[_finding("overclaim"), _finding("typo", severity="minor")],
            ),
        ),
        (
            "Methods reviewer",
            ReviewResult(
                summary="## Summary\n\nMethods read.",
                findings=[
                    _finding("conclusion outruns data", severity="major", start=20),
                    _finding("no power analysis"),
                ],
            ),
        ),
    ]


def _rec(*, agreed=(), disagreements=(), verdict="") -> ReviewerReconciliation:
    return ReviewerReconciliation(
        verdict=verdict, agreed=list(agreed), disagreements=list(disagreements)
    )


def test_a_point_two_reviewers_raised_is_one_finding() -> None:
    """The reason the merge exists: the same point from two reviewers must
    reach the author once, not once per reviewer."""
    rec = _rec(
        agreed=[
            AgreedPoint(
                issue="The claim outruns the data", finding_ids=["R1.1", "R2.1"]
            )
        ]
    )
    combined, groups = combine_reviews(_reviews(), reconciliation=rec)
    titles = [f.title for f in combined.findings]
    assert titles == ["conclusion outruns data", "typo", "no power analysis"]
    merged = combined.findings[0]
    # The most severe member is kept, and names the other reviewer.
    assert merged.severity == "major"
    assert merged.perspective == "Methods reviewer"
    assert merged.corroborated_by == ["Claims reviewer"]
    # Both passages survive on the one finding.
    assert [q.char_start for q in merged.quotes] == [20, 0]
    assert groups[0].raised_by == ["Claims reviewer", "Methods reviewer"]


def test_unmerged_findings_carry_their_reviewer() -> None:
    combined, _ = combine_reviews(_reviews(), reconciliation=_rec())
    assert [f.perspective for f in combined.findings] == [
        "Claims reviewer",
        "Claims reviewer",
        "Methods reviewer",
        "Methods reviewer",
    ]


def test_a_point_from_one_reviewer_only_is_not_agreement() -> None:
    """Two findings of ONE reviewer are not independent agreement, and merging
    them is consolidation's job, not this one's."""
    rec = _rec(agreed=[AgreedPoint(issue="x", finding_ids=["R1.1", "R1.2"])])
    combined, groups = combine_reviews(_reviews(), reconciliation=rec)
    assert groups == []
    assert len(combined.findings) == 4


def test_an_unknown_or_reused_id_is_ignored_not_trusted() -> None:
    rec = _rec(
        agreed=[
            AgreedPoint(issue="first", finding_ids=["[r1.1]", "R2.1", "R9.9"]),
            # R1.1 already belongs to the first point; what is left is one
            # reviewer, so the second point merges nothing.
            AgreedPoint(issue="second", finding_ids=["R1.1", "R2.2"]),
        ]
    )
    combined, groups = combine_reviews(_reviews(), reconciliation=rec)
    assert [g.issue for g in groups] == ["first"]
    assert len(combined.findings) == 3


def test_a_failed_reconciliation_keeps_everything_and_says_so() -> None:
    combined, groups = combine_reviews(_reviews(), reconciliation=None)
    assert groups == []
    assert len(combined.findings) == 4
    assert "may appear more than once" in combined.summary


def test_the_summary_leads_with_agreement_and_does_not_restate_findings() -> None:
    rec = _rec(
        verdict="The draft overclaims.",
        agreed=[
            AgreedPoint(
                issue="The claim outruns the data", finding_ids=["R1.1", "R2.1"]
            )
        ],
        disagreements=[
            Disagreement(
                about="whether the claim is overstated",
                positions=[
                    "Claims reviewer: overstated",
                    "Framing reviewer: undersold",
                ],
            )
        ],
    )
    combined, _ = combine_reviews(_reviews(), reconciliation=rec)
    out = combined.summary
    assert out.startswith("The draft overclaims.")
    assert "Raised by more than one reviewer (1)" in out
    assert "The claim outruns the data *(Claims reviewer, Methods reviewer)*" in out
    # Both sides of a disagreement survive: choosing is the author's call.
    assert "Claims reviewer: overstated" in out
    assert "Framing reviewer: undersold" in out
    # Each reviewer's synopsis, but not its weaknesses — those are findings,
    # and the findings follow the summary.
    assert "*Claims reviewer:* Claims read." in out
    assert "a weakness bullet" not in out
    assert "no power analysis" not in out


def test_a_reviewer_that_does_not_fit_the_document_is_named_in_the_summary() -> None:
    reviews = _reviews()
    reviews[1][1].reviewer_header = ReviewerHeader(
        identity="Methods reviewer",
        description="d",
        fit_warning="This reviewer is meant for academic papers.",
    )
    combined, _ = combine_reviews(reviews, reconciliation=_rec())
    assert (
        "Methods reviewer: This reviewer is meant for academic papers."
        in combined.summary
    )


def test_each_reviewers_comments_are_signed_by_that_reviewer() -> None:
    from andamentum.whetstone.renderers.docx import _to_annotations

    rec = _rec(agreed=[AgreedPoint(issue="p", finding_ids=["R1.1", "R2.1"])])
    combined, _ = combine_reviews(_reviews(), reconciliation=rec)
    _, notes = _to_annotations(
        combined,
        author="Line editor (AI)",
        comment_authors={
            "Claims reviewer": "Claims (AI)",
            "Methods reviewer": "Methods (AI)",
        },
    )
    assert [n.author for n in notes] == ["Methods (AI)", "Claims (AI)", "Methods (AI)"]
    assert "Also raised by: Claims reviewer." in notes[0].body
