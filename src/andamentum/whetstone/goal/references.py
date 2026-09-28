"""The goal loop's ``citations`` check: the reference checker, every round.

The same checker as ``andamentum-whetstone references``
(:mod:`andamentum.references`), so the loop and the subcommand cannot disagree
about a document: every citation style — numbers in brackets or parentheses,
superscript, author-year, footnotes and endnotes, Pandoc keys — and the
formatting evidence a Word file carries (superscript runs, reference-manager
fields, note marks), read by :func:`~..docx.citation_hints.citation_hints`.

Two differences from the subcommand, both because this runs every round:

- **No Crossref lookup.** The goal is that the citations and the list agree,
  which an edit to the draft can bring about. Whether Crossref indexes a work
  is not something an edit can change, and asking it again every round would
  spend the service's patience on the same answer.
- **The loop's model settles what the text leaves open** — how the document
  cites, or whether an ambiguous number is a citation — exactly as
  ``--model`` does for the subcommand. A document the text settles on its own
  (most of them) makes no model call.

The problems come back as findings, located in the document the round reads,
so the editor answers them beside the review's own; the loop confirms the goal
from the next round's check, not from the editor's word.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..schemas import Finding

if TYPE_CHECKING:
    from ..reviewed_document import ReviewedDocument


@dataclass(frozen=True)
class ReferenceGoalReport:
    """One round's reading of the references.

    ``unresolvable`` is set when the goal cannot be checked at all — no
    reference list and no citation notes, or no way to tell how the document
    cites — and then nothing else is meaningful.
    """

    satisfied: bool
    reason: str
    findings: list[Finding] = field(default_factory=list)
    unresolvable: str | None = None


async def check_reference_goal(
    document: "ReviewedDocument", *, model: str
) -> ReferenceGoalReport:
    from andamentum.references import check_references

    from ..docx.citation_hints import citation_hints
    from ..references_check import to_review_result

    report = await check_references(
        document.text,
        sections=[(s.title, s.start, s.end) for s in document.sections],
        hints=citation_hints(document.docx_path, expected_text=document.text),
        model=model,
        lookup=False,
    )
    if not report.found_list and report.style is None:
        return ReferenceGoalReport(
            satisfied=False,
            reason="",
            unresolvable=(
                "No reference list (a section headed References, "
                "Bibliography, Works cited or Literature cited) and no "
                "citation notes were found, so there is nothing to check the "
                "citations against."
            ),
        )
    if report.style is None or report.style.style == "undecided":
        why = report.style.reason if report.style is not None else ""
        return ReferenceGoalReport(
            satisfied=False,
            reason="",
            unresolvable=f"It is not clear how the document cites its references. {why}".strip(),
        )

    findings = to_review_result(report, document=document).findings
    unjudged = sum(1 for j in report.judged if j.reason.startswith("not judged"))
    if not findings and not unjudged:
        return ReferenceGoalReport(
            satisfied=True,
            reason="Every citation points to an entry in the list, and every entry is cited.",
        )
    parts = []
    if findings:
        parts.append(
            f"{len(findings)} reference problem(s): "
            + "; ".join(f.title for f in findings[:5])
            + ("; …" if len(findings) > 5 else "")
        )
    if unjudged:
        # Only a failed model call leaves candidates unjudged here, since the
        # loop always passes its model. Until they are read, "every entry is
        # cited" cannot be confirmed.
        parts.append(f"{unjudged} possible citation(s) could not be judged")
    return ReferenceGoalReport(satisfied=False, reason=". ".join(parts) + ".", findings=findings)
