"""The reference check, as a whetstone result.

``andamentum.references`` does the checking; this module only feeds it the
reviewed text and turns what it found into the same ``ReviewResult`` the
review modes produce, so the three renderers — Word comments included — show
reference problems exactly as they show any other finding.

The text checked is :func:`~.reviewed_document.open_source`'s: for a ``.docx``
the projection of its own paragraphs. So every problem's offsets are addresses
in the Word file being annotated, and the comment lands on the citation or
entry it is about with no searching (see :mod:`.reviewed_document`).

A model is used only when the text leaves open how the document cites, or
whether a given number is a citation — and only when one is passed. The
online lookup sends each reference entry's text to Crossref; the draft's own
text never leaves the machine (a model call sees short excerpts around the
candidates, and goes wherever *model* points — a local model keeps it local).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .schemas import Finding, Quote, ReviewResult, SectionCard

if TYPE_CHECKING:
    from andamentum.core.decisions import DecisionModel
    from andamentum.references import ReferenceReport

    from .reviewed_document import ReviewedDocument

REFERENCE_CHECKER = "Reference checker"
REFERENCE_CHECKER_AUTHOR = "Reference checker (andamentum-whetstone, AI)"


def _offline(kind: str) -> bool:
    from andamentum.references import LOOKUP_KINDS

    return kind not in LOOKUP_KINDS


async def check_open_document(
    document: "ReviewedDocument",
    *,
    model: str | None,
    lookup: bool,
    mailto: str | None = None,
    confirm_own_draft: bool = False,
    decision_model: "DecisionModel | None" = None,
) -> ReviewResult:
    """Check the references of a document the caller already opened.

    The confidentiality tripwire runs first, over the same text a review
    scans — the body with the running headers and notes, where a
    "CONFIDENTIAL" stamp usually sits — because with *lookup* the reference
    entries leave the machine here, and in the default review this runs
    before the review's own tripwire.
    """
    from andamentum.references import check_references

    from ._confidentiality import check_confidentiality
    from .docx.citation_hints import citation_hints

    if not confirm_own_draft:
        check_confidentiality(document.confidentiality_text)

    report = await check_references(
        document.text,
        sections=[(s.title, s.start, s.end) for s in document.sections],
        # Superscript runs and reference-manager fields, read from the file
        # the text is the projection of. A document converted from another
        # format has neither, and the checker works from the text alone.
        hints=citation_hints(document.docx_path, expected_text=document.text),
        model=model,
        lookup=lookup,
        mailto=mailto,
        decision_model=decision_model,
    )
    return to_review_result(report, document=document)


def to_review_result(
    report: "ReferenceReport", *, document: "ReviewedDocument"
) -> ReviewResult:
    """The report as findings located in *document*, plus a summary."""
    findings: list[Finding] = []
    for problem in report.problems:
        section_id = _section_at(document, problem.start)
        findings.append(
            Finding(
                title=problem.headline,
                severity=problem.severity,
                # The number checks are exact; a lookup verdict rests on one
                # service's coverage, so it is not claimed as certain.
                confidence="high" if _offline(problem.kind) else "medium",
                rationale=problem.issue,
                fix=problem.fix,
                quotes=[
                    Quote(
                        section_id=section_id,
                        char_start=problem.start,
                        char_end=problem.end,
                        text=document.text[problem.start : problem.end],
                    )
                ],
                sections_involved=[section_id] if section_id else [],
                source="deterministic" if _offline(problem.kind) else "lookup",
                category=f"references:{problem.kind}",
                # Signs its Word comment, and tells the editor and the reader
                # which check raised it, beside a review's own findings.
                perspective=REFERENCE_CHECKER,
            )
        )
    unchecked = [r for r in report.lookups if r.status == "service_error"]
    degraded_reason = ""
    if unchecked:
        degraded_reason = (
            f"{len(unchecked)} of {len(report.lookups)} references could not "
            f"be looked up, because Crossref did not answer; they are "
            f"unchecked, not wrong"
        )
    return ReviewResult(
        summary=_summary(report),
        findings=findings,
        degraded=bool(unchecked),
        degraded_reason=degraded_reason,
        document_map=[
            SectionCard(section_id=s.id, title=s.title) for s in document.sections
        ],
    )


def _section_at(document: "ReviewedDocument", offset: int) -> str:
    for section in document.sections:
        if section.start <= offset < section.end:
            return section.id
    return ""


_STYLE_NAMES = {
    "numeric_brackets": "numbers in square brackets, like [3]",
    "numeric_superscript": "superscript numbers",
    "numeric_parentheses": "numbers in parentheses, like (3)",
    "author_year": "author and year, like (Smith et al., 2020)",
    "notes": "footnotes or endnotes",
    "pandoc_keys": "Pandoc citation keys, like [@smith2020]",
}


def style_name(style: str) -> str:
    """A citation style in the words a reader uses. The one wording: the
    reference report and the review prompt's citation line both use it."""
    return _STYLE_NAMES.get(style, style)


def _summary(report: "ReferenceReport") -> str:
    notes_only = not report.found_list and report.style is not None
    if not report.found_list and not notes_only:
        return (
            "No reference list was found, so nothing was checked. The checker "
            "looks for a section headed References, Bibliography, Works cited "
            "or Literature cited."
        )
    if not report.entries and not notes_only:
        return (
            "A reference list was found, but no entries could be read from it "
            "(entries need a year, a DOI or a number), so nothing was checked."
        )
    style = report.style
    lines = [
        "No bibliography: the notes were checked on their own, and their full "
        "citations stand in for its entries."
        if notes_only
        else f"Checked {len(report.entries)} references."
    ]
    if style is None or style.style == "undecided":
        lines += [
            "",
            f"- **How the document cites:** undecided. {style.reason if style else ''}",
            "- **Citations:** not checked, because it is not clear which "
            "numbers or names in the text are citations.",
        ]
    else:
        how = {"evidence": "from the text", "model": "by the model"}[style.decided_by]
        lines.append("")
        lines.append(
            f"- **How the document cites:** {style_name(style.style)} "
            f"(decided {how}: {style.reason})"
        )
        accepted = [j for j in report.judged if j.is_citation]
        rejected = [
            j
            for j in report.judged
            if not j.is_citation
            and not j.reason.startswith(("not judged", "rule: not the document"))
        ]
        unjudged = [j for j in report.judged if j.reason.startswith("not judged")]
        by_model = sum(1 for j in report.judged if j.reason == "model")
        lines.append(
            f"- **Citations read:** {len(accepted)}"
            + (
                f"; {len(rejected)} set aside as not citations ({_reasons(rejected)})"
                if rejected
                else ""
            )
            + (f"; {by_model} judged by the model" if by_model else "")
            + (
                f"; {len(unjudged)} left unjudged — pass --model to judge them"
                if unjudged
                else ""
            )
            + "."
        )
        offline = [p for p in report.problems if _offline(p.kind)]
        lines.append(
            (
                "- **The notes, checked in order:** "
                if notes_only
                else "- **Citations against the list:** "
            )
            + (
                f"{len(offline)} problem{'s' if len(offline) != 1 else ''}."
                if offline
                else (
                    "every short form and Ibid. has a source to point to."
                    if notes_only
                    else "every citation points to an entry, and every entry is cited."
                )
            )
        )
    if report.looked_up:
        counts = {
            status: 0
            for status in ("found", "not_found", "details_differ", "service_error")
        }
        for result in report.lookups:
            counts[result.status] += 1
        lines.append(
            f"- **Crossref lookup:** {counts['found']} found as written, "
            f"{counts['details_differ']} with details that differ, "
            f"{counts['not_found']} not found"
            + (
                f", {counts['service_error']} not checked (Crossref did not answer)"
                if counts["service_error"]
                else ""
            )
            + "."
        )
        lines.append(
            "- *Only the reference entries were sent to Crossref, never the "
            'text of the draft. "Not found" means Crossref has no matching '
            "record; some conference papers and reports are not indexed "
            "there.*"
        )
    else:
        lines.append(
            "- **Crossref lookup:** off, so the works themselves were not checked."
        )
    return "\n".join(lines)


def _reasons(judged: list) -> str:
    counts: dict[str, int] = {}
    for j in judged:
        why = j.reason.removeprefix("rule: ")
        counts[why] = counts.get(why, 0) + 1
    return "; ".join(
        f"{n} {why}" for why, n in sorted(counts.items(), key=lambda kv: -kv[1])[:3]
    )
