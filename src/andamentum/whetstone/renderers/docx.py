"""Word document renderer for ``ReviewResult``.

An ``Edit`` and a quoted ``Finding`` already know where they go. Their offsets
index the text the reviewer read, and for both input paths that text IS the
projection of the document being annotated (see
:mod:`andamentum.whetstone.reviewed_document`) — so this renderer converts a
finding into a :class:`~andamentum.whetstone.docx.annotate.Note` and an edit
into a :class:`~andamentum.whetstone.docx.annotate.Rewrite` by copying two
integers, and the docx layer places them without looking for the text.

  • Edit    → Rewrite(start, end, new_text)          — a tracked change
  • Finding → Note(start, end, "<title>\\n\\n<rationale>")  — a comment

It used to build ``DocumentPatch`` objects carrying ``text_pattern`` and throw
the offsets away, and the docx layer then searched a *different* rendering of
the document for that pattern. Measured on one real manuscript: 15 of 43
rewrites lost, every one reporting a nearest match at 0.97 similarity or
better, because Docling wrote ``aim 3,4 .`` where Word held
``aim3,4.``. That path survives only for ``andamentum-whetstone
apply-patches``, whose input is a pattern list from an outside agent that never
saw the document's coordinates.

**Every finding is also listed in the prepended report**, via the
``critical_issues`` payload — the comment is an anchored convenience, not the
only copy. Two things still stop a comment reaching the reader: a quote-less
finding has nothing to attach to (all checklist absence findings, and any mode
can produce one), and a comment is invisible in any viewer that does not render
docx markup. What is no longer on that list is "the anchor missed".

Panel mode (``mode="panel"``) is ALSO rendered: expert biosketches, scored
expert reviews, and the panel synthesis are folded into the prepended review
report. v2's ExpertProfile / ExpertReview pydantic schemas are field-compatible
with the v1 reader (which uses ``model_dump()`` then categorises by ``_score``
/ ``_justification`` suffix).

Novelty findings (category="novelty") are pulled out and passed as
``novelty_findings`` so they appear in their own report subsection rather than
mixed with the criterion findings — they are document-level claims with no
passage to point at.

**Sequential rounds are one parameter: ``author``.** A second review takes the
first's output as its input, and nothing has to be told that is what is
happening — the file already carries round one's changes, and the projection
reads them as Word will once they are accepted, so round two reviews the
revised text and its offsets address the revised document. An edit landing
inside text round one inserted nests as ``<w:ins A><w:del B>``, which is Word's
own markup for "B revised what A wrote"; that falls out of the surgery rule
(rewrite the runs the span covers, in place) rather than being coded for.

Two consequences worth knowing before running two rounds:

*One name twice is legal and lossy.* ``w:author`` is the only thing Word groups
changes by, so a second round under the same name merges into the first in the
reviewing pane and the author cannot accept one round while holding the other
back. Nothing is corrupted — every change keeps its own revision id and is
accepted or rejected individually — but the attribution is gone. Pass a
different name for a second round.

*Reports stack.* Each round prepends its own report above the last rather than
replacing it: an earlier report is the earlier reviewer's record and the author
may not have worked through it yet. Every report is bracketed by a bookmark and
excluded from the next round's reading, so what accumulates is pages in the
file, not text under review.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .._watermark import (
    BANNER_TITLE,
    DISCLAIMER_SHORT,
    provenance_line,
    stamp_docx_core_properties,
)
from ..schemas import Edit, Finding, Quote, ReviewResult
from ._finding_text import (
    NEXT_STEP_HEADING,
    REFERENCE_CHECK_HEADING,
    PRIORITY_LABEL,
    confidence_label,
    corroboration_clause,
    coverage_warning,
    declined_line,
    editor_answers,
    fix_line,
    reading_order,
    snippet,
    summary_body,
)
from ._panel_layout import body_markdown
from ._reviewer_header import configuration_markdown, header_markdown

if TYPE_CHECKING:
    from ..docx.annotate import Note, Rewrite
    from ..reviewed_document import ReviewedDocument

logger = logging.getLogger("andamentum.whetstone.renderers.docx")


# The default tracked-change author. Deliberately includes "(AI)" so the
# Word File→Info pane and the tracked-changes attribution make AI
# authorship unmissable. Override via --allow-author-override on the CLI
# only; misrepresenting AI-generated edits as a human reviewer's may
# constitute research misconduct under most institutional codes.
DEFAULT_AI_AUTHOR = "andamentum-whetstone (AI)"


def render_docx(
    result: ReviewResult,
    *,
    document: "ReviewedDocument",
    output_path: str | Path,
    author: str = DEFAULT_AI_AUTHOR,
    comment_authors: Mapping[str, str] | None = None,
    model: str | None = None,
    visible_watermark: bool = True,
) -> Any:
    """Render a ReviewResult as a .docx with track changes + comments.

    Edits become tracked changes (deletion of the original span, insertion of
    the new span). The executive summary becomes a prepended review report.

    **Every finding is listed in that report**, and separately anchored as a
    Word comment where it carries a quote. The list is the record; the comment
    is the convenience — a quote-less finding has nothing to attach to, and a
    comment is invisible in any viewer that does not render docx markup. See
    :func:`_findings_as_issues`.

    For panel-mode results, the prepended report includes the panel synthesis
    prose, scored per-expert reviews, and expert biosketches.

    Parameters
    ----------
    result:
        The ReviewResult to render.
    document:
        The document that was reviewed — its text and the ``.docx`` that text
        is the projection of. Every offset in *result* is an address in
        ``document.text``, and the docx layer refuses to run if the file does
        not read as exactly that. Passing a different document is therefore a
        loud failure, not a quietly misplaced annotation.
    output_path:
        Where to write the reviewed .docx.
    author:
        Author name attributed to the tracked changes and comments. This is
        the whole of the sequential-round surface: give a second round over
        the first round's output a different name and Word shows two
        reviewers. See the module docstring for what the same name twice
        costs.
    comment_authors:
        Who signs a finding's comment, by the finding's ``perspective``. A
        multi-reviewer run passes one entry per reviewer, so each reviewer's
        comments are grouped under its own name in Word's reviewing pane. A
        finding whose perspective is not in the map — or every finding, when
        the map is ``None`` — is signed by *author*.

    Returns
    -------
    The :class:`~andamentum.whetstone.docx.annotate.AnnotationReport` (what
    reached the document, and what the renderer dropped on purpose).
    """
    # Defer the heavy docx import: the track-changes machinery is large, and
    # we want the renderers importable without dragging it in by default.
    from andamentum.whetstone.docx.finalization import finalize_annotated_document

    rewrites, notes = _to_annotations(
        result, author=author, comment_authors=comment_authors
    )
    findings = list(result.findings) + list(result.deterministic_findings)

    review_summary = summary_body(
        _strip_leading_exec_heading(result.summary or _fallback_summary(result))
    )
    # Said at the top of the report, not only in the markdown output: a Word
    # reader of a run where checks crashed otherwise saw a complete review.
    warning = coverage_warning(result)
    if warning:
        review_summary = f"{warning}\n\n{review_summary}"
    questions = _collect_author_questions(result)
    if questions:
        review_summary = f"{review_summary}\n\n{questions}"
    # In the summary block, which is where the report's own text goes: a
    # section of its own would mean a keyword through two builder signatures
    # (see the reviewer block below for the same trade).
    if result.reference_summary:
        review_summary += (
            f"\n\n**{REFERENCE_CHECK_HEADING}**\n\n{result.reference_summary}"
        )
    if result.next_steps:
        review_summary += f"\n\n**{NEXT_STEP_HEADING}**\n\n" + "\n\n".join(
            result.next_steps
        )
    if result.panel_synthesis is not None:
        # Panel mode: the synthesis is the authoritative summary. ``result.summary``
        # is intentionally empty (see panel_synthesise.py); render the synthesis
        # body (no ``## Panel synthesis`` heading — content goes inside the
        # ``## Executive Summary`` already added by _build_report_header).
        review_summary = body_markdown(
            result.panel_synthesis, list(result.expert_reviews)
        )

    # The reviewer block, when a reviewer other than the default one ran. It
    # goes into the review summary rather than above the report title: the
    # title is written three call frames down, inside the report builder, and
    # threading a keyword through two signatures to move this block up a
    # heading is not worth what it costs to read. Accepted asymmetry with the
    # markdown and HTML renderers, where the block sits in the prelude.
    if result.reviewer_header is not None:
        review_summary = f"{header_markdown(result.reviewer_header)}\n{review_summary}"

    if visible_watermark:
        banner = (
            f"> **⚠ {BANNER_TITLE}.** {DISCLAIMER_SHORT}\n"
            f">\n"
            f"> *Produced by {provenance_line(model=model)}.*\n\n"
        )
        review_summary = banner + (review_summary or "")

    novelty_findings_text = _collect_novelty_findings(result)

    _, report = finalize_annotated_document(
        Path(document.docx_path),
        rewrites=rewrites,
        notes=notes,
        reviewed_text=document.text,
        review_summary=review_summary,
        issues_count=len(findings),
        # The findings themselves, not just their count. Without this the
        # report announced N issues and listed none — see _findings_as_issues.
        critical_issues=_findings_as_issues(result) or None,
        output_path=Path(output_path),
        author=author,
        # Panel-mode payload (None / empty when not in panel mode)
        expert_reviews=list(result.expert_reviews) or None,
        generated_experts=list(result.expert_profiles) or None,
        # Novelty-check payload (empty string when not used)
        novelty_findings=novelty_findings_text,
        declined_findings=_collect_declined(result),
        other_edits=_collect_own_edits(result),
        configuration=(
            configuration_markdown(result.reviewer_header)
            if result.reviewer_header is not None
            else ""
        ),
        novelty_claims=_collect_novelty_claims(result),
    )
    # Invisible metadata: always on, regardless of visible_watermark.
    stamp_docx_core_properties(Path(output_path), model=model)
    return report


# ── Adapter: findings and edits → resolved annotations ──────────────────


def _to_annotations(
    result: ReviewResult,
    *,
    author: str,
    comment_authors: Mapping[str, str] | None = None,
) -> tuple[list["Rewrite"], list["Note"]]:
    """Turn the review into annotations addressed at their own coordinates.

    No text is matched, normalised or searched for here, and nothing in this
    function may start doing so: the offsets came from the document under
    annotation, and reaching for its text again would be the deleted
    coordinate system growing back. What the docx layer receives is two
    integers, a body, and an author.

    One comment per finding, and the editor's answer rides IN it: a rewrite
    that answers a finding is announced in that finding's comment, and a
    refusal is stated there as what is left to the author. A tracked change
    that answers nothing (the editor's own edit) gets no comment: on one real
    manuscript a comment per such edit was 51 of 81 comments, and the findings
    were lost among them. Their reasons are listed once in the report instead
    (:func:`_collect_own_edits`).
    """
    from andamentum.whetstone.docx.annotate import Note, Rewrite

    rewrites = [
        Rewrite(
            start=edit.char_start,
            end=edit.char_end,
            new_text=edit.new_text,
            author=author,
            label=edit.title,
        )
        for edit in result.edits
        if edit.new_text
    ]
    answers = editor_answers(result)

    notes: list[Note] = []
    for finding in list(result.findings) + list(result.deterministic_findings):
        # Novelty findings are surfaced separately via novelty_findings; they
        # are document-level claims with no passage to point at.
        if finding.category == "novelty":
            continue
        anchor = _anchor_quote(finding)
        if anchor is None:
            continue
        notes.append(
            Note(
                start=anchor.char_start,
                end=anchor.char_end,
                body=_comment_body(
                    finding,
                    anchor=anchor,
                    edits=answers.edits_by_finding.get(finding.id, []),
                    declined_why=answers.declined_by_finding.get(finding.id, ""),
                ),
                author=(comment_authors or {}).get(finding.perspective or "", author),
                label=finding.title,
            )
        )
    return rewrites, notes


def _anchor_quote(finding: Finding) -> Quote | None:
    """The located span a Word comment attaches to, or ``None``.

    ``None`` means the finding is document-level (or names something missing),
    so it belongs in the report body rather than in the margin. Both the
    annotation adapter and the report collector route on this one predicate,
    so a finding can never fall between them.

    A zero-width or reversed span is treated as no anchor: it is not an
    address, and the alternative — letting the docx layer refuse the whole
    render over one malformed finding — would lose the other forty.
    """
    for quote in finding.quotes:
        if quote.char_start < quote.char_end:
            return quote
    return None


def _comment_body(
    finding: Finding,
    *,
    anchor: Quote | None = None,
    edits: list[Edit] | None = None,
    declined_why: str = "",
) -> str:
    """The text of one Word comment, readable on its own.

    A margin comment is read beside the sentence, without the report open, so
    it carries everything the author needs to act: how urgent it is, the
    problem, the fix, what the editor did about it, and who else raised it. It
    used to open with the criterion's name and carry no severity at all.
    """
    parts = [f"{PRIORITY_LABEL.get(finding.priority, 'Consider')}: {finding.title}"]
    if finding.rationale.strip():
        parts.append(finding.rationale.strip())
    if fix_line(finding):
        parts.append(fix_line(finding))
    for edit in edits or []:
        overlaps = anchor is not None and (
            edit.char_start < anchor.char_end and anchor.char_start < edit.char_end
        )
        if overlaps:
            parts.append("Suggested rewrite: the tracked change here.")
        else:
            parts.append(
                "Suggested rewrite: the tracked change at "
                f"\u201c{snippet(edit.original_text, limit=60)}\u201d."
            )
    if declined_why:
        parts.append(declined_line(declined_why))
    clause = _corroboration_clause(finding)
    if clause:
        parts.append(clause)
    return "\n\n".join(parts)


def _corroboration_clause(finding: Finding) -> str:
    return corroboration_clause(finding)


def _strip_leading_exec_heading(summary: str) -> str:
    """Drop a leading ``## Executive Summary`` heading from the summary.

    The docx report header (``_build_report_header``) supplies its own
    ``## Executive Summary`` heading; v2's synthesised summary leads with the
    same heading, which produced "Executive Summary" twice in the rendered
    .docx. Stripped only here — the markdown/HTML renderers have no report
    header and keep the heading.
    """
    from andamentum.whetstone.docx.constants import EXECUTIVE_SUMMARY_HEADER

    stripped = summary.lstrip()
    if stripped.startswith(EXECUTIVE_SUMMARY_HEADER):
        return stripped[len(EXECUTIVE_SUMMARY_HEADER) :].lstrip("\n")
    return summary


def _fallback_summary(result: ReviewResult) -> str:
    """Generate a plain summary for when Synthesise didn't run."""
    n_findings = len(result.findings) + len(result.deterministic_findings)
    n_edits = len(result.edits)
    n_questions = len(result.author_questions)
    parts = [
        f"Whetstone review: {n_findings} finding(s), {n_edits} edit(s), "
        f"{n_questions} author question(s)."
    ]
    if result.expert_reviews:
        parts.append(f"Panel: {len(result.expert_reviews)} expert review(s).")
    if not (result.summary or "").strip():
        parts.append("Run with model= for synthesis prose.")
    return " ".join(parts)


def _collect_declined(result: ReviewResult) -> str:
    """The editor's refusals that name no listed finding, as report markdown.

    A refusal that names a finding is stated WITH that finding — in its
    comment and in its report entry — so listing it here as well gave the
    author the same item three times. What remains is a refusal whose finding
    is not in this result, which should not happen and is shown rather than
    dropped if it does.

    Empty string when the editor did not run, which is NOT the same as "every
    finding was fixed"; the report simply has nothing to say about edits.
    """
    declined = editor_answers(result).unlinked_declined
    if not declined:
        return ""
    # Asterisks, not underscores: the markdown-to-Word renderer in
    # review_report.py recognises *italic* and leaves _italic_ as literal
    # characters in the document.
    lines = [
        "*The editor read these and judged that rewriting a passage cannot "
        "answer them. Each needs a decision, new content, or a restructure.*",
        "",
    ]
    for d in declined:
        line = f"- {d.issue}"
        if d.why:
            # One line: the report's markdown reader has no nested lists, and a
            # "  - " sub-bullet reached Word as literal characters.
            line += f" *{declined_line(d.why)}*"
        lines.append(line)
    return "\n".join(lines)


def _collect_novelty_claims(result: ReviewResult) -> str:
    """What the draft stakes as new, as markdown for the prepended report.

    Distinct from ``_collect_novelty_findings`` and easy to confuse with it:
    that one carries the VERDICTS of a literature search that was asked for,
    this one carries the CLAIMS, found on every run and checked against
    nothing. Both can appear in one report; the headings and the caveat line
    are what keep them apart for a reader.

    Empty string when the extractor found no assertion of being first, novel
    or unique — the ordinary case for a tutorial or a short story, and no
    reason to put a notice in the report.
    """
    if not result.novelty_claims:
        return ""
    lines = [
        "*These are checked against nothing. This is what the draft asserts, "
        "not what the literature says.*",
        "",
    ]
    for claim in result.novelty_claims:
        line = f"- \u201c{claim.claim_text}\u201d"
        if claim.why_load_bearing:
            # One line, for the same reason as in _collect_declined.
            line += f" *Why it matters:* {claim.why_load_bearing}"
        lines.append(line)
    lines += [
        "",
        "*To check them automatically, re-run with --check-novelty. That sends "
        "these sentences to a search engine.*",
    ]
    return "\n".join(lines)


def _collect_novelty_findings(result: ReviewResult) -> str:
    """Concatenate novelty-category findings into a markdown block.

    The v1 finalize_reviewed_document accepts a ``novelty_findings``
    string and renders it as its own report subsection. Pulling the
    novelty findings out of the lens-finding pool prevents them from
    becoming anchored comments against fictional text spans (they are
    document-level claims, not section-anchored observations).
    """
    novelty = [
        f
        for f in (list(result.findings) + list(result.deterministic_findings))
        if f.category == "novelty"
    ]
    if not novelty:
        return ""

    lines = [
        f"{len(novelty)} novelty claim(s) flagged by literature search:",
        "",
    ]
    for f in novelty:
        confidence = confidence_label(f)
        lines.append(
            f"- **{f.title}** ({f.severity}{', ' + confidence if confidence else ''})"
        )
        if f.rationale:
            # Indent rationale under its bullet
            for ln in f.rationale.splitlines():
                lines.append(f"    {ln}")
        lines.append("")
    return "\n".join(lines)


def _collect_own_edits(result: ReviewResult) -> str:
    """The editor's own edits with their reasons, as report markdown, or ``""``.

    Each is a tracked change in the text already; this list is where its
    reason is, since ``Rewrite.label`` reaches only the logs. One line each —
    the report's markdown reader has no nested lists.
    """
    edits = editor_answers(result).own_edits
    if not edits:
        return ""
    section_titles = {c.section_id: c.title for c in result.document_map}
    lines = [
        "*Tracked changes the editor made on its own, answering no finding. "
        "Each is marked in the text; this is why.*",
        "",
    ]
    for edit in sorted(edits, key=lambda e: e.char_start):
        where = section_titles.get(edit.section_id, edit.section_id)
        reason = edit.rationale.strip() or edit.title
        lines.append(
            f"- *{where}*, \u201c{snippet(edit.original_text, limit=60)}\u201d: {reason}"
        )
    return "\n".join(lines)


def _collect_author_questions(result: ReviewResult) -> str:
    """Questions only the author can answer, as report markdown, or ``""``.

    The .docx used to drop them — so a goal-loop run that stopped with a
    question for the author said so only in the markdown and HTML outputs.
    """
    if not result.author_questions:
        return ""
    lines = ["**Questions only you can answer**", ""]
    for q in result.author_questions:
        line = f"- {q.question}"
        if q.why:
            line += f" *{q.why}*"
        lines.append(line)
    return "\n".join(lines)


def _finding_location(finding: Finding, *, section_titles: Mapping[str, str]) -> str:
    """Where the reader should look for this finding, in prose.

    The report list has to stand on its own: a Word comment sits beside the
    text, but a quote-less finding never had anywhere to point. Sections are
    named by their title — the id ("s3") is the pipeline's address, not the
    author's.
    """
    sections = ", ".join(
        section_titles.get(sid, sid) for sid in finding.sections_involved
    )
    anchor = _anchor_quote(finding)
    quote = anchor.text.strip() if anchor is not None else ""
    if quote:
        return (
            f"{sections} — \u201c{snippet(quote)}\u201d"
            if sections
            else f"\u201c{snippet(quote)}\u201d"
        )
    return sections or "the document as a whole"


def _findings_as_issues(result: ReviewResult) -> list[dict[str, Any]]:
    """Every finding, as the issue records the .docx report body renders.

    The report is the complete record and the comments are the convenience:
    a comment is invisible in any viewer that does not render docx markup
    (Preview, Quick Look, Word with markup hidden), and a finding with no quote
    has no comment at all. So every finding is listed — in the order the
    author works through them: most urgent first, then down the document.

    Each entry carries its fix and what the editor did about it, so a finding
    is one item here rather than an issue in one list and a refusal in
    another. Novelty findings are excluded — ``_collect_novelty_findings``
    already gives them their own subsection.
    """
    findings = [
        f
        for f in (list(result.findings) + list(result.deterministic_findings))
        if f.category != "novelty"
    ]
    section_titles = {c.section_id: c.title for c in result.document_map}
    answers = editor_answers(result)
    issues: list[dict[str, Any]] = []
    for finding in reading_order(findings):
        notes = [finding.rationale.strip()] if finding.rationale.strip() else []
        if answers.edits_by_finding.get(finding.id):
            notes.append("Suggested rewrite: shown as a tracked change.")
        why = answers.declined_by_finding.get(finding.id, "")
        if why:
            notes.append(declined_line(why))
        clause = _corroboration_clause(finding)
        if clause:
            notes.append(clause)
        issues.append(
            {
                "title": finding.title,
                "description": "\n\n".join(notes),
                # The same urgency words the comments and the other two formats
                # use; the report formatter upper-cases them.
                "severity": PRIORITY_LABEL.get(finding.priority, "Consider"),
                "location": _finding_location(finding, section_titles=section_titles),
                "recommendation": finding.fix.strip(),
                # The reviewer that raised it, in panel mode and in a
                # multi-reviewer run; empty otherwise.
                "source_reviewers": [finding.perspective]
                if finding.perspective
                else [],
            }
        )
    return issues
