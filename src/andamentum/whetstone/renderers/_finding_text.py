"""The words a reader sees about one finding, in one place for all three renderers.

A finding reaches the author three ways — a Word comment, a markdown entry and
an HTML callout — and each renderer used to phrase the same facts its own way:
"MUST FIX" in one, "(MAJOR)" in another, "Critical Issues" for a list that
included the minor ones. Anything that names a finding's standing, its fix, or
what the editor did about it lives here, so the three formats say the same
thing in the same words.

Also here: the order a reader meets findings in, and the link between a
finding and the editor's answer to it. Both used to differ by renderer, and
both decide whether the author sees one point once or twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import (
    _DEFAULT_PRIORITY_FROM_SEVERITY,
    DeclinedFinding,
    Edit,
    Finding,
    ReviewResult,
)

PRIORITY_ORDER = ("must_fix", "should_fix", "consider")

# Section headings every renderer uses. "Reference check", not "References":
# the draft has a section of that name, and the report sits inside the draft.
REFERENCE_CHECK_HEADING = "Reference check"
NEXT_STEP_HEADING = "Next step"
PRIORITY_LABEL = {
    "must_fix": "Must fix",
    "should_fix": "Should fix",
    "consider": "Consider",
}


def corroboration_clause(finding: Finding) -> str:
    """One sentence naming who else raised this finding, or ``""``.

    ``corroborated_by`` holds panel perspective names, or the other reviewers
    of a multi-reviewer run that raised the same point — in both cases the
    other party made the same observation, and the sentence says so.

    Checklist findings are the exception and get nothing here. Their
    ``corroborated_by`` holds probe ids, which mean "another check landed on
    this passage", not "another check made this point" — two probes on one
    sentence usually report two different defects. The checklist builder
    already states that in the finding's own text, with the probes' titles
    rather than their ids (see ``checklist.build_result``); a second sentence
    here repeated it in words that claimed agreement it does not have.
    """
    if not finding.corroborated_by or finding.source == "checklist":
        return ""
    return f"Also raised by: {', '.join(finding.corroborated_by)}."


def confidence_label(finding: Finding) -> str:
    """``"low confidence"`` / ``"high confidence"``, or ``""`` for medium.

    Review and checklist mode set every finding to medium: neither has a signal
    that varies it (see ``synth._to_wfinding`` and
    ``checklist.build_result._to_public_finding``). Printing "medium
    confidence" on every entry told the reader nothing and cost a chip on each.
    A value other than medium comes from a producer that measured something,
    and is shown.
    """
    return "" if finding.confidence == "medium" else f"{finding.confidence} confidence"


def severity_note(finding: Finding) -> str:
    """The severity, when it says something the priority group does not.

    Priority is derived from severity (major → must fix, and so on), and no
    producer in the package overrides it, so printing both repeated the group
    heading on every entry. A caller CAN set priority on its own, though, and
    then the two axes differ and both are worth seeing.
    """
    if _DEFAULT_PRIORITY_FROM_SEVERITY.get(finding.severity) == finding.priority:
        return ""
    return f"{finding.severity} severity"


def fix_line(finding: Finding) -> str:
    """``"Fix: …"`` or ``""`` when the producer wrote none."""
    return f"Fix: {finding.fix.strip()}" if finding.fix.strip() else ""


def declined_line(why: str) -> str:
    """What the editor handed back, said to the author.

    ``why`` is the reason no rewrite could answer the finding, not a second
    fix: the finding's own fix line is the one instruction. An earlier wording
    asked the editor for an action here, and on a real manuscript it restated
    the fix nearly word for word, so the comment gave it three times.
    """
    reason = why.strip().rstrip(".")
    if not reason:
        return "Left to you: no rewrite can fix this in place."
    return f"Left to you: no rewrite can fix this in place ({reason})."


def position(finding: Finding) -> int:
    """Where the finding sits in the document; a quote-less one sorts last."""
    starts = [q.char_start for q in finding.quotes if q.char_start < q.char_end]
    return min(starts) if starts else 1 << 62


def reading_order(findings: list[Finding]) -> list[Finding]:
    """Most urgent group first, and within a group, in document order.

    Within a group the findings used to follow the pipeline's own order —
    alphabetical by criterion — so a reader working through "should fix" jumped
    back and forth through the draft. Document order lets them work down it
    once. Every finding in one result addresses the same text (a multi-reviewer
    result included: no round edits), so the offsets are comparable.
    """
    indexed = list(enumerate(findings))
    indexed.sort(
        key=lambda pair: (
            PRIORITY_ORDER.index(pair[1].priority)
            if pair[1].priority in PRIORITY_ORDER
            else len(PRIORITY_ORDER),
            position(pair[1]),
            pair[0],
        )
    )
    return [f for _, f in indexed]


@dataclass
class EditorAnswers:
    """The editor's output, sorted by the finding it answers.

    A rewrite or a refusal that names a finding shown in the report is
    rendered WITH that finding — once. Only what names no such finding (the
    editor's own style edits, or a refusal whose finding is not in the list)
    is rendered on its own.
    """

    edits_by_finding: dict[str, list[Edit]] = field(default_factory=dict)
    declined_by_finding: dict[str, str] = field(default_factory=dict)
    own_edits: list[Edit] = field(default_factory=list)
    unlinked_declined: list[DeclinedFinding] = field(default_factory=list)


def editor_answers(result: ReviewResult) -> EditorAnswers:
    shown = {f.id for f in list(result.findings) + list(result.deterministic_findings)}
    out = EditorAnswers()
    for edit in result.edits:
        if edit.finding_id is not None and edit.finding_id in shown:
            out.edits_by_finding.setdefault(edit.finding_id, []).append(edit)
        else:
            out.own_edits.append(edit)
    for declined in result.declined_findings:
        if declined.finding_id and declined.finding_id in shown:
            out.declined_by_finding[declined.finding_id] = declined.why
        else:
            out.unlinked_declined.append(declined)
    return out


def summary_body(summary: str) -> str:
    """The summary without a leading ``## Summary`` heading.

    Review mode's synthesised summary opens with that heading; a multi-reviewer
    summary and a checklist coverage table do not. Each renderer supplies its
    own heading, so the body is stripped to one shape — otherwise the HTML and
    Word reports printed "Summary" twice, one heading directly under the other.
    """
    text = summary.strip()
    if text.startswith("## Summary"):
        text = text[len("## Summary") :].lstrip()
    return text


def coverage_warning(result: ReviewResult) -> str:
    """One paragraph saying the review is incomplete, or ``""``.

    Every format carries it. It used to reach only the markdown report, so an
    HTML or Word reader of a run where criteria crashed saw a complete-looking
    review.
    """
    parts: list[str] = []
    if result.failed_criteria:
        names = ", ".join(result.failed_criteria)
        noun = "check" if len(result.failed_criteria) == 1 else "checks"
        parts.append(
            f"These {noun} failed to run and found nothing: {names}. "
            "Treat the findings below as incomplete."
        )
    if result.degraded and result.degraded_reason:
        parts.append(f"This run is degraded: {result.degraded_reason}.")
    if not parts:
        return ""
    return "**Partial review.** " + " ".join(parts)


def snippet(text: str, *, limit: int = 80) -> str:
    """A quote shortened for a location line, on one line."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"
