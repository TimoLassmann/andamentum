"""Adapter: verified ``ProbeFinding``s → the public ``ReviewResult``.

The renderer contract is ``whetstone.schemas.ReviewResult``, and checklist mode
fills it with the same field meanings review mode uses so all three renderers
consume it unchanged. Where the two modes differ, and why:

- ``summary`` is the deterministic coverage table (see :mod:`.summary`), not
  LLM-written prose. Two runs of the same bank therefore produce summaries you
  can diff, which is the entire reason this mode exists beside review mode.
  Absence findings used to be restated under the table, because the .docx once
  listed only quoted findings and they had nowhere else to appear. Every format
  now lists every finding, so the restatement only showed each one twice.
- ``source`` on each finding is ``"checklist"``, so a consumer holding a mixed
  pile of findings can tell a quota-driven probe hit from a criterion-cascade
  observation without inspecting anything else.
- ``confidence`` is one constant value for every finding (see
  :func:`_to_public_finding`), where review mode varies it per finding.
- ``edits``, ``author_questions`` and ``metrics.reflection_rounds_used`` stay
  empty/zero: this mode extracts and verifies instances of named defect
  classes, and has no rewrite, question-asking or reflection stage to fill them.

No LLM call happens here. The stage is a pure function of the findings, the
bank, the crash list, and the verification drop counts.
"""

from __future__ import annotations

import logging

from ...schemas import Finding as PublicFinding
from ...schemas import Quote, ReviewMetrics, ReviewResult, SectionCard
from ..model import DocumentModel
from .probes import Probe, ProbeFinding
from .summary import coverage_table

logger = logging.getLogger("andamentum.whetstone.pipeline")

# Aggregate-failure gate, matching the pipeline default the review and panel
# graphs already use (``ReviewDeps.soft_failure_threshold`` in ``pipeline/graph.py`` and
# ``PanelDeps.soft_failure_threshold``). Held as a constant rather than a
# parameter because the deliverable of this mode is the coverage table, which
# reports every individual crash by name in the row that would otherwise have
# carried a result — the flag is only the coarse "this run lost most of its
# coverage" signal on top of that.
_SOFT_FAILURE_THRESHOLD = 0.5


def _append_sentence(body: str, *, tail: str) -> str:
    """Append one sentence to a body that may or may not end in punctuation.

    Behind ``*`` because both halves are ``str`` and transposing them produces
    a plausible-looking sentence in the wrong order rather than an error.
    """
    if not body:
        return tail
    separator = " " if body.endswith((".", "!", "?", ":")) else ". "
    return f"{body}{separator}{tail}"


def _join_titles(titles: list[str]) -> str:
    """``"A"`` / ``"A and B"`` / ``"A, B and C"`` — for reader-facing prose."""
    if len(titles) == 1:
        return titles[0]
    return f"{', '.join(titles[:-1])} and {titles[-1]}"


def _rationale(finding: ProbeFinding) -> str:
    """The finding's body text, made self-contained for absence findings.

    A presence finding arrives at the reader with a quote under it, so the
    issue statement can lean on that context. An absence finding has no quote
    by construction — there is nothing in the document to point at — so the one
    sentence in ``rationale`` is the entire finding as far as the reader is
    concerned. Dropping ``missing_item`` here would leave sentences like "the
    reader cannot reproduce this" with no statement of what is actually
    missing.

    The containment check avoids the common case where the probe already named
    the missing item inside its issue statement, which would otherwise read as
    a stutter.
    """
    issue = finding.issue.strip()
    missing = finding.missing_item.strip()
    if not missing or missing.casefold() in issue.casefold():
        return issue
    return _append_sentence(issue, tail=f"Missing from the document: {missing}.")


def _to_public_finding(
    finding: ProbeFinding,
    model: DocumentModel,
    *,
    title: str,
    corroborator_titles: list[str],
) -> PublicFinding:
    """Map one internal probe finding onto the public finding schema.

    ``title`` is the probe's short label and ``rationale`` is the issue text —
    never the same string in both. Every renderer shows title AND rationale
    (docx concatenates them, markdown and HTML stack them), so mapping one
    statement into both fields shows the reader the same sentence twice.

    ``Quote`` and ``Span`` are the same coordinate system — the whole reviewed
    document — so the span is copied across untouched. It used to be rebased
    onto the section, and the docx renderer then had to find the quote by text
    instead; that search lost 15 of 43 rewrites on one real manuscript. Any
    arithmetic here is a bug: the offsets are addresses, and an address you
    adjust is an address you have lost.

    ``corroborator_titles`` names the OTHER checks that landed on this same
    passage. It is written into ``corroborated_by`` (as probe ids) for machine
    consumers, and stated as a sentence in the rationale (as probe titles) for
    the reader. The sentence is the ONLY place a reader sees it: the renderers
    deliberately print no corroboration clause for a checklist finding (see
    ``renderers._finding_text.corroboration_clause``), because theirs claims
    that the other party raised the same point, and here it only flagged the
    same passage.

    The sentence says the checks flagged the same passage, not that they agree.
    They have different instructions, so the ordinary case is one passage
    carrying two distinct defects — telling that apart from two readings of one
    defect needs their prose compared, which is the model call this mode
    deliberately does not make. See :mod:`.corroborate`.
    """
    quotes: list[Quote] = []
    if finding.shape == "presence":
        if finding.span is None:
            # verify.py is supposed to have dropped these. Rendering one
            # without an anchor is survivable; doing it silently is not,
            # because the reader gets an unquoted presence claim.
            logger.warning(
                "[checklist.build_result] presence finding from probe %r has "
                "no located span — rendering it without a quote anchor: %r",
                finding.probe_id,
                finding.quote[:60],
            )
        else:
            quotes.append(
                Quote(
                    section_id=finding.span.section_id,
                    char_start=finding.span.start,
                    char_end=finding.span.end,
                    text=finding.quote,
                )
            )
    rationale = _rationale(finding)
    if corroborator_titles:
        rationale = _append_sentence(
            rationale,
            tail=(
                f"This passage was also flagged by {_join_titles(corroborator_titles)}."
            ),
        )
    return PublicFinding(
        title=title,
        severity=finding.severity,
        # One value for every checklist finding, deliberately. The two shapes
        # pass different gates in .verify (a located quote vs. search terms
        # none of which appear), but neither gate checks what confidence is
        # about — whether the flagged thing is really a defect. That judgment
        # came from a single probe call in both cases, so splitting the value
        # by shape would dress the anchoring check up as a second opinion.
        # "medium" says: mechanically anchored, judged once. Agreement between
        # probes is the signal that does vary, and it travels in
        # ``corroborated_by`` and the rationale sentence above.
        confidence="medium",
        rationale=rationale,
        fix=finding.fix.strip(),
        quotes=quotes,
        sections_involved=[finding.span.section_id] if finding.span else [],
        source="checklist",
        # The probe id, verbatim — it is the join key back to the bank, the
        # coverage-table row label, and what corroboration cross-links on.
        # Lower-casing or prettifying it here would break that join.
        category=finding.probe_id,
        corroborated_by=list(finding.corroborated_by),
    )


def build_checklist_result(
    model: DocumentModel,
    findings: list[ProbeFinding],
    probes: list[Probe],
    *,
    failed_probe_ids: list[str],
    unverifiable_counts: dict[str, int],
    llm_calls: int,
) -> ReviewResult:
    """Assemble the run's ``ReviewResult``. Deterministic; no model is called.

    ``failed_probe_ids`` names the probes that crashed and contributed nothing.
    They are surfaced three ways, because a checklist that quietly shrank is
    worse than one that reports fewer findings: as ``not run`` rows in the
    coverage table (the channel every renderer shows, since every renderer
    prints ``summary``), as ``failed_criteria``, and — once the failure rate
    reaches ``_SOFT_FAILURE_THRESHOLD``, or when the bank was empty — as
    ``degraded`` with a reason. The last two are weaker channels than the
    table: at the time of writing only the markdown renderer prints
    ``failed_criteria``, and no renderer reads ``degraded`` at all, so both are
    effectively for machine consumers and the CLI. A degraded result still
    carries every finding acquired; the flag says the coverage is not what a
    green run would have produced.

    ``unverifiable_counts`` maps probe id → how many of that probe's findings
    verification dropped. It is passed straight through to
    :func:`.summary.coverage_table`, which owns how a dropped finding is
    reported; nothing else here reads it.
    """
    title_by_id = {p.id: p.title for p in probes}
    unknown_ids = sorted({f.probe_id for f in findings} - set(title_by_id))
    if unknown_ids:
        logger.warning(
            "[checklist.build_result] %d finding(s) reference probe id(s) "
            "absent from the bank; their raw id is used as the finding title: %s",
            sum(1 for f in findings if f.probe_id in set(unknown_ids)),
            ", ".join(unknown_ids),
        )

    failed = list(failed_probe_ids)
    total = len(probes)
    degraded_reason = ""
    if total == 0:
        # Without this, an empty bank produces degraded=False, findings=[],
        # failed_criteria=[] — byte-identical to a document that passed every
        # check. The two must not be confusable: one is a clean draft, the
        # other is a misconfigured run that looked at nothing.
        degraded_reason = (
            "the checklist ran with an empty probe bank — nothing was checked"
        )
        logger.warning("[checklist.build_result] degraded — %s", degraded_reason)
    elif len(failed) / total >= _SOFT_FAILURE_THRESHOLD:
        failed_labels = ", ".join(title_by_id.get(i, i) for i in failed)
        degraded_reason = (
            f"{len(failed)}/{total} checklist probes crashed and checked "
            f"nothing: {failed_labels}"
        )
        logger.warning("[checklist.build_result] degraded — %s", degraded_reason)

    gist_by_section = {g.section_id: g.gist for g in model.gists}
    summary = coverage_table(
        probes,
        findings,
        failed_probe_ids=failed,
        unverifiable_counts=unverifiable_counts,
    )

    return ReviewResult(
        summary=summary,
        findings=[
            _to_public_finding(
                f,
                model,
                title=title_by_id.get(f.probe_id, f.probe_id),
                corroborator_titles=[
                    title_by_id.get(other, other) for other in f.corroborated_by
                ],
            )
            for f in findings
        ],
        edits=[],
        # Human-readable labels, matching what review mode puts here
        # (``Criterion.name``); the machine-readable id stays on the finding's
        # ``category`` and in the coverage table's row order.
        failed_criteria=[title_by_id.get(i, i) for i in failed],
        degraded=bool(degraded_reason),
        degraded_reason=degraded_reason,
        document_map=[
            SectionCard(
                section_id=s.id,
                title=s.title,
                one_line_gist=gist_by_section.get(s.id, ""),
            )
            for s in model.sections
        ],
        metrics=ReviewMetrics(
            llm_calls=llm_calls,
            # The shared metrics type's generic "how many findings" slot; it is
            # named for review mode's investigate stage, which checklist mode
            # does not have. Nothing was investigated — these are probe hits.
            investigated_findings_count=len(findings),
            sections_processed=len(model.sections),
            edits_count=0,
            # Checklist mode has no gap loop and no reflection rounds by
            # design — predictable cost is the point.
            reflection_rounds_used=0,
        ),
    )
