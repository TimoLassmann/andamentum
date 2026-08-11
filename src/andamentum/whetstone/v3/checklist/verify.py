"""The two verification routes — deterministic, no model, no I/O but logging.

Review mode has one gate: a finding carries a verbatim quote, and a quote that
is not in the source is a hallucination. Checklist mode needs two, because a
probe can report a defect in two directions (see :mod:`.probes`), and only one
of them has an anchor to check.

The asymmetry is the point:

``presence``
    The defect is *in* the text. Verified by FINDING the quote — the same
    hallucination gate :func:`..review.verify_findings` applies, and a miss is
    a drop. The survivor is a copy carrying the located
    :attr:`ProbeFinding.span`.

``absence``
    The defect is that something is *missing*. There is nothing to anchor, so
    the finding instead ships the search terms that would appear if the item
    were present. Verified by FAILING to find any of them — a hit means the
    thing is right there and the finding is refuted. ``span`` stays ``None``:
    an absence has no location in the document, and inventing one to satisfy
    the renderer would be a lie about where to look.

Both shapes are checkable against the source. They are checkable in opposite
directions, which is what makes an absence claim trustworthy at all rather
than an unfalsifiable assertion the model gets to make for free.

**Why ANY term refutes, not ALL of them.** Requiring every term to be absent
before dropping would keep findings whose subject is demonstrably present and
merely phrased differently from one of the probe's guesses. That trade is
deliberately lopsided: checklist output lands in the author's Word file as a
tracked comment, and "you never state your sample size" next to a paragraph
stating the sample size is the single most damaging thing this mode can emit —
it costs the author time, and it costs the whole run its credibility. A
dropped true finding is invisible; a surviving false one is not. So the rule
is the conservative one: one hit and the finding goes.

The corollary is that a probe's ``search_terms`` should be the terms that
would appear if the item were present, NOT synonyms for the defect. Terms that
over-match simply make the probe quiet, which is the safe failure.
"""

from __future__ import annotations

import logging
from collections import Counter

from ..locate import is_present, locate
from ..model import DocumentModel, Span
from .probes import ProbeFinding

logger = logging.getLogger("andamentum.whetstone.v3")


def _refuting_term(finding: ProbeFinding, *, source: str) -> str | None:
    """The first search term present in *source*, or ``None`` if none are.

    Blank terms are skipped rather than searched: an empty needle matches
    everywhere, so leaving one in would silently refute every absence finding
    the probe produced. :class:`ProbeFinding` only guarantees the tuple is
    non-empty, not that its members are, so the guard belongs here.
    """
    for term in finding.search_terms:
        needle = term.strip()
        if not needle:
            logger.warning(
                "[v3.verify_probe_findings] %s: blank search term ignored "
                "(missing_item=%r) — an empty term matches everything and "
                "would refute the finding spuriously",
                finding.probe_id,
                finding.missing_item[:60],
            )
            continue
        if is_present(needle, source):
            return needle
    return None


def _has_searchable_terms(finding: ProbeFinding) -> bool:
    """True when at least one search term is non-blank."""
    return any(term.strip() for term in finding.search_terms)


def verify_probe_findings(
    findings: list[ProbeFinding], model: DocumentModel
) -> tuple[list[ProbeFinding], dict[str, int]]:
    """Verify every finding by the route its shape demands; drop the failures.

    Returns ``(kept, dropped_per_probe)``: the survivors, plus how many
    findings each probe lost here, keyed by probe id (a probe that lost none is
    absent from the mapping). The losses travel onward with the survivors
    because the coverage table cannot tell the truth without them — a probe
    whose findings were ALL discarded here would otherwise render as ``0``, the
    cell that means "this defect was looked for and is not there", when what
    actually happened is that the probe reported instances nobody could check.
    The shape mirrors the ``(findings, failed_probe_ids)`` pair
    :func:`~.run_probes.run_probes` returns one stage earlier, and for the same
    reason: what a stage lost is as much its result as what it kept.

    Deterministic and pure (logging aside): nothing the caller passed in is
    mutated — a surviving presence finding is a ``model_copy`` carrying its
    located span — so the mode's headline property, that the same findings
    produce the same table, survives this stage regardless of what the probe
    agents did upstream.

    Presence findings are matched against the WHOLE source rather than the
    section the probe was looking at, because a probe reports what it read via
    ``read_section`` / ``search_paper`` and does not reliably track which
    section id that came from; the section is recovered from the located
    offset instead, exactly as review mode does.

    Absence findings are dropped on the first search term that turns up (see
    the module docstring for why one hit is enough) and keep ``span=None`` on
    the way through.
    """
    kept: list[ProbeFinding] = []
    dropped_per_probe: Counter[str] = Counter()
    presence_kept = 0
    presence_dropped = 0
    absence_kept = 0
    absence_dropped = 0

    for finding in findings:
        if finding.shape == "absence":
            if not _has_searchable_terms(finding):
                # Every term was blank. The claim is now unfalsifiable, and an
                # unverifiable absence is indistinguishable from a guess — the
                # exact thing the search-term requirement exists to prevent.
                logger.warning(
                    "[v3.verify_probe_findings] %s: dropped absence finding with "
                    "no usable search terms (missing_item=%r) — nothing to check "
                    "it against",
                    finding.probe_id,
                    finding.missing_item[:60],
                )
                absence_dropped += 1
                dropped_per_probe[finding.probe_id] += 1
                continue
            refuted_by = _refuting_term(finding, source=model.source)
            if refuted_by is not None:
                logger.info(
                    "[v3.verify_probe_findings] %s: dropped (refuted, %r is in "
                    "the source): %r",
                    finding.probe_id,
                    refuted_by,
                    finding.missing_item[:60],
                )
                absence_dropped += 1
                dropped_per_probe[finding.probe_id] += 1
                continue
            absence_kept += 1
            kept.append(finding)
            continue

        location = locate(finding.quote, model.source)
        if location is None:
            logger.info(
                "[v3.verify_probe_findings] %s: dropped (not in source): %r",
                finding.probe_id,
                finding.quote[:60],
            )
            presence_dropped += 1
            dropped_per_probe[finding.probe_id] += 1
            continue
        section = next(
            (s for s in model.sections if s.start <= location[0] < s.end), None
        )
        span = Span(
            section_id=section.id if section else "?",
            start=location[0],
            end=location[1],
        )
        presence_kept += 1
        kept.append(finding.model_copy(update={"span": span}))

    logger.info(
        "[v3.verify_probe_findings] %d/%d kept — presence %d kept / %d dropped, "
        "absence %d kept / %d dropped",
        len(kept),
        len(findings),
        presence_kept,
        presence_dropped,
        absence_kept,
        absence_dropped,
    )
    return kept, dict(dropped_per_probe)
