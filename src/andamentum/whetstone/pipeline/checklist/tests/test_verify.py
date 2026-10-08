"""Tests for the two verification routes (pure, deterministic, no model).

Presence findings are verified by FINDING the quote; absence findings by
FAILING to find any search term. The asymmetry — and the conservative rule
that a single hit refutes an absence finding — is what makes an absence claim
trustworthy rather than a free assertion, so it is asserted explicitly below.
"""

from __future__ import annotations

from andamentum.whetstone.pipeline.checklist.probes import ProbeFinding
from andamentum.whetstone.pipeline.checklist.verify import verify_probe_findings
from andamentum.whetstone.pipeline.model import DocumentModel, Section

_INTRO = "# Intro\n\nWe used a threshold of 0.5 throughout.\n\n"
_METHODS = "# Methods\n\nThe cohort had 40 participants.\n"
_SOURCE = _INTRO + _METHODS


def _model() -> DocumentModel:
    return DocumentModel(
        source=_SOURCE,
        sections=[
            Section(id="s1", title="Intro", text=_INTRO, start=0, end=len(_INTRO)),
            Section(
                id="s2",
                title="Methods",
                text=_METHODS,
                start=len(_INTRO),
                end=len(_SOURCE),
            ),
        ],
    )


def _presence(quote: str, *, probe_id: str = "term_drift") -> ProbeFinding:
    return ProbeFinding(probe_id=probe_id, issue="an issue", quote=quote)


def _absence(*terms: str, probe_id: str = "missing_to_act") -> ProbeFinding:
    return ProbeFinding(
        probe_id=probe_id,
        issue="an issue",
        missing_item="statistical analysis plan",
        search_terms=terms,
    )


# ── presence ────────────────────────────────────────────────────────────────


def test_locatable_quote_is_kept_with_its_span_filled_in() -> None:
    quote = "The cohort had 40 participants."
    kept, _ = verify_probe_findings([_presence(quote)], _model())

    assert len(kept) == 1
    span = kept[0].span
    assert span is not None
    assert _SOURCE[span.start : span.end] == quote


def test_span_resolves_the_section_the_quote_actually_sits_in() -> None:
    """The section comes from the located offset, not from the first section."""
    kept, _ = verify_probe_findings(
        [_presence("The cohort had 40 participants.")], _model()
    )
    assert kept[0].span is not None
    assert kept[0].span.section_id == "s2"


def test_unlocatable_quote_is_dropped() -> None:
    kept, _ = verify_probe_findings(
        [_presence("The cohort had 400 participants.")], _model()
    )
    assert kept == []


# ── absence ─────────────────────────────────────────────────────────────────


def test_absence_finding_survives_when_no_search_term_is_present() -> None:
    kept, _ = verify_probe_findings([_absence("ANOVA", "regression model")], _model())

    assert len(kept) == 1
    # An absence has no location in the document; inventing one would be a lie
    # about where the author should look.
    assert kept[0].span is None


def test_absence_finding_is_dropped_when_one_of_several_terms_is_present() -> None:
    """ANY term hitting refutes the finding — this is the conservative rule.

    Requiring every term to be absent would keep findings whose subject is
    demonstrably in the document and merely worded differently from one of the
    probe's guesses. "You never state your sample size", stamped next to the
    paragraph that states it, is the single most damaging thing this mode can
    emit: a dropped true finding is invisible, a surviving false one is not.
    So one hit out of several is enough to drop.
    """
    kept, _ = verify_probe_findings(
        [_absence("ANOVA", "40 participants", "regression model")], _model()
    )
    assert kept == []


def test_absence_finding_with_only_blank_terms_is_dropped() -> None:
    """Nothing to check it against — an unverifiable absence is a guess."""
    kept, _ = verify_probe_findings([_absence("", "   ")], _model())
    assert kept == []


# ── both shapes together ────────────────────────────────────────────────────


def test_mixed_findings_are_verified_independently_and_in_order() -> None:
    findings = [
        _presence("The cohort had 40 participants.", probe_id="keep_presence"),
        _presence("The cohort had 400 participants.", probe_id="drop_presence"),
        _absence("ANOVA", "regression model", probe_id="keep_absence"),
        _absence("ANOVA", "40 participants", probe_id="drop_absence"),
    ]
    kept, _ = verify_probe_findings(findings, _model())

    assert [f.probe_id for f in kept] == ["keep_presence", "keep_absence"]


# ── what was lost travels with what survived ────────────────────────────────


def test_every_drop_route_is_counted_against_its_probe() -> None:
    """The coverage table cannot distinguish "found nothing" from "kept
    nothing" unless this stage reports what it discarded, so all three drop
    routes — unlocatable quote, refuted absence, unusable search terms — are
    counted."""
    findings = [
        _presence("The cohort had 40 participants.", probe_id="keep_presence"),
        _presence("The cohort had 400 participants.", probe_id="drop_presence"),
        _presence("Nothing like this is in the source.", probe_id="drop_presence"),
        _absence("ANOVA", "40 participants", probe_id="drop_refuted"),
        _absence("", "  ", probe_id="drop_unsearchable"),
    ]
    kept, dropped = verify_probe_findings(findings, _model())

    assert [f.probe_id for f in kept] == ["keep_presence"]
    assert dropped == {
        "drop_presence": 2,
        "drop_refuted": 1,
        "drop_unsearchable": 1,
    }
    # A probe that lost nothing is absent rather than present with a 0, so a
    # caller cannot mistake "kept everything" for "reported nothing".
    assert "keep_presence" not in dropped


def test_verification_does_not_mutate_the_findings_it_was_given() -> None:
    """The stage is pure: the survivor is a copy carrying the located span.

    Mutating the caller's objects would make the input list depend on whether
    verification has run yet, which is exactly the kind of hidden state a
    diffable mode cannot afford.
    """
    original = _presence("The cohort had 40 participants.")
    kept, _ = verify_probe_findings([original], _model())

    assert original.span is None
    assert kept[0] is not original
    assert kept[0].span is not None
