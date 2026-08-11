"""Tests for deterministic co-location cross-linking (pure, no model).

The design invariant this module exists for is that NOTHING is merged, ranked
or dropped: a passage that more than one check landed on is worth reading
first, and consolidating the two findings would hide that the passage was
reached twice. Every test here is either about who gets cross-linked or about
that invariant holding.

Note the deliberately narrow claim under test: cross-linking says the checks
covered the same passage, NOT that they agree — see
:mod:`~andamentum.whetstone.v3.checklist.corroborate`.
"""

from __future__ import annotations

from andamentum.whetstone.v3.checklist.corroborate import corroborate
from andamentum.whetstone.v3.checklist.probes import ProbeFinding
from andamentum.whetstone.v3.model import Span


def _presence(
    probe_id: str,
    *,
    start: int,
    end: int,
    section_id: str = "s1",
    issue: str = "an issue",
) -> ProbeFinding:
    return ProbeFinding(
        probe_id=probe_id,
        issue=issue,
        quote=f"quote-{probe_id}-{start}",
        span=Span(section_id=section_id, start=start, end=end),
    )


def _absence(probe_id: str, missing_item: str) -> ProbeFinding:
    return ProbeFinding(
        probe_id=probe_id,
        issue="an issue",
        missing_item=missing_item,
        search_terms=("participants",),
    )


def _links(findings: list[ProbeFinding]) -> dict[str, tuple[str, ...]]:
    return {f.probe_id: f.corroborated_by for f in findings}


# ── presence: span overlap ──────────────────────────────────────────────────


def test_substantially_overlapping_spans_from_different_probes_link_both_ways() -> None:
    # 20 of the shorter span's 25 chars are shared — the same sentence, quoted
    # with slightly different edges by two checks.
    findings = [
        _presence("claim_outruns_evidence", start=10, end=40),
        _presence("reach_beyond_scope", start=20, end=45),
    ]
    out = corroborate(findings)

    assert _links(out) == {
        "claim_outruns_evidence": ("reach_beyond_scope",),
        "reach_beyond_scope": ("claim_outruns_evidence",),
    }


def test_marginal_overlap_does_not_link() -> None:
    """Barely-touching spans are adjacent sentences, not one passage.

    Bare interval overlap linked these, which on a real five-paragraph draft
    cross-linked 11 of 16 findings and made the signal meaningless. The rule is
    now a share of the SHORTER span, so a long quote that merely runs into a
    neighbouring one no longer counts.
    """
    findings = [
        _presence("claim_outruns_evidence", start=10, end=40),
        _presence("reach_beyond_scope", start=30, end=60),  # 10 of 30 shared
    ]
    out = corroborate(findings)

    assert _links(out) == {
        "claim_outruns_evidence": (),
        "reach_beyond_scope": (),
    }


def test_short_quote_inside_a_long_one_links() -> None:
    """Containment is the clearest same-passage case: all of the shorter span."""
    findings = [
        _presence("claim_outruns_evidence", start=10, end=60),
        _presence("term_drift", start=20, end=35),
    ]
    out = corroborate(findings)

    assert _links(out) == {
        "claim_outruns_evidence": ("term_drift",),
        "term_drift": ("claim_outruns_evidence",),
    }


def test_the_input_findings_are_not_mutated() -> None:
    findings = [
        _presence("claim_outruns_evidence", start=10, end=40),
        _presence("reach_beyond_scope", start=30, end=60),
    ]
    corroborate(findings)

    assert all(f.corroborated_by == () for f in findings)


def test_two_findings_from_the_same_probe_never_link() -> None:
    """A probe's quota already ranks its own findings — self-agreement is noise."""
    findings = [
        _presence("term_drift", start=10, end=40),
        _presence("term_drift", start=30, end=60),
    ]
    out = corroborate(findings)

    assert _links(out) == {"term_drift": ()}


def test_abutting_spans_do_not_count_as_overlap() -> None:
    """Half-open intervals: adjacent sentences are adjacency, not agreement."""
    findings = [
        _presence("claim_outruns_evidence", start=10, end=40),
        _presence("reach_beyond_scope", start=40, end=70),
    ]
    out = corroborate(findings)

    assert _links(out) == {"claim_outruns_evidence": (), "reach_beyond_scope": ()}


def test_identical_offsets_in_different_sections_do_not_link() -> None:
    findings = [
        _presence("claim_outruns_evidence", start=10, end=40, section_id="s1"),
        _presence("reach_beyond_scope", start=10, end=40, section_id="s2"),
    ]
    out = corroborate(findings)

    assert _links(out) == {"claim_outruns_evidence": (), "reach_beyond_scope": ()}


# ── absence: never corroborates ─────────────────────────────────────────────


def test_absence_findings_never_link_even_on_an_identical_missing_item() -> None:
    """An absence has no shared anchor — only the words its own probe chose.

    Two independently-prompted probes emitting the same ``missing_item`` string
    is not evidence they agree on a location, and no such match fires here.
    """
    findings = [
        _absence("missing_to_act", "sample size"),
        _absence("unanswered_aim", "sample size"),
    ]
    out = corroborate(findings)

    assert _links(out) == {"missing_to_act": (), "unanswered_aim": ()}


def test_an_absence_finding_never_links_to_a_presence_finding() -> None:
    findings = [
        _presence("claim_outruns_evidence", start=10, end=40),
        _absence("missing_to_act", "sample size"),
    ]
    out = corroborate(findings)

    assert _links(out) == {"claim_outruns_evidence": (), "missing_to_act": ()}


# ── determinism and the no-loss invariant ───────────────────────────────────


def _mixed_findings() -> list[ProbeFinding]:
    """One passage reached by three checks, one neighbouring span, two absences.

    ``term_drift``, ``alternative_explanation`` and ``claim_outruns_evidence``
    all cover the same passage. ``reach_beyond_scope`` only clips its edge (10
    of 30 chars) and must stay unlinked, so the fixture exercises both sides of
    the threshold rather than only the linking side.
    """
    return [
        _presence("claim_outruns_evidence", start=10, end=40),
        _presence("reach_beyond_scope", start=30, end=60),
        _presence("term_drift", start=20, end=35),
        _presence("alternative_explanation", start=22, end=36),
        _absence("missing_to_act", "sample size"),
        _absence("unanswered_aim", "SAMPLE  size"),
    ]


def test_repeated_runs_produce_identical_output() -> None:
    findings = _mixed_findings()
    assert corroborate(findings) == corroborate(findings)


def test_cross_links_do_not_depend_on_input_order() -> None:
    findings = _mixed_findings()
    reordered = [findings[i] for i in (3, 1, 5, 0, 2, 4)]

    assert _links(corroborate(findings)) == _links(corroborate(reordered))


def test_cross_links_are_sorted_and_deduplicated() -> None:
    findings = _mixed_findings()
    out = corroborate(findings)

    linked = next(f for f in out if f.probe_id == "term_drift")
    assert linked.corroborated_by == (
        "alternative_explanation",
        "claim_outruns_evidence",
    )


def test_edge_clipping_neighbour_stays_unlinked_among_linked_findings() -> None:
    """The threshold holds even when other findings around it do link."""
    out = corroborate(_mixed_findings())

    neighbour = next(f for f in out if f.probe_id == "reach_beyond_scope")
    assert neighbour.corroborated_by == ()


def test_nothing_is_ever_dropped_or_merged() -> None:
    """The invariant that separates corroborate from consolidate."""
    findings = _mixed_findings()
    out = corroborate(findings)

    assert len(out) == len(findings)
    assert [f.probe_id for f in out] == [f.probe_id for f in findings]
    assert [f.issue for f in out] == [f.issue for f in findings]
