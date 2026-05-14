"""Tests for the audit report renderer.

The audit report is the single HTML renderer for the epistemic
system. These tests pin the v2 rendering contract using in-memory
``ReportData`` (no DB, no LLM, no HTTP).

The v2 spec is documented in
``docs/superpowers/plans/2026-05-14-audit-report-prd.md`` and
``docs/superpowers/plans/2026-05-14-audit-report-v2-implementation.md``.
Key invariants the tests pin:

- One closed verdict vocabulary at the renderer boundary.
- Single source of truth for evidence counts (per-claim bucketer wins).
- Adversarial-prefix items never appear in supporting evidence.
- Summary section is narrative only (no agent-prefix preamble, no
  self-quoting blockquote).
- Caveats section is system-level only (gate-trace anomalies + scope
  gaps), NOT a re-dump of per-evidence judgements.
- Gate trace surfaces routing + thresholds + observed values in plain
  text (``satisfied`` / ``failed`` / ``skipped``).
- Reproducibility footer carries snapshot id, model, pipeline version,
  and the literal CLI re-run command.
- Posterior reads directionally — ``Probability the claim is true:
  0.115 · verdict Refuted`` — with an inline decisive-bands legend.
- Rendered HTML carries no red/green inline styles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from andamentum.epistemic.audit_report import (
    _extract_strength_flags,
    _is_adversarial_judgement,
    _normalised_verdict,
    _source_url,
    _strip_summary_preamble,
    build_audit_report,
)
from andamentum.epistemic.report_data import (
    AdversarialSummary,
    ClaimSummary,
    ConfidenceScores,
    EvidenceSummary,
    GateTraceEntry,
    IBECandidate,
    InvestigationRound,
    InvestigationStats,
    ReportData,
    UncertaintySummary,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _atoms_with_heading(
    atoms: list[dict[str, Any]], heading: str
) -> list[dict[str, Any]]:
    return [a for a in atoms if a.get("heading") == heading]


def _atoms_of_kind(
    atoms: list[dict[str, Any]], kind: str
) -> list[dict[str, Any]]:
    return [a for a in atoms if a.get("kind") == kind]


def _atom_content_contains(
    atoms: list[dict[str, Any]], needle: str
) -> bool:
    """True if any prose/card/items atom's content (or details) contains
    the given substring."""
    for a in atoms:
        for field in ("content", "details", "body"):
            text = str(a.get(field, ""))
            if needle in text:
                return True
        # items atoms have entries with label+body.
        for entry in a.get("entries") or []:
            if needle in str(entry.get("body", "")):
                return True
            if needle in str(entry.get("label", "")):
                return True
    return False


def _make_report_data(
    *,
    claims: list[ClaimSummary] | None = None,
    evidence: list[EvidenceSummary] | None = None,
    adversarial: list[AdversarialSummary] | None = None,
    uncertainties: list[UncertaintySummary] | None = None,
    stats: InvestigationStats | None = None,
    terminal_state: str = "completed",
    posterior: float | None = 0.85,
    snapshot_id: str | None = "abc12345-snap-test-9999-000000000000",
    artefact_id: str | None = "def67890-art-test-aaaa-000000000000",
    pipeline_version: str = "0.3.0-rc1",
    pipeline_git_ref: str | None = "deadbee",
    reproduction_command: str = (
        'andamentum-epistemic verify "Aspirin reduces colorectal cancer risk" '
        "--model openai:gpt-5.4-nano --database test"
    ),
    direct_answer: str = "The evidence supports the claim with caveats.",
) -> ReportData:
    evidence = evidence or []
    # If the caller didn't supply explicit stats, derive them from the
    # evidence list so the renderer's count invariant
    # (stats >= data.evidence counts) holds trivially. Tests that want
    # to exercise the invariant pass an explicit ``stats`` arg.
    if stats is None:
        stats = InvestigationStats(
            total_evidence=len(evidence),
            evidence_supports=sum(
                1 for ev in evidence if ev.support_judgment == "supports"
            ),
            evidence_contradicts=sum(
                1 for ev in evidence if ev.support_judgment == "contradicts"
            ),
            evidence_no_bearing=sum(
                1 for ev in evidence if ev.support_judgment == "no_bearing"
            ),
        )
    return ReportData(
        research_question="Aspirin reduces colorectal cancer risk",
        clarified_question="Aspirin reduces colorectal cancer risk in adults",
        investigation_date=datetime(2026, 5, 13),
        model_used="openai:gpt-5.4-nano",
        database_name="test",
        direct_answer=direct_answer,
        question_type="verificatory",
        verdict="",  # let renderer fall back to _normalised_verdict
        claims=claims or [],
        evidence=evidence,
        uncertainties=uncertainties or [],
        adversarial=adversarial or [],
        convergence=[],
        open_questions=[],
        stats=stats,
        confidence_scores=ConfidenceScores(
            posterior=posterior,
            posterior_question_type="verificatory",
            terminal_state=terminal_state,
        ),
        snapshot_id=snapshot_id,
        artefact_id=artefact_id,
        pipeline_version=pipeline_version,
        pipeline_git_ref=pipeline_git_ref,
        reproduction_command=reproduction_command,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Source-URL helper (unchanged from v1)
# ──────────────────────────────────────────────────────────────────────────────


class TestSourceUrl:
    def test_doi_with_prefix(self):
        assert _source_url("doi:10.1234/abc.5678") == "https://doi.org/10.1234/abc.5678"

    def test_doi_without_prefix(self):
        assert _source_url("10.1234/abc") == "https://doi.org/10.1234/abc"

    def test_pmid_with_prefix(self):
        assert (
            _source_url("PMID:12345678")
            == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
        )

    def test_pmid_without_prefix(self):
        assert (
            _source_url("12345678") == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
        )

    def test_nct(self):
        assert (
            _source_url("NCT04501978")
            == "https://clinicaltrials.gov/study/NCT04501978"
        )

    def test_https_url_unchanged(self):
        url = "https://example.com/paper"
        assert _source_url(url) == url

    def test_http_url_unchanged(self):
        url = "http://example.com/paper"
        assert _source_url(url) == url

    def test_unknown_unchanged(self):
        assert _source_url("some-random-id") == "some-random-id"

    def test_empty(self):
        assert _source_url("") == ""


# ──────────────────────────────────────────────────────────────────────────────
# Closed verdict vocabulary — single source of truth
# ──────────────────────────────────────────────────────────────────────────────


class TestNormalisedVerdict:
    """The verdict label rendered on the claim-card badge and in the
    Q&A panel comes from one closed-vocabulary mapping. The labels are
    chosen so their lowercased ``data-value`` never matches the
    existing green/red CSS rules in typeset/atoms.py — i.e. badges
    render neutral. The *word* signals the verdict, not pigment."""

    def test_high_posterior_confirms(self):
        data = _make_report_data(posterior=0.92)
        assert _normalised_verdict(data) == "Confirmed"

    def test_high_posterior_with_refined_assessment(self):
        claim = ClaimSummary(
            claim_id="c1",
            statement="x",
            scope="g",
            assumptions=[],
            stage="supported",
            integrated_assessment="supports_refined",
        )
        data = _make_report_data(claims=[claim], posterior=0.92)
        assert _normalised_verdict(data) == "Confirmed with refinement"

    def test_low_posterior_refutes(self):
        data = _make_report_data(posterior=0.12)
        assert _normalised_verdict(data) == "Refuted"

    def test_middling_posterior_is_inconclusive(self):
        data = _make_report_data(posterior=0.50)
        assert _normalised_verdict(data) == "Inconclusive"

    def test_terminal_not_completed_is_insufficient(self):
        data = _make_report_data(posterior=0.92, terminal_state="oscillation_detected")
        assert _normalised_verdict(data) == "Insufficient evidence"

    def test_no_posterior_is_insufficient(self):
        data = _make_report_data(posterior=None)
        assert _normalised_verdict(data) == "Insufficient evidence"

    def test_vocabulary_avoids_css_color_collisions(self):
        """The five verdict labels must not lowercase to any of the
        words the existing typeset CSS tints green or red. If any
        future change introduces a new verdict label, this test pins
        the constraint."""
        labels = {
            "Confirmed",
            "Confirmed with refinement",
            "Refuted",
            "Inconclusive",
            "Insufficient evidence",
        }
        forbidden = {
            "supports",
            "supported",
            "pass",
            "approved",
            "contradicts",
            "contradicted",
            "challenged",
            "fail",
            "rejected",
        }
        for lbl in labels:
            assert lbl.lower() not in forbidden, (
                f"Verdict label {lbl!r} lowercases to a CSS color trigger"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Q&A panel — verdict-first, directional posterior with legend
# ──────────────────────────────────────────────────────────────────────────────


class TestQAPanel:
    def test_qa_panel_present_with_required_rows(self):
        data = _make_report_data(posterior=0.92)
        atoms = build_audit_report(data)
        items_atoms = _atoms_of_kind(atoms, "items")
        assert items_atoms, "Q&A items panel should be present"
        labels = [e["label"] for e in items_atoms[0]["entries"]]
        # Verdict leads.
        assert labels[0] == "What did we find?"
        # Required other rows present.
        for required in (
            "What was studied?",
            "What type of question?",
            "How confident are we?",
            "How thorough was the investigation?",
        ):
            assert required in labels

    def test_verdict_row_carries_closed_vocabulary_label(self):
        data = _make_report_data(posterior=0.12, direct_answer="")
        atoms = build_audit_report(data)
        items_atoms = _atoms_of_kind(atoms, "items")
        find = next(
            e for e in items_atoms[0]["entries"]
            if e["label"] == "What did we find?"
        )
        assert "Refuted" in find["body"]

    def test_confidence_body_uses_directional_phrasing_with_legend(self):
        """The posterior is read directionally so a non-Bayesian reader
        doesn't misread 11.5% as 'low confidence'. The legend defines
        BOTH bands inline — the verdict band (which thresholds determine
        the label) and the decisive band (which thresholds determine
        whether the pipeline stops iterating). The two are different
        in the system and the report names both so the gate-trace row
        for ``posterior_decisive`` stays interpretable."""
        data = _make_report_data(posterior=0.115)
        atoms = build_audit_report(data)
        items_atoms = _atoms_of_kind(atoms, "items")
        conf = next(
            e for e in items_atoms[0]["entries"]
            if e["label"] == "How confident are we?"
        )
        body = conf["body"]
        assert "Probability the claim is true" in body
        assert "0.115" in body
        # Closed-vocabulary verdict appears.
        assert "Refuted" in body
        # Both bands named in the legend.
        assert "Verdict band" in body
        assert "Decisive band" in body
        # The directional and decisive thresholds both appear.
        assert "0.66" in body  # POSTERIOR_DIRECTIONAL_BREAKPOINT
        assert "0.85" in body  # POSTERIOR_DECISIVE_THRESHOLD
        # The v1 "Posterior: 11.5%" phrasing is gone.
        assert "Posterior:" not in body

    def test_directional_breakpoint_imported_not_hardcoded(self):
        """The verdict-band thresholds in the report come from
        ``thresholds.POSTERIOR_DIRECTIONAL_BREAKPOINT``, not a hardcoded
        constant. If the pipeline ever moves the breakpoint, the report
        moves with it. Pins the import so a future refactor that drops
        the import is caught."""
        from andamentum.epistemic.audit_report import (
            _DIRECTIONAL_HI,
            _DIRECTIONAL_LO,
        )
        from andamentum.epistemic.thresholds import (
            POSTERIOR_DIRECTIONAL_BREAKPOINT,
        )

        assert _DIRECTIONAL_HI == POSTERIOR_DIRECTIONAL_BREAKPOINT
        assert _DIRECTIONAL_LO == 1.0 - POSTERIOR_DIRECTIONAL_BREAKPOINT

    def test_confidence_surfaces_terminal_when_not_completed(self):
        data = _make_report_data(terminal_state="oscillation_detected")
        atoms = build_audit_report(data)
        items_atoms = _atoms_of_kind(atoms, "items")
        conf = next(
            e for e in items_atoms[0]["entries"]
            if e["label"] == "How confident are we?"
        )
        assert "IBE-certified verdict" in conf["body"]
        assert "Probability the claim is true" not in conf["body"]

    def test_no_verdict_callouts_at_all(self):
        """Audit layout must not emit success/warning-toned callouts —
        the user explicitly didn't want red/green tone cues."""
        data = _make_report_data(posterior=0.85)
        atoms = build_audit_report(data)
        for c in _atoms_of_kind(atoms, "callout"):
            tone = c.get("tone")
            assert tone in (None, "note", "info"), (
                f"Audit layout used non-neutral callout tone: {tone}"
            )

    def test_thoroughness_includes_round_count(self):
        stats = InvestigationStats(
            total_evidence=100, investigation_rounds_total=6
        )
        data = _make_report_data(stats=stats)
        atoms = build_audit_report(data)
        items_atoms = _atoms_of_kind(atoms, "items")
        thorough = next(
            e for e in items_atoms[0]["entries"]
            if e["label"] == "How thorough was the investigation?"
        )
        assert "100 evidence sources" in thorough["body"]
        assert "6 investigation rounds" in thorough["body"]

    def test_reproduction_row_present_when_command_set(self):
        data = _make_report_data()
        atoms = build_audit_report(data)
        items_atoms = _atoms_of_kind(atoms, "items")
        labels = [e["label"] for e in items_atoms[0]["entries"]]
        assert "Reproduction" in labels


# ──────────────────────────────────────────────────────────────────────────────
# Summary — narrative only, preamble stripped
# ──────────────────────────────────────────────────────────────────────────────


class TestSummaryPreambleStripping:
    def test_strips_research_question_prefix_lines(self):
        text = (
            "**Research Question:** *Aspirin reduces CRC*\n"
            "**Evidence Sources:** 10\n"
            "> Quoted self-summary line.\n"
            "\n"
            "Real answer text."
        )
        assert _strip_summary_preamble(text) == "Real answer text."

    def test_strips_blockquote_preamble(self):
        text = "> blockquote\n> more quote\n\nReal answer."
        assert _strip_summary_preamble(text) == "Real answer."

    def test_passthrough_when_no_preamble(self):
        text = "Just narrative answer text."
        assert _strip_summary_preamble(text) == text

    def test_summary_section_in_report_has_no_second_evidence_count(self):
        """End-to-end: the rendered Summary section must not carry the
        agent-prefixed metadata lines, so the rendered report does not
        show the same Evidence Sources count twice with different
        numbers (the v1 bug)."""
        data = _make_report_data(
            direct_answer=(
                "**Research Question:** *Aspirin reduces CRC*\n"
                "**Evidence Sources:** 10 | **Claims Established:** 1 of 1\n"
                "> Quoted preamble.\n\n"
                "The evidence supports the claim with caveats."
            ),
            stats=InvestigationStats(total_evidence=33),
        )
        atoms = build_audit_report(data)
        summary = _atoms_with_heading(atoms, "Summary")
        assert summary
        body = summary[0]["content"]
        assert "Evidence Sources" not in body
        assert "Research Question" not in body
        assert body.startswith("The evidence supports")


# ──────────────────────────────────────────────────────────────────────────────
# Evidence at a glance — single source of truth for counts
# ──────────────────────────────────────────────────────────────────────────────


class TestEvidenceAtAGlance:
    def test_section_heading_is_evidence_at_a_glance(self):
        """v1 called this section 'Summary of findings'. v2 uses
        'Evidence at a glance' — the rename signals that the table is a
        scan-at-a-glance summary, not the system's findings (which live
        in the Q&A panel)."""
        data = _make_report_data()
        atoms = build_audit_report(data)
        assert _atoms_with_heading(atoms, "Evidence at a glance")
        assert not _atoms_with_heading(atoms, "Summary of findings")

    def test_counts_table_present_with_directions(self):
        stats = InvestigationStats(
            total_evidence=100,
            evidence_supports=24,
            evidence_contradicts=4,
            evidence_no_bearing=72,
        )
        data = _make_report_data(stats=stats)
        atoms = build_audit_report(data)
        section = _atoms_with_heading(atoms, "Evidence at a glance")[0]
        text = section["content"]
        assert "100 evidence items" in text
        assert "Supporting" in text
        assert "Contradicting" in text
        assert "No bearing" in text

    def test_per_claim_verdict_row_present(self):
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim about something",
            scope="general",
            assumptions=[],
            stage="supported",
        )
        data = _make_report_data(claims=[claim], posterior=0.92)
        atoms = build_audit_report(data)
        section = _atoms_with_heading(atoms, "Evidence at a glance")[0]
        text = section["content"]
        assert "Test claim about something" in text
        assert "Confirmed" in text


class TestCountInvariant:
    """PRD R2: the renderer raises if data.stats reports FEWER items
    in a direction than data.evidence actually contains. The inverse
    (stats > data.evidence) is legitimate because stats is tallied
    from the pre-dedup raw entity list while data.evidence is the
    dedup'd view; only the inverted case suggests an upstream bug."""

    def test_invariant_raises_when_stats_below_evidence(self):
        ev = [
            EvidenceSummary(
                evidence_id=f"s{i}",
                source_type="pubmed",
                source_ref=f"PMID:{i}",
                extracted_content="x",
                judgment_reasoning="Direct support.",
                support_judgment="supports",
            )
            for i in range(5)
        ]
        # stats says 2 supports — but data.evidence has 5. Inversion.
        data = _make_report_data(
            evidence=ev,
            stats=InvestigationStats(
                total_evidence=5,
                evidence_supports=2,
                evidence_contradicts=0,
                evidence_no_bearing=0,
            ),
        )
        with pytest.raises(ValueError, match="Evidence count invariant"):
            build_audit_report(data)

    def test_invariant_allows_stats_above_evidence_dedup_case(self):
        """stats counts can legitimately exceed data.evidence counts
        because stats includes pre-dedup duplicates."""
        ev = [
            EvidenceSummary(
                evidence_id=f"s{i}",
                source_type="pubmed",
                source_ref=f"PMID:{i}",
                extracted_content="x",
                judgment_reasoning="Direct support.",
                support_judgment="supports",
            )
            for i in range(3)
        ]
        # stats reports 7 (raw with dupes); data.evidence has 3 (dedup'd).
        data = _make_report_data(
            evidence=ev,
            stats=InvestigationStats(
                total_evidence=7,
                evidence_supports=7,
                evidence_contradicts=0,
                evidence_no_bearing=0,
            ),
        )
        # Must not raise.
        atoms = build_audit_report(data)
        assert atoms

    def test_invariant_passes_when_stats_match_evidence(self):
        ev = [
            EvidenceSummary(
                evidence_id=f"s{i}",
                source_type="pubmed",
                source_ref=f"PMID:{i}",
                extracted_content="x",
                judgment_reasoning="Direct support.",
                support_judgment="supports",
            )
            for i in range(4)
        ]
        data = _make_report_data(evidence=ev)  # default stats matches
        atoms = build_audit_report(data)
        assert atoms


class TestSingleSourceOfTruthForCounts:
    """Counts must agree across the Q&A panel thoroughness row, the
    Evidence-at-a-glance table, and the per-claim card details. The
    per-claim bucketer is canonical; if upstream stats disagree, the
    bucketer wins. The v1 bug had 11 supporting in one place and 9 in
    another."""

    def test_counts_consistent_across_sections(self):
        ev_supp = [
            EvidenceSummary(
                evidence_id=f"s{i}",
                source_type="pubmed",
                source_ref=f"PMID:{1000 + i}",
                extracted_content="content",
                judgment_reasoning="Direct support of claim.",
                support_judgment="supports",
            )
            for i in range(7)
        ]
        ev_con = [
            EvidenceSummary(
                evidence_id=f"c{i}",
                source_type="pubmed",
                source_ref=f"PMID:{2000 + i}",
                extracted_content="content",
                judgment_reasoning="Null result.",
                support_judgment="contradicts",
            )
            for i in range(3)
        ]
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="g",
            assumptions=[],
            stage="supported",
            evidence_ids=[ev.evidence_id for ev in ev_supp + ev_con],
        )
        # Deliberately set stats counts WRONG to confirm the bucketer
        # wins.
        data = _make_report_data(
            claims=[claim],
            evidence=ev_supp + ev_con,
            stats=InvestigationStats(
                total_evidence=10,
                evidence_supports=99,  # wrong
                evidence_contradicts=99,  # wrong
                evidence_no_bearing=99,  # wrong
            ),
        )
        atoms = build_audit_report(data)
        section = _atoms_with_heading(atoms, "Evidence at a glance")[0]
        text = section["content"]
        assert " 7 " in text or "| 7 |" in text, (
            f"Expected supporting=7 in evidence-at-a-glance table, got {text!r}"
        )
        assert " 3 " in text or "| 3 |" in text, (
            "Expected contradicting=3 in evidence-at-a-glance table"
        )
        # And the wrong upstream counts must NOT appear.
        assert "99" not in text


# ──────────────────────────────────────────────────────────────────────────────
# Adversarial bucketing defence
# ──────────────────────────────────────────────────────────────────────────────


class TestAdversarialBucketingDefence:
    """Items whose judge prose begins with 'Adversarial (…)' are
    adversarial-search output, regardless of the upstream
    ``support_judgment`` label. The renderer's defence prevents them
    from leaking into the supporting list — the v1 mis-bucketing bug
    where the Strongest supporting evidence section contained items
    that were actually adversarial counter-evidence."""

    def test_helper_recognises_adversarial_prefix(self):
        assert _is_adversarial_judgement("Adversarial (statistical): blah blah")
        assert _is_adversarial_judgement(" Adversarial (generalization): foo")
        assert _is_adversarial_judgement("ADVERSARIAL (interpretation): bar")

    def test_helper_ignores_non_adversarial(self):
        assert not _is_adversarial_judgement("Direct support of the claim.")
        assert not _is_adversarial_judgement("")
        assert not _is_adversarial_judgement(None)

    def test_adversarial_labelled_item_routed_to_contradicting(self):
        """Even though support_judgment='supports', the item ends up in
        the contradicting bucket because the judge prose betrays its
        true origin."""
        ev = [
            EvidenceSummary(
                evidence_id="bad",
                source_type="web_search",
                source_ref="https://example.com/p",
                extracted_content="x",
                judgment_reasoning=(
                    "Adversarial (statistical): The counterargument "
                    "directly engages the claim..."
                ),
                support_judgment="supports",  # mislabelled upstream
            )
        ]
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="g",
            assumptions=[],
            stage="supported",
            evidence_ids=["bad"],
        )
        atoms = build_audit_report(_make_report_data(claims=[claim], evidence=ev))
        # The supports section either is absent or does not mention this URL.
        supports_atoms = [
            a
            for a in atoms
            if a.get("kind") == "prose"
            and "Supporting evidence" in str(a.get("content", ""))
        ]
        for atom in supports_atoms:
            assert "example.com/p" not in atom["content"], (
                "Adversarial-prefix item leaked into Supporting evidence"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Reasoning trace — claim card, gate trace, IBE cards
# ──────────────────────────────────────────────────────────────────────────────


class TestReasoningTrace:
    def test_section_heading_is_reasoning_trace(self):
        """v1 called this 'Detailed analysis'. v2 renames to 'Reasoning
        trace' — names the externalisation move directly."""
        data = _make_report_data(
            claims=[
                ClaimSummary(
                    claim_id="c1",
                    statement="Test",
                    scope="g",
                    assumptions=[],
                    stage="supported",
                )
            ]
        )
        atoms = build_audit_report(data)
        assert _atoms_with_heading(atoms, "Reasoning trace")
        assert not _atoms_with_heading(atoms, "Detailed analysis")
        assert not _atoms_with_heading(atoms, "Findings")

    def test_intro_points_reader_to_qa_panel_for_verdict(self):
        atoms = build_audit_report(
            _make_report_data(
                claims=[
                    ClaimSummary(
                        claim_id="c1",
                        statement="Single claim",
                        scope="g",
                        assumptions=[],
                        stage="supported",
                    )
                ]
            )
        )
        section = _atoms_with_heading(atoms, "Reasoning trace")[0]
        intro = section["content"]
        assert "Q&A panel above" in intro
        assert "claim under investigation" in intro

    def test_claim_card_statement_labelled_as_claim(self):
        claim = ClaimSummary(
            claim_id="c1",
            statement="Hydroxychloroquine reduces COVID-19 mortality",
            scope="g",
            assumptions=[],
            stage="supported",
        )
        atoms = build_audit_report(_make_report_data(claims=[claim]))
        card = next(
            c for c in _atoms_of_kind(atoms, "card")
            if "Hydroxychloroquine" in str(c.get("content", ""))
        )
        content = str(card["content"])
        assert content.startswith("**Claim:**")

    def test_claim_card_badge_uses_closed_vocabulary_not_stage(self):
        """The v1 bug: claim.stage='supported' rendered a green
        'supported' badge on a refuted claim. The v2 badge reads the
        normalised verdict instead — and the badge value must be one of
        the closed-vocabulary labels, not the raw stage."""
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="g",
            assumptions=[],
            stage="supported",  # raw stage label
        )
        # Low posterior → Refuted, regardless of stage.
        data = _make_report_data(claims=[claim], posterior=0.12)
        atoms = build_audit_report(data)
        card = next(c for c in _atoms_of_kind(atoms, "card") if "Test claim" in str(c.get("content", "")))
        assert card.get("badge") == "Refuted"
        # Never carries the raw stage word.
        assert card.get("badge") not in ("supported", "robust", "provisional")

    def test_reasoning_block_embeds_h3_h4_subheadings(self):
        """Per-claim sub-sections nest as h3/h4 INSIDE a prose body so
        the heading hierarchy is visibly nested under the parent h2
        'Reasoning trace'. Embedded h3/h4 uses the existing
        .typeset-prose CSS — no typeset module changes."""
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="g",
            assumptions=[],
            stage="supported",
            investigation_rounds=[
                InvestigationRound(round_index=1, intent="x", evidence_count=3),
            ],
            gate_trace=[
                GateTraceEntry(
                    name="scrutiny",
                    routing="PRIMARY",
                    required="pass",
                    observed="pass",
                    status="satisfied",
                ),
            ],
            ibe_candidates=[
                IBECandidate(
                    candidate_id="A",
                    verdict="supports",
                    description="x",
                    loveliness=0.8,
                    likeliness=0.7,
                    chosen=True,
                ),
            ],
        )
        atoms = build_audit_report(_make_report_data(claims=[claim]))
        reasoning_prose = [
            a
            for a in atoms
            if a.get("kind") == "prose"
            and str(a.get("id", "")).startswith("reasoning-")
        ]
        assert reasoning_prose
        body = reasoning_prose[0]["content"]
        # Top-level reasoning section header inside the body is h3.
        assert body.startswith("### How the system reasoned")
        # Sub-sections are h4.
        assert "#### Investigation rounds" in body
        assert "#### Gate trace" in body
        assert "#### Alternative explanations the system considered" in body


class TestGateTrace:
    def _claim_with_gates(
        self, entries: list[GateTraceEntry]
    ) -> ClaimSummary:
        return ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="g",
            assumptions=[],
            stage="supported",
            gate_trace=entries,
        )

    def test_gate_trace_table_has_all_columns(self):
        claim = self._claim_with_gates(
            [
                GateTraceEntry(
                    name="scrutiny",
                    routing="PRIMARY",
                    required="pass",
                    observed="pass",
                    status="satisfied",
                ),
                GateTraceEntry(
                    name="convergence",
                    routing="PRIMARY",
                    required="≥ 2 independent sources",
                    observed="14 independent sources",
                    status="satisfied",
                ),
            ]
        )
        atoms = build_audit_report(_make_report_data(claims=[claim]))
        reasoning_prose = next(
            a
            for a in atoms
            if a.get("kind") == "prose"
            and str(a.get("id", "")).startswith("reasoning-")
        )
        body = reasoning_prose["content"]
        assert "| Gate |" in body
        assert "| Routing |" in body
        assert "| Required |" in body
        assert "| Observed |" in body
        assert "| Status |" in body
        # Both gates rendered.
        assert "Scrutiny" in body
        assert "Convergence" in body
        # Plain-text status words — no icons.
        assert "satisfied" in body
        assert "✓" not in body
        assert "✗" not in body

    def test_skipped_gate_renders_with_routing_skip(self):
        claim = self._claim_with_gates(
            [
                GateTraceEntry(
                    name="deductive_validation",
                    routing="SKIP",
                    required="n/a — not routed for this question type",
                    observed="—",
                    status="skipped",
                ),
            ]
        )
        atoms = build_audit_report(_make_report_data(claims=[claim]))
        reasoning_prose = next(
            a
            for a in atoms
            if a.get("kind") == "prose"
            and str(a.get("id", "")).startswith("reasoning-")
        )
        body = reasoning_prose["content"]
        assert "Deductive validation" in body
        assert "skipped" in body
        assert "SKIP" in body

    def test_failed_gate_status_renders_as_word_failed(self):
        claim = self._claim_with_gates(
            [
                GateTraceEntry(
                    name="scrutiny",
                    routing="PRIMARY",
                    required="pass",
                    observed="fail",
                    status="failed",
                ),
            ]
        )
        atoms = build_audit_report(_make_report_data(claims=[claim]))
        reasoning_prose = next(
            a
            for a in atoms
            if a.get("kind") == "prose"
            and str(a.get("id", "")).startswith("reasoning-")
        )
        body = reasoning_prose["content"]
        assert "| failed |" in body or "failed" in body

    def test_no_gate_trace_omits_section(self):
        claim = self._claim_with_gates([])
        # Also no IBE / rounds — fully bare claim.
        atoms = build_audit_report(_make_report_data(claims=[claim]))
        reasoning_prose = [
            a
            for a in atoms
            if a.get("kind") == "prose"
            and str(a.get("id", "")).startswith("reasoning-")
        ]
        # The reasoning prose either is absent or doesn't mention "Gate trace".
        for a in reasoning_prose:
            assert "Gate trace" not in a["content"]


class TestIBECandidateCards:
    def test_each_candidate_is_its_own_card_atom(self):
        candidates = [
            IBECandidate(
                candidate_id="A",
                verdict="supports",
                description="Explanation A.",
                loveliness=0.8,
                likeliness=0.7,
                chosen=True,
            ),
            IBECandidate(
                candidate_id="B",
                verdict="contradicts",
                description="Explanation B.",
                loveliness=0.3,
                likeliness=0.5,
                runner_up=True,
            ),
            IBECandidate(
                candidate_id="C",
                verdict="insufficient",
                description="Explanation C.",
                loveliness=0.2,
                likeliness=0.4,
            ),
        ]
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="g",
            assumptions=[],
            stage="supported",
            ibe_candidates=candidates,
        )
        atoms = build_audit_report(_make_report_data(claims=[claim]))
        ibe_cards = [
            c for c in _atoms_of_kind(atoms, "card")
            if str(c.get("id", "")).startswith("ibe-")
        ]
        assert len(ibe_cards) == 3
        # Selected first.
        assert ibe_cards[0].get("badge") == "selected"
        assert ibe_cards[1].get("badge") == "runner-up"
        # Third one uses "not selected" — the v1 "rejected" word would
        # trigger the red CSS rule.
        assert ibe_cards[2].get("badge") == "not selected"

    def test_ibe_intro_anchors_evidence_pattern_with_counts(self):
        """IBE candidate descriptions often reference 'the evidence
        pattern' (because that's how LLM-generated rationales talk
        about the inputs). The sub-section intro must name the pattern
        concretely — N supporting / M contradicting / K no-bearing —
        immediately above the candidate cards so the referent is
        visible without scrolling."""
        supports = [
            EvidenceSummary(
                evidence_id=f"s{i}",
                source_type="pubmed",
                source_ref=f"PMID:{1000 + i}",
                extracted_content="x",
                judgment_reasoning="Direct support.",
                support_judgment="supports",
            )
            for i in range(4)
        ]
        contradicts = [
            EvidenceSummary(
                evidence_id=f"c{i}",
                source_type="pubmed",
                source_ref=f"PMID:{2000 + i}",
                extracted_content="x",
                judgment_reasoning="Null result.",
                support_judgment="contradicts",
            )
            for i in range(7)
        ]
        no_bearing = [
            EvidenceSummary(
                evidence_id=f"n{i}",
                source_type="pubmed",
                source_ref=f"PMID:{3000 + i}",
                extracted_content="x",
                judgment_reasoning="Out of scope.",
                support_judgment="no_bearing",
            )
            for i in range(12)
        ]
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="g",
            assumptions=[],
            stage="supported",
            evidence_ids=[
                ev.evidence_id for ev in supports + contradicts + no_bearing
            ],
            ibe_candidates=[
                IBECandidate(
                    candidate_id="A",
                    verdict="supports",
                    description=(
                        "The evidence pattern is heterogeneous enough "
                        "that a directional verdict is not warranted."
                    ),
                    loveliness=0.5,
                    likeliness=0.5,
                    chosen=True,
                ),
            ],
        )
        atoms = build_audit_report(
            _make_report_data(
                claims=[claim],
                evidence=supports + contradicts + no_bearing,
            )
        )
        reasoning_prose = next(
            a for a in atoms
            if a.get("kind") == "prose"
            and str(a.get("id", "")).startswith("reasoning-")
        )
        body = reasoning_prose["content"]
        # The anchor sentence is present.
        assert "The evidence pattern for this claim" in body
        # The exact counts appear, so the candidate's "evidence pattern"
        # noun-phrase has a concrete referent right above it.
        assert "4 supporting" in body
        assert "7 contradicting" in body
        assert "12 no-bearing" in body
        # The anchor appears BEFORE the loveliness/likeliness scoring
        # explanation, i.e. immediately under the IBE heading.
        ibe_heading_idx = body.index(
            "#### Alternative explanations the system considered"
        )
        anchor_idx = body.index("The evidence pattern for this claim")
        scoring_idx = body.index("loveliness")
        assert ibe_heading_idx < anchor_idx < scoring_idx

    def test_candidate_description_is_not_truncated(self):
        long = "Word " * 60  # 300+ chars
        candidate = IBECandidate(
            candidate_id="A",
            verdict="supports",
            description=long.strip(),
            loveliness=0.8,
            likeliness=0.7,
            chosen=True,
        )
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="g",
            assumptions=[],
            stage="supported",
            ibe_candidates=[candidate],
        )
        atoms = build_audit_report(_make_report_data(claims=[claim]))
        ibe_card = next(
            c for c in _atoms_of_kind(atoms, "card")
            if str(c.get("id", "")).startswith("ibe-")
        )
        assert "…" not in ibe_card["content"]


class TestAdaptiveAuditTrailIntro:
    def test_intro_acknowledges_follow_up_rounds_when_present(self):
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="g",
            assumptions=[],
            stage="supported",
            investigation_rounds=[
                InvestigationRound(round_index=1, intent="x", evidence_count=3),
            ],
            ibe_candidates=[
                IBECandidate(
                    candidate_id="A",
                    verdict="supports",
                    description="x",
                    loveliness=0.8,
                    likeliness=0.7,
                    chosen=True,
                ),
            ],
        )
        atoms = build_audit_report(_make_report_data(claims=[claim]))
        reasoning_prose = next(
            a for a in atoms
            if a.get("kind") == "prose"
            and str(a.get("id", "")).startswith("reasoning-")
        )
        body = reasoning_prose["content"]
        assert "initial evidence gather" in body
        assert "did not fully resolve" in body
        assert "inference-to-the-best-explanation" in body

    def test_intro_adapts_when_no_investigation_rounds(self):
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="g",
            assumptions=[],
            stage="supported",
            investigation_rounds=[],
            ibe_candidates=[
                IBECandidate(
                    candidate_id="A",
                    verdict="supports",
                    description="x",
                    loveliness=0.7,
                    likeliness=0.7,
                    chosen=True,
                ),
            ],
        )
        atoms = build_audit_report(_make_report_data(claims=[claim]))
        reasoning_prose = next(
            a for a in atoms
            if a.get("kind") == "prose"
            and str(a.get("id", "")).startswith("reasoning-")
        )
        body = reasoning_prose["content"]
        assert "initial gather was sufficient" in body
        assert "Round 1" not in body


# ──────────────────────────────────────────────────────────────────────────────
# Evidence-line — reference leads, provider as pill, strength flags
# ──────────────────────────────────────────────────────────────────────────────


class TestEvidenceLine:
    def test_line_leads_with_reference_then_provider_pill(self):
        ev = [
            EvidenceSummary(
                evidence_id="e1",
                source_type="europepmc",
                source_ref="doi:10.1234/abc",
                extracted_content="full text",
                judgment_reasoning="Direct support of claim outcome.",
                support_judgment="supports",
            )
        ]
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="g",
            assumptions=[],
            stage="supported",
            evidence_ids=["e1"],
        )
        atoms = build_audit_report(_make_report_data(claims=[claim], evidence=ev))
        supports_prose = next(
            a for a in atoms
            if a.get("kind") == "prose"
            and "Supporting evidence" in str(a.get("content", ""))
        )
        body = supports_prose["content"]
        line = next(ln for ln in body.splitlines() if "doi:10.1234/abc" in ln)
        ref_idx = line.find("doi:10.1234/abc")
        provider_idx = line.find("`Europe PMC`")
        text_idx = line.find("Direct support of claim")
        assert ref_idx >= 0 and provider_idx >= 0 and text_idx >= 0
        assert ref_idx < provider_idx < text_idx

    def test_provider_uses_human_readable_name(self):
        ev = [
            EvidenceSummary(
                evidence_id="e1",
                source_type="europepmc",
                source_ref="doi:10.1234/abc",
                extracted_content="full text",
                judgment_reasoning="Direct support.",
                support_judgment="supports",
            )
        ]
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="g",
            assumptions=[],
            stage="supported",
            evidence_ids=["e1"],
        )
        atoms = build_audit_report(_make_report_data(claims=[claim], evidence=ev))
        # Human-readable label, not the slug.
        assert _atom_content_contains(atoms, "`Europe PMC`")
        # Slug must not appear as the pill (only the human label).
        supports_prose = next(
            a for a in atoms
            if a.get("kind") == "prose"
            and "Supporting evidence" in str(a.get("content", ""))
        )
        # `europepmc` should not appear inside the support body — only Europe PMC.
        assert "`europepmc`" not in supports_prose["content"]

    def test_strength_flags_extracted_from_judgement_prose(self):
        assert "RCT" in _extract_strength_flags(
            "Randomized controlled trial in hospitalized patients."
        )
        assert "meta-analysis" in _extract_strength_flags(
            "This meta-analysis pools 12 trials."
        )
        assert "observational" in _extract_strength_flags(
            "Large retrospective cohort study."
        )
        assert "single-arm" in _extract_strength_flags(
            "Single-arm reporting no deaths."
        )
        assert "combination intervention" in _extract_strength_flags(
            "HCQ + azithromycin combination treatment."
        )

    def test_flags_render_as_italic_parenthetical(self):
        ev = [
            EvidenceSummary(
                evidence_id="e1",
                source_type="pubmed",
                source_ref="PMID:1234567",
                extracted_content="x",
                judgment_reasoning="Randomized controlled trial result.",
                support_judgment="supports",
            )
        ]
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="g",
            assumptions=[],
            stage="supported",
            evidence_ids=["e1"],
        )
        atoms = build_audit_report(_make_report_data(claims=[claim], evidence=ev))
        supports_prose = next(
            a for a in atoms
            if a.get("kind") == "prose"
            and "Supporting evidence" in str(a.get("content", ""))
        )
        # Italic parenthetical at the end of the line.
        assert "*(RCT" in supports_prose["content"] or "*(RCT," in supports_prose["content"]


# ──────────────────────────────────────────────────────────────────────────────
# Caveats — system-level only, NOT per-evidence dump
# ──────────────────────────────────────────────────────────────────────────────


class TestCaveatsAndLimitations:
    def test_section_named_caveats_and_limitations(self):
        unc = UncertaintySummary(
            uncertainty_id="u1",
            uncertainty_type="scope_gap",
            description="A scope-level limit.",
            scope="global",
            is_blocking=False,
            is_resolved=False,
        )
        data = _make_report_data(uncertainties=[unc])
        atoms = build_audit_report(data)
        # New merged heading.
        assert _atoms_with_heading(atoms, "Caveats and limitations")
        # Old separate sections gone.
        assert not _atoms_with_heading(atoms, "Caveats")
        assert not _atoms_with_heading(atoms, "Limitations")

    def test_failed_gate_surfaces_as_system_level_caveat(self):
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="g",
            assumptions=[],
            stage="supported",
            gate_trace=[
                GateTraceEntry(
                    name="scrutiny",
                    routing="PRIMARY",
                    required="pass",
                    observed="fail",
                    status="failed",
                ),
            ],
        )
        data = _make_report_data(claims=[claim])
        atoms = build_audit_report(data)
        caveats = _atoms_with_heading(atoms, "Caveats and limitations")
        assert caveats, "Caveats section should be present when a gate failed"
        body = caveats[0]["content"]
        assert "Scrutiny" in body
        # The phrase about the verdict being resolved by remaining gates.
        assert "remaining gates" in body or "remaining gate" in body

    def test_caveats_does_not_dump_per_evidence_judgements(self):
        """The v1 bug: 'Limitations' contained 19 bullets, each a
        re-phrasing of an individual contradicting evidence judgement.
        v2's Caveats section is system-level only and must contain NONE
        of those per-evidence prose strings."""
        contra = [
            EvidenceSummary(
                evidence_id=f"c{i}",
                source_type="pubmed",
                source_ref=f"PMID:{2000 + i}",
                extracted_content="x",
                judgment_reasoning=(
                    "The evidence reports no mortality reduction in this "
                    f"specific cohort (cohort #{i})."
                ),
                support_judgment="contradicts",
            )
            for i in range(5)
        ]
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="g",
            assumptions=[],
            stage="supported",
            evidence_ids=[ev.evidence_id for ev in contra],
        )
        data = _make_report_data(claims=[claim], evidence=contra)
        atoms = build_audit_report(data)
        caveats = _atoms_with_heading(atoms, "Caveats and limitations")
        # No caveats section at all when there are no system-level
        # caveats — the per-evidence prose lives in the Contradicting
        # evidence section, not here.
        for atom in caveats:
            for ev in contra:
                assert ev.judgment_reasoning not in atom["content"]


# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility footer + repro command
# ──────────────────────────────────────────────────────────────────────────────


class TestReproducibilityFooter:
    def test_footer_aside_with_groups_present(self):
        data = _make_report_data()
        atoms = build_audit_report(data)
        # Identify the sidebar aside — it has 'groups' set.
        sidebar_asides = [
            a for a in atoms
            if a.get("kind") == "aside" and a.get("groups")
        ]
        assert sidebar_asides, "Reproducibility sidebar aside is missing"
        groups = sidebar_asides[0]["groups"]
        # Three groups expected.
        assert "Pipeline" in groups
        assert "Model" in groups
        assert "Persistence" in groups
        # Persistence carries the snapshot id (truncated to 12 chars + ellipsis).
        persistence = groups["Persistence"]
        assert "snapshot" in persistence
        assert persistence["snapshot"].startswith("abc12345-sna")
        assert persistence["snapshot"].endswith("…")

    def test_reproduction_command_rendered_once_in_qa_panel(self):
        """The reproduction command appears in the Q&A panel's
        'Reproduction' row only — not duplicated as a separate prose
        block at the bottom. The sidebar carries the supporting metadata
        (snapshot, version)."""
        data = _make_report_data()
        atoms = build_audit_report(data)
        items_atoms = _atoms_of_kind(atoms, "items")
        repro_entry = next(
            (
                e for e in items_atoms[0]["entries"]
                if e["label"] == "Reproduction"
            ),
            None,
        )
        assert repro_entry is not None
        assert "andamentum-epistemic verify" in repro_entry["body"]
        # No standalone prose atom with id="reproduction" — that would
        # render the command a third time.
        assert not any(a.get("id") == "reproduction" for a in atoms)

    def test_meta_line_carries_pipeline_version_and_snapshot(self):
        data = _make_report_data()
        atoms = build_audit_report(data)
        heading = next(a for a in atoms if a.get("kind") == "heading")
        meta = str(heading.get("meta", ""))
        assert "0.3.0-rc1" in meta
        assert "deadbee" in meta
        assert "snapshot" in meta


# ──────────────────────────────────────────────────────────────────────────────
# Research mode (decomposition → multiple sub-claims)
# ──────────────────────────────────────────────────────────────────────────────


class TestResearchMode:
    def test_sub_claims_numbered_under_reasoning_trace(self):
        claims = [
            ClaimSummary(
                claim_id=f"c{i}",
                statement=f"Sub-claim {i}",
                scope="g",
                assumptions=[],
                stage="supported",
            )
            for i in range(3)
        ]
        atoms = build_audit_report(_make_report_data(claims=claims))
        section = _atoms_with_heading(atoms, "Reasoning trace")
        assert section
        sub_cards = [
            c for c in _atoms_of_kind(atoms, "card")
            if "Sub-claim" in str(c.get("content", ""))
        ]
        assert len(sub_cards) == 3
        for i, c in enumerate(sub_cards, start=1):
            content = str(c["content"])
            assert content.startswith(f"**Claim #{i}:**")


# ──────────────────────────────────────────────────────────────────────────────
# Appendix
# ──────────────────────────────────────────────────────────────────────────────


class TestAppendix:
    def test_appendix_card_present_with_evidence(self):
        ev = [
            EvidenceSummary(
                evidence_id=f"e{i}",
                source_type="pubmed",
                source_ref=f"PMID:{1000 + i}",
                extracted_content="x",
                judgment_reasoning="r",
                support_judgment="supports",
            )
            for i in range(3)
        ]
        atoms = build_audit_report(_make_report_data(evidence=ev))
        appendix = next(
            (c for c in _atoms_of_kind(atoms, "card") if c.get("id") == "appendix"),
            None,
        )
        assert appendix is not None
        details = appendix.get("details", "")
        assert "Supporting evidence (3)" in details

    def test_gate_trace_json_appendix_when_traces_exist(self):
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test",
            scope="g",
            assumptions=[],
            stage="supported",
            gate_trace=[
                GateTraceEntry(
                    name="scrutiny",
                    routing="PRIMARY",
                    required="pass",
                    observed="pass",
                    status="satisfied",
                ),
            ],
        )
        atoms = build_audit_report(_make_report_data(claims=[claim]))
        json_appendix = next(
            (
                c for c in _atoms_of_kind(atoms, "card")
                if c.get("id") == "appendix-gate-json"
            ),
            None,
        )
        assert json_appendix is not None
        details = json_appendix.get("details", "")
        assert '"name": "scrutiny"' in details
        assert '"status": "satisfied"' in details

    def test_no_appendix_when_no_evidence(self):
        atoms = build_audit_report(_make_report_data(evidence=[]))
        assert not any(
            c.get("id") == "appendix"
            for c in _atoms_of_kind(atoms, "card")
        )


# ──────────────────────────────────────────────────────────────────────────────
# Rendered-HTML drift defence — no green/red inline styles
# ──────────────────────────────────────────────────────────────────────────────


class TestRenderedHtmlIsColorNeutral:
    """If a future change reintroduces a tone-CSS variant or accidentally
    passes a badge value that triggers the existing green/red rule, this
    test catches it. The constraint is load-bearing — the user's
    explicit aesthetic constraint."""

    def test_no_green_or_red_inline_style_in_rendered_html(self):
        from andamentum.typeset import render

        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="g",
            assumptions=[],
            stage="supported",
            gate_trace=[
                GateTraceEntry(
                    name="scrutiny",
                    routing="PRIMARY",
                    required="pass",
                    observed="pass",
                    status="satisfied",
                ),
            ],
            ibe_candidates=[
                IBECandidate(
                    candidate_id="A",
                    verdict="supports",
                    description="x",
                    loveliness=0.8,
                    likeliness=0.7,
                    chosen=True,
                ),
            ],
        )
        ev = [
            EvidenceSummary(
                evidence_id="e1",
                source_type="pubmed",
                source_ref="PMID:1234567",
                extracted_content="x",
                judgment_reasoning="Direct support.",
                support_judgment="supports",
            )
        ]
        atoms = build_audit_report(
            _make_report_data(claims=[claim], evidence=ev, posterior=0.92)
        )
        html = render(atoms, style="article")
        # The badge CSS rules in typeset/atoms.py tint badges green when
        # data-value matches supports/supported/pass/approved, and red
        # when it matches contradicts/contradicted/challenged/fail/
        # rejected. The constraint is that no badge in the rendered HTML
        # carries a data-value matching those triggers — i.e. v2 verdict
        # labels lowercase to neutral CSS values.
        # The forbidden values appear in the inline ``<style>...</style>``
        # block (they're the CSS selectors themselves); we strip the
        # stylesheet so the assertion only sees real attributes.
        import re as _re

        body_html = _re.sub(
            r"<style[\s\S]*?</style>", "", html, flags=_re.IGNORECASE
        )
        forbidden_data_values = {
            'data-value="supports"',
            'data-value="supported"',
            'data-value="pass"',
            'data-value="approved"',
            'data-value="contradicts"',
            'data-value="contradicted"',
            'data-value="challenged"',
            'data-value="fail"',
            'data-value="rejected"',
        }
        for fdv in forbidden_data_values:
            assert fdv not in body_html, (
                f"Found color-triggering data-value in rendered HTML body: {fdv}"
            )
