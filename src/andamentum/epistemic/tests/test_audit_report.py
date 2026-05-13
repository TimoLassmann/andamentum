"""Tests for the Cochrane-style audit report renderer.

The audit report is a parallel layout to ``typeset_report.py`` — same
ReportData input, different shape. These tests pin the rendering
contract using in-memory ``ReportData`` (no DB, no LLM, no HTTP).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from andamentum.epistemic.audit_report import (
    _source_url,
    build_audit_report,
)
from andamentum.epistemic.report_data import (
    AdversarialSummary,
    ClaimSummary,
    ConfidenceScores,
    EvidenceSummary,
    IBECandidate,
    InvestigationRound,
    InvestigationStats,
    ReportData,
)


def _atoms_with_heading(atoms: list[dict[str, Any]], heading: str) -> list[dict[str, Any]]:
    return [a for a in atoms if a.get("heading") == heading]


def _atoms_of_kind(atoms: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [a for a in atoms if a.get("kind") == kind]


def _make_report_data(
    *,
    claims: list[ClaimSummary] | None = None,
    evidence: list[EvidenceSummary] | None = None,
    adversarial: list[AdversarialSummary] | None = None,
    stats: InvestigationStats | None = None,
    terminal_state: str = "completed",
    posterior: float | None = 0.85,
) -> ReportData:
    return ReportData(
        research_question="Aspirin reduces colorectal cancer risk",
        clarified_question="Aspirin reduces colorectal cancer risk in adults",
        investigation_date=datetime(2026, 5, 13),
        model_used="openai:gpt-5.4-nano",
        database_name="test",
        direct_answer="The evidence supports the claim with caveats.",
        question_type="verificatory",
        verdict="Supported with caveats.",
        claims=claims or [],
        evidence=evidence or [],
        uncertainties=[],
        adversarial=adversarial or [],
        convergence=[],
        open_questions=[],
        stats=stats or InvestigationStats(),
        confidence_scores=ConfidenceScores(
            posterior=posterior,
            posterior_question_type="verificatory",
            terminal_state=terminal_state,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Source-URL helper
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
# Verdict callout and headline
# ──────────────────────────────────────────────────────────────────────────────


class TestVerdictCallout:
    def test_supported_when_posterior_above_threshold(self):
        data = _make_report_data(posterior=0.85)
        atoms = build_audit_report(data)
        callouts = _atoms_of_kind(atoms, "callout")
        assert any("**Supported**" in str(a.get("content", "")) for a in callouts)
        # tone should be success.
        assert any(a.get("tone") == "success" for a in callouts)

    def test_refuted_when_posterior_below(self):
        data = _make_report_data(posterior=0.15)
        atoms = build_audit_report(data)
        callouts = _atoms_of_kind(atoms, "callout")
        assert any("**Refuted**" in str(a.get("content", "")) for a in callouts)
        assert any(a.get("tone") == "warning" for a in callouts)

    def test_inconclusive_in_middle_band(self):
        data = _make_report_data(posterior=0.5)
        atoms = build_audit_report(data)
        callouts = _atoms_of_kind(atoms, "callout")
        assert any("**Inconclusive**" in str(a.get("content", "")) for a in callouts)

    def test_insufficient_when_non_completed_terminal(self):
        data = _make_report_data(
            terminal_state="oscillation_detected", posterior=0.5
        )
        atoms = build_audit_report(data)
        callouts = _atoms_of_kind(atoms, "callout")
        assert any(
            "Insufficient evidence" in str(a.get("content", "")) for a in callouts
        )
        # Pill should say "Suspended" not a percentage.
        assert any("Suspended" in str(a.get("content", "")) for a in callouts)


# ──────────────────────────────────────────────────────────────────────────────
# Summary-of-findings table
# ──────────────────────────────────────────────────────────────────────────────


class TestSummaryOfFindingsTable:
    def test_table_rendered(self):
        stats = InvestigationStats(
            total_evidence=100,
            evidence_supports=24,
            evidence_contradicts=4,
            evidence_no_bearing=72,
        )
        data = _make_report_data(stats=stats)
        atoms = build_audit_report(data)
        sof = _atoms_with_heading(atoms, "Summary of findings")
        assert sof, "Summary of findings section should exist"
        text = sof[0]["content"]
        # The total appears.
        assert "100 evidence items" in text
        # Each direction appears in the table.
        assert "Supporting" in text
        assert "24" in text
        assert "Contradicting" in text
        assert "No bearing" in text
        # Percentages show.
        assert "%" in text


# ──────────────────────────────────────────────────────────────────────────────
# Per-claim layout — verify mode
# ──────────────────────────────────────────────────────────────────────────────


class TestSingleClaimRendering:
    def test_no_inline_reference_number_list_in_claim_card(self):
        """The classic layout dumps every evidence number on the claim
        card. The audit layout must NOT — counts only, in details."""
        claim = ClaimSummary(
            claim_id="c1",
            statement="Aspirin reduces CRC risk",
            scope="general",
            assumptions=[],
            stage="supported",
            evidence_ids=[f"e{i}" for i in range(98)],  # huge list
            evidence_refs_display=list(range(1, 99)),
        )
        data = _make_report_data(claims=[claim])
        atoms = build_audit_report(data)
        card_atoms = _atoms_of_kind(atoms, "card")
        # The claim card should not carry a refs list.
        for c in card_atoms:
            assert "refs" not in c or not c.get("refs"), (
                "Audit claim card must not list inline reference numbers"
            )

    def test_evidence_counts_in_card_details(self):
        ev_supp = [
            EvidenceSummary(
                evidence_id=f"s{i}",
                source_type="pubmed",
                source_ref=f"PMID:{i}",
                extracted_content=f"supports paper {i}",
                support_judgment="supports",
            )
            for i in range(7)
        ]
        ev_con = [
            EvidenceSummary(
                evidence_id="c1",
                source_type="pubmed",
                source_ref="PMID:99",
                extracted_content="null result",
                support_judgment="contradicts",
            )
        ]
        claim = ClaimSummary(
            claim_id="c1",
            statement="Aspirin reduces CRC risk",
            scope="g",
            assumptions=[],
            stage="supported",
            evidence_ids=[ev.evidence_id for ev in ev_supp + ev_con],
        )
        data = _make_report_data(claims=[claim], evidence=ev_supp + ev_con)
        atoms = build_audit_report(data)
        card_atoms = _atoms_of_kind(atoms, "card")
        claim_card = next(c for c in card_atoms if "Aspirin" in str(c.get("content", "")))
        details = claim_card.get("details", "")
        assert "7 supporting" in details
        assert "1 contradicting" in details

    def test_top_supporting_section_has_clickable_links(self):
        ev = [
            EvidenceSummary(
                evidence_id="e1",
                source_type="pubmed",
                source_ref="PMID:12345678",
                extracted_content="Strong RCT result.",
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
        # Look for the supporting-evidence section.
        prose_atoms = _atoms_of_kind(atoms, "prose")
        supp = next(
            (
                a for a in prose_atoms
                if "Strongest supporting evidence" in str(a.get("heading", ""))
            ),
            None,
        )
        assert supp is not None
        # The body should contain a markdown link to the PubMed URL.
        body = supp["content"]
        assert "https://pubmed.ncbi.nlm.nih.gov/12345678/" in body

    def test_audit_trail_in_card_details(self):
        """Investigation rounds and IBE candidates go into the claim
        card's collapsible details, not as separate top-level sections."""
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="g",
            assumptions=[],
            stage="supported",
            investigation_rounds=[
                InvestigationRound(
                    round_index=1,
                    intent="adversarial evidence",
                    evidence_count=3,
                ),
            ],
            ibe_candidates=[
                IBECandidate(
                    candidate_id="A",
                    verdict="supports",
                    description="Direct mechanism explanation.",
                    loveliness=0.8,
                    likeliness=0.7,
                    chosen=True,
                ),
            ],
        )
        atoms = build_audit_report(_make_report_data(claims=[claim]))
        card_atoms = _atoms_of_kind(atoms, "card")
        claim_card = next(c for c in card_atoms if "Test claim" in str(c.get("content", "")))
        details = claim_card.get("details", "")
        # Investigation round rendered.
        assert "Round 1" in details
        assert "adversarial evidence" in details
        assert "(yielded 3 items)" in details
        # IBE table rendered.
        assert "Candidate" in details or "| ID |" in details
        assert "selected" in details
        # Loveliness/likeliness scores show.
        assert "0.80" in details or "0.8" in details


# ──────────────────────────────────────────────────────────────────────────────
# Research mode (decomposition → multiple sub-claims)
# ──────────────────────────────────────────────────────────────────────────────


class TestResearchMode:
    def test_subinvestigations_section_when_multiple_claims(self):
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
        data = _make_report_data(claims=claims)
        atoms = build_audit_report(data)
        sub_sections = _atoms_with_heading(atoms, "Sub-investigations")
        assert sub_sections, "Research-mode report should have a Sub-investigations heading"
        # Each sub-claim is rendered as a card.
        card_atoms = _atoms_of_kind(atoms, "card")
        sub_cards = [
            c for c in card_atoms
            if "Sub-claim" in str(c.get("content", ""))
        ]
        assert len(sub_cards) == 3
        # Each sub-claim is prefixed with #1, #2, #3.
        for i, c in enumerate(sub_cards, start=1):
            assert f"#{i}" in str(c.get("content", ""))

    def test_key_evidence_heading_when_single_claim(self):
        """Verify-mode reports use 'Key evidence' instead of
        'Sub-investigations' for the per-claim section."""
        claims = [
            ClaimSummary(
                claim_id="c1",
                statement="Single claim",
                scope="g",
                assumptions=[],
                stage="supported",
            )
        ]
        data = _make_report_data(claims=claims)
        atoms = build_audit_report(data)
        assert _atoms_with_heading(atoms, "Key evidence")
        assert not _atoms_with_heading(atoms, "Sub-investigations")


# ──────────────────────────────────────────────────────────────────────────────
# Appendix — full evidence trail (collapsible)
# ──────────────────────────────────────────────────────────────────────────────


class TestAppendix:
    def test_full_evidence_appendix_card_present(self):
        ev = [
            EvidenceSummary(
                evidence_id=f"e{i}",
                source_type="pubmed",
                source_ref=f"PMID:{i}",
                extracted_content="content",
                support_judgment="supports",
            )
            for i in range(3)
        ]
        atoms = build_audit_report(_make_report_data(evidence=ev))
        card_atoms = _atoms_of_kind(atoms, "card")
        appendix = next(
            (c for c in card_atoms if c.get("id") == "appendix-evidence"),
            None,
        )
        assert appendix is not None, "Appendix card should be present when evidence exists"
        # The details should list each item.
        details = appendix.get("details", "")
        assert "PMID:0" in details or "https://pubmed" in details
        assert "Supporting evidence (3)" in details

    def test_no_appendix_when_no_evidence(self):
        atoms = build_audit_report(_make_report_data(evidence=[]))
        card_atoms = _atoms_of_kind(atoms, "card")
        assert not any(c.get("id") == "appendix-evidence" for c in card_atoms)
