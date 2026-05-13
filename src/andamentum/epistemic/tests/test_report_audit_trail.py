"""Tests for the audit-trail sections of the typeset report.

The report renders four pieces of audit-trail data that pre-existed in
the entity state but were not previously surfaced:

1. **Per-claim investigation rounds** — every follow-up intent the
   gap-analysis agent proposed, with yield-per-intent counts.
2. **Evidence judgement breakdown** — total support / contradict /
   no_bearing counts as a "Sources" header line, plus
   judgement-per-item rendering (already partially in place).
3. **Adversarial probes** — counterarguments rendered with an
   explicit "we searched for contradicting evidence" intro, so the
   reader sees the probe, not just the result.
4. **IBE chain candidates** — alternative explanations the
   integration step considered, with their loveliness / likeliness
   scores and which was selected.

These tests pin the rendering shape and the data plumbing. They do
not require a live LLM or HTTP call — they construct ReportData in
memory and assert against the rendered atoms.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

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
from andamentum.epistemic.typeset_report import build_typeset_report


def _atoms_with_text(atoms: list[dict[str, Any]], needle: str) -> list[dict[str, Any]]:
    """Filter atoms whose content (or heading) contains ``needle``."""
    out: list[dict[str, Any]] = []
    for a in atoms:
        text = str(a.get("content", "")) + " " + str(a.get("heading", ""))
        if needle in text:
            out.append(a)
    return out


def _make_minimal_data(
    *,
    claims: list[ClaimSummary] | None = None,
    evidence: list[EvidenceSummary] | None = None,
    adversarial: list[AdversarialSummary] | None = None,
    stats: InvestigationStats | None = None,
) -> ReportData:
    return ReportData(
        research_question="Test claim",
        clarified_question="Test claim",
        investigation_date=datetime.now(),
        model_used="stub",
        database_name="test",
        direct_answer="answer",
        question_type="verificatory",
        verdict="Test verdict.",
        claims=claims or [],
        evidence=evidence or [],
        uncertainties=[],
        adversarial=adversarial or [],
        convergence=[],
        open_questions=[],
        stats=stats or InvestigationStats(),
        confidence_scores=ConfidenceScores(
            posterior=0.5,
            posterior_question_type="verificatory",
            terminal_state="completed",
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Investigation rounds
# ──────────────────────────────────────────────────────────────────────────────


class TestInvestigationRoundsSection:
    def test_rendered_when_rounds_present(self) -> None:
        claim = ClaimSummary(
            claim_id="c1",
            statement="Aspirin reduces colorectal cancer risk",
            scope="general",
            assumptions=[],
            stage="supported",
            investigation_rounds=[
                InvestigationRound(
                    round_index=1,
                    intent="adversarial evidence: replication failures in cohorts",
                    evidence_count=4,
                ),
                InvestigationRound(
                    round_index=2,
                    intent="mechanistic studies at the molecular level",
                    evidence_count=12,
                ),
                InvestigationRound(
                    round_index=3,
                    intent="independent replication in a different model system",
                    evidence_count=0,
                ),
            ],
        )
        data = _make_minimal_data(claims=[claim])
        atoms = build_typeset_report(data)
        prose_atoms = _atoms_with_text(atoms, "How this claim was investigated")
        assert prose_atoms, "Investigation-rounds section should render"
        full_text = prose_atoms[0]["content"]
        # Each round's intent shows with its yield count.
        assert "Round 1" in full_text
        assert "(yielded 4 items)" in full_text
        assert "adversarial evidence" in full_text
        assert "Round 2" in full_text
        assert "(yielded 12 items)" in full_text
        # Singular for yield=1 / dead-end signal preserved as 0.
        assert "Round 3" in full_text
        assert "(yielded 0 items)" in full_text

    def test_skipped_when_no_rounds(self) -> None:
        """A claim that reached a verdict on initial gather alone has
        no investigation rounds — the section must not render an empty
        block."""
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test claim",
            scope="general",
            assumptions=[],
            stage="supported",
            investigation_rounds=[],
        )
        data = _make_minimal_data(claims=[claim])
        atoms = build_typeset_report(data)
        assert not _atoms_with_text(atoms, "How this claim was investigated")

    def test_singular_item_yield_renders_correctly(self) -> None:
        """``yielded 1 item`` (no plural ``s``) for round with single result."""
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test",
            scope="g",
            assumptions=[],
            stage="supported",
            investigation_rounds=[
                InvestigationRound(round_index=1, intent="angle X", evidence_count=1)
            ],
        )
        atoms = build_typeset_report(_make_minimal_data(claims=[claim]))
        full_text = _atoms_with_text(atoms, "How this claim")[0]["content"]
        assert "(yielded 1 item)" in full_text
        assert "(yielded 1 items)" not in full_text


# ──────────────────────────────────────────────────────────────────────────────
# IBE chain candidates
# ──────────────────────────────────────────────────────────────────────────────


class TestIBESection:
    def test_rendered_when_candidates_present(self) -> None:
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test",
            scope="g",
            assumptions=[],
            stage="robust",
            ibe_candidates=[
                IBECandidate(
                    candidate_id="A",
                    verdict="supports",
                    description="The claim's headline mechanism is correct.",
                    loveliness=0.78,
                    likeliness=0.82,
                    chosen=True,
                ),
                IBECandidate(
                    candidate_id="B",
                    verdict="contradicts",
                    description="The observed effect is confounding by healthy-user bias.",
                    loveliness=0.55,
                    likeliness=0.45,
                    runner_up=True,
                ),
                IBECandidate(
                    candidate_id="C",
                    verdict="insufficient",
                    description="The effect modifies by tumour subtype only.",
                    loveliness=0.30,
                    likeliness=0.40,
                ),
            ],
            integrated_assessment="supports",
        )
        atoms = build_typeset_report(_make_minimal_data(claims=[claim]))
        ibe_atoms = _atoms_with_text(atoms, "Inference to the best explanation")
        assert ibe_atoms, "IBE chain section should render"
        text = ibe_atoms[0]["content"]
        # All three candidates appear, with their score tags.
        assert "Candidate A" in text
        assert "**selected**" in text
        assert "loveliness 0.78" in text
        assert "Candidate B" in text
        assert "runner-up" in text
        assert "Candidate C" in text
        # Integrated assessment surfaced.
        assert "**Integrated assessment**" in text
        assert "supports" in text

    def test_skipped_when_no_candidates(self) -> None:
        """A claim that never reached IBE (cycle-capped or abandoned)
        has no candidates — the section must not render."""
        claim = ClaimSummary(
            claim_id="c1",
            statement="Test",
            scope="g",
            assumptions=[],
            stage="hypothesis",
            ibe_candidates=[],
        )
        atoms = build_typeset_report(_make_minimal_data(claims=[claim]))
        assert not _atoms_with_text(atoms, "Inference to the best explanation")


# ──────────────────────────────────────────────────────────────────────────────
# Adversarial probe reframing
# ──────────────────────────────────────────────────────────────────────────────


class TestAdversarialProbe:
    def test_probe_intro_rendered(self) -> None:
        """When counterarguments exist for a claim, the report renders
        an explicit intro saying the system searched for contradicting
        evidence — not just the bare counterargument list."""
        claim = ClaimSummary(
            claim_id="c1",
            statement="Aspirin reduces CRC risk",
            scope="general",
            assumptions=[],
            stage="supported",
        )
        adv = AdversarialSummary(
            claim_id="c1",
            counterargument="The Physicians' Health Study reported no association.",
            strength=0.7,
            source_ref="PMID:9556464",
            rebuttal=None,
        )
        atoms = build_typeset_report(
            _make_minimal_data(claims=[claim], adversarial=[adv])
        )
        intro_atoms = _atoms_with_text(atoms, "Adversarial probe")
        assert intro_atoms, "Adversarial probe intro should render"
        intro_text = intro_atoms[0]["content"]
        assert "**contradict**" in intro_text
        assert "1 challenge" in intro_text  # singular


# ──────────────────────────────────────────────────────────────────────────────
# Evidence judgement breakdown
# ──────────────────────────────────────────────────────────────────────────────


class TestEvidenceJudgementBreakdown:
    def test_breakdown_in_sources_section(self) -> None:
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
        stats = InvestigationStats(
            total_evidence=10,
            evidence_supports=3,
            evidence_contradicts=4,
            evidence_no_bearing=3,
        )
        atoms = build_typeset_report(_make_minimal_data(evidence=ev, stats=stats))
        # Filter on the Sources *heading* specifically — the summary
        # atom also mentions "Evidence Sources".
        sources_atoms = [
            a for a in atoms if a.get("heading") == "Sources"
        ]
        assert sources_atoms
        text = sources_atoms[0]["content"]
        assert "10 evidence items" in text
        assert "**3 supports**" in text
        assert "**4 contradicts**" in text
        assert "**3 no bearing**" in text
        # Percentages computed.
        assert "30%" in text or "40%" in text

    def test_skipped_when_no_evidence(self) -> None:
        atoms = build_typeset_report(_make_minimal_data(evidence=[]))
        sources_atoms = [
            a for a in atoms if a.get("heading") == "Sources"
        ]
        assert not sources_atoms
