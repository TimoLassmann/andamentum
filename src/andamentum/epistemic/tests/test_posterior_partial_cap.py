"""Tests for Phase 2: compute_posterior partial-cap fix.

Bug context: under multi-seed-claim, ONE Objective hosts N Claims. The
previous oscillation short-circuit fired if ANY claim was cycle_capped,
discarding healthy verdicts on sibling claims. The fix:

* All N claims capped → emit ``terminal_state="oscillation_detected"``
  with posterior=0.5 (honest: no signal to aggregate).
* Some capped, some healthy → drop the capped from aggregation, compute
  posterior on the healthy subset, surface the partial-cap count in
  the explanation. The combiner / consumer sees a real posterior with
  a NOTE about the cap.

This restores multi-claim resilience: one cycling claim doesn't take
down 4 healthy ones with it.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.document_store import DocumentStore
from andamentum.epistemic.confidence import compute_posterior
from andamentum.epistemic.entities import Claim, Evidence, Objective
from andamentum.epistemic.entities.claim import ClaimStage
from andamentum.epistemic.repository import EpistemicRepository


async def _setup_with_mixed_capped_claims(
    tmp_path: Path,
    db_name: str,
    *,
    capped_count: int,
    healthy_count: int,
    healthy_verdict: str = "supports",
    healthy_confidence: float = 0.8,
) -> tuple[Objective, EpistemicRepository]:
    """Build an Objective with `capped_count` cycle_capped claims and
    `healthy_count` claims with a real integration verdict."""
    store = DocumentStore.for_database(db_name, db_dir=tmp_path)
    await store.initialize()
    repo = EpistemicRepository(store)
    obj = Objective(
        description="parent",
        question_type="verificatory",
        clarified_question="parent",
    )
    obj.objective_id = obj.entity_id
    await repo.save(obj)
    for i in range(capped_count):
        claim = Claim(
            objective_id=obj.entity_id,
            statement=f"capped claim {i}",
            scope="scope",
            stage=ClaimStage.HYPOTHESIS,
            cycle_capped=True,
            persistent_concerns=[f"unc-{i}-1", f"unc-{i}-2"],
        )
        await repo.save(claim)
    for i in range(healthy_count):
        claim = Claim(
            objective_id=obj.entity_id,
            statement=f"healthy claim {i}",
            scope="scope",
            stage=ClaimStage.SUPPORTED,
            scrutiny_verdict="pass",
            integrated_assessment=healthy_verdict,
            integrated_confidence=healthy_confidence,
        )
        await repo.save(claim)
    return obj, repo


class TestPartialCapAggregates:
    async def test_one_capped_four_healthy_aggregates_healthy_only(
        self, tmp_path: Path
    ) -> None:
        """Case 54-style scenario: 1 capped claim + 4 supporting claims.
        Pre-fix: posterior=0.5 oscillation_detected (4 verdicts dropped).
        Post-fix: aggregate the 4 supports; explanation notes the cap."""
        obj, repo = await _setup_with_mixed_capped_claims(
            tmp_path,
            "partial_one_capped",
            capped_count=1,
            healthy_count=4,
            healthy_verdict="supports",
            healthy_confidence=0.8,
        )
        report = await compute_posterior(repo, obj.entity_id)
        assert report is not None
        # supports at 0.8 → claim_p = 0.5 + 0.8/2 = 0.9; aggregated over 4.
        # Mean of [0.9]*4 weighted by 0.8 each = 0.9.
        assert report.posterior > 0.85
        # Terminal state stays "completed" — partial cap is a NOTE, not
        # a terminal failure. The healthy aggregate is real.
        assert report.terminal_state == "completed"
        # Explanation surfaces the cap count.
        assert "1 claim" in report.explanation
        assert "cycle cap" in report.explanation

    async def test_two_capped_three_healthy_contradicts_aggregate(
        self, tmp_path: Path
    ) -> None:
        """Three contradicting claims at 0.7 + two capped → posterior
        leans contradicts (≈0.15), not 0.5."""
        obj, repo = await _setup_with_mixed_capped_claims(
            tmp_path,
            "partial_two_capped",
            capped_count=2,
            healthy_count=3,
            healthy_verdict="contradicts",
            healthy_confidence=0.7,
        )
        report = await compute_posterior(repo, obj.entity_id)
        assert report is not None
        # contradicts at 0.7 → claim_p = 0.5 - 0.7/2 = 0.15.
        assert report.posterior < 0.20
        assert report.terminal_state == "completed"
        assert "2 claim" in report.explanation


class TestAllCappedShortCircuits:
    async def test_all_active_capped_emits_oscillation_detected(
        self, tmp_path: Path
    ) -> None:
        """When EVERY active claim is capped, there's nothing to
        aggregate — posterior=0.5 with terminal_state oscillation_detected
        is the honest report."""
        obj, repo = await _setup_with_mixed_capped_claims(
            tmp_path,
            "all_capped",
            capped_count=3,
            healthy_count=0,
        )
        report = await compute_posterior(repo, obj.entity_id)
        assert report is not None
        assert report.posterior == 0.5
        assert report.terminal_state == "oscillation_detected"
        assert "ALL" in report.explanation


class TestNoCappedUnchanged:
    async def test_no_cap_normal_aggregation(self, tmp_path: Path) -> None:
        """Sanity: when no claim is capped, behavior is exactly as
        before — no NOTE about cap, no terminal_state mismatch."""
        obj, repo = await _setup_with_mixed_capped_claims(
            tmp_path,
            "none_capped",
            capped_count=0,
            healthy_count=3,
            healthy_verdict="supports",
            healthy_confidence=0.8,
        )
        report = await compute_posterior(repo, obj.entity_id)
        assert report is not None
        assert report.posterior > 0.85
        assert report.terminal_state == "completed"
        # No NOTE about cap exclusion when there's nothing to exclude.
        assert "cycle cap" not in report.explanation


class TestCase54Reproduction:
    """Reproduce the case 54 scenario in miniature: 7 claims, 2 cycle-
    capped, 1 with contradicts verdict at 0.75 confidence, 4 with no
    integration verdict (counting fallback territory)."""

    async def test_case_54_mini(self, tmp_path: Path) -> None:
        store = DocumentStore.for_database("case54_mini", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(
            description="AMPK activation increases fibrosis",
            clarified_question="AMPK activation increases fibrosis",
            question_type="verificatory",  # Phase 3 will force this
        )
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        # 2 cycle-capped (B and E in the report)
        for i in range(2):
            claim = Claim(
                objective_id=obj.entity_id,
                statement=f"capped sub {i}",
                scope="scope",
                stage=ClaimStage.HYPOTHESIS,
                cycle_capped=True,
                persistent_concerns=[f"unc-{i}"],
            )
            await repo.save(claim)

        # 1 with contradicts at 0.75 (sub A in the report)
        claim_a = Claim(
            objective_id=obj.entity_id,
            statement="AMPK has direct cytokine effects",
            scope="scope",
            stage=ClaimStage.SUPPORTED,
            integrated_assessment="contradicts",
            integrated_confidence=0.75,
        )
        await repo.save(claim_a)

        # 4 with no integration verdict (subs C, D, F, G in the report).
        # We'll add sparse evidence so counting-fallback isn't 0/0.
        for i in range(4):
            claim = Claim(
                objective_id=obj.entity_id,
                statement=f"no-verdict sub {i}",
                scope="scope",
                stage=ClaimStage.HYPOTHESIS,
            )
            await repo.save(claim)
            ev = Evidence(
                objective_id=obj.entity_id,
                source_type="web",
                source_ref=f"https://ex.com/{i}",
                extracted_content="x",
                extracted=True,
                support_judgment="contradicts" if i % 2 else "supports",
            )
            await repo.save(ev)
            claim.evidence_ids = [ev.entity_id]
            claim.evidence_count = 1
            await repo.save(claim)

        report = await compute_posterior(repo, obj.entity_id)
        assert report is not None
        # Pre-fix: posterior=0.5 oscillation_detected (drops claim A's
        # contradicts verdict).
        # Post-fix: aggregate the integrated claim. claim A says
        # contradicts at 0.75 → claim_p = 0.125. With only one
        # integrated claim, the integration aggregate is 0.125.
        # The NOTE in explanation flags the 2 capped claims.
        assert report.posterior < 0.20
        assert report.terminal_state == "completed"
        assert "2 claim" in report.explanation
        assert report.integration_verdict == "contradicts"
