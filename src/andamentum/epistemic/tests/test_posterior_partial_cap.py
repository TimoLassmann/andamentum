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


class TestSciFactCase54Shape:
    """Single-claim verify mode with cycle_capped=True AND an
    integrated_assessment. Reproduces the SciFact case 54 shape: the
    actual production failure was that compute_posterior returned 0.500
    even though the claim had verdict=contradicts conf=0.727 (gold=CON,
    refute correct). Three-way rule should now use the verdict with
    a confidence penalty rather than discard it."""

    async def test_capped_with_contradicts_verdict_returns_low_posterior(
        self, tmp_path: Path
    ) -> None:
        store = DocumentStore.for_database("scifact_54", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(
            description="AMPK activation increases inflammation-related fibrosis.",
            clarified_question="AMPK activation increases inflammation-related fibrosis.",
            question_type="verificatory",
            claim_to_verify="AMPK activation increases inflammation-related fibrosis.",
        )
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        claim = Claim(
            objective_id=obj.entity_id,
            statement="AMPK activation increases inflammation-related fibrosis.",
            scope="lung tissue",
            stage=ClaimStage.SUPPORTED,
            cycle_capped=True,
            integrated_assessment="contradicts",
            integrated_confidence=0.727,
        )
        await repo.save(claim)

        report = await compute_posterior(repo, obj.entity_id)
        assert report is not None
        # Pre-fix: posterior=0.5 oscillation_detected (cap discarded
        # the verdict). Post-fix: contradicts at 0.727 with cap penalty
        # 0.7 → effective conf 0.509 → claim_p = 0.246. Posterior
        # ≈ 0.25 — directional, correctly leaning CON.
        assert report.posterior < 0.30
        assert report.posterior > 0.15
        assert report.terminal_state == "completed"
        assert report.integration_verdict == "contradicts"
        # Provenance must surface the cap.
        assert "cycle cap" in report.explanation
        assert "penalty" in report.explanation


class TestSciFactCase957Shape:
    """Single-claim verify mode with cycle_capped=True, NO
    integrated_assessment, but evidence one-sided. Reproduces the
    SciFact case 957 shape: evidence pool was 23 supports / 5 contradicts
    (gold=SUP) but cycle_capped suppressed the counting signal,
    returning 0.500. Three-way rule routes to the counting fallback."""

    async def test_capped_no_verdict_one_sided_evidence_uses_counting(
        self, tmp_path: Path
    ) -> None:
        store = DocumentStore.for_database("scifact_957", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(
            description="Podocytes are motile and migrate in the presence of injury.",
            clarified_question="Podocytes are motile and migrate in the presence of injury.",
            question_type="verificatory",
            claim_to_verify="Podocytes are motile and migrate in the presence of injury.",
        )
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        claim = Claim(
            objective_id=obj.entity_id,
            statement="Podocytes are motile and migrate in the presence of injury.",
            scope="kidney injury",
            stage=ClaimStage.HYPOTHESIS,
            cycle_capped=True,
            scrutiny_verdict="pass",
        )
        await repo.save(claim)

        # Mimic the case 957 evidence shape: 23 supports / 5 contradicts.
        evidence_ids: list[str] = []
        for i in range(23):
            ev = Evidence(
                objective_id=obj.entity_id,
                source_type="pubmed",
                source_ref=f"https://pubmed/sup_{i}",
                extracted_content="supportive content",
                extracted=True,
                support_judgment="supports",
            )
            await repo.save(ev)
            evidence_ids.append(ev.entity_id)
        for i in range(5):
            ev = Evidence(
                objective_id=obj.entity_id,
                source_type="pubmed",
                source_ref=f"https://pubmed/con_{i}",
                extracted_content="contradicting content",
                extracted=True,
                support_judgment="contradicts",
            )
            await repo.save(ev)
            evidence_ids.append(ev.entity_id)
        claim.evidence_ids = evidence_ids
        claim.evidence_count = len(evidence_ids)
        await repo.save(claim)

        report = await compute_posterior(repo, obj.entity_id)
        assert report is not None
        # Pre-cycle-cap-fix: posterior=0.5 (oscillation_detected discarded
        # the capped claim's signal entirely).
        # Pre-counting-path-fix: posterior ≈ 1.0 (counting fallback ran
        # at full strength on the capped claim's evidence).
        # Post-counting-path-fix: counting fallback runs but the cap
        # penalty pulls the posterior toward neutral. log_odds = 18
        # → counting_posterior ≈ 1.0; pulled by 0.7 → 0.5 + 0.5*0.7 = 0.85.
        # Still directionally correct (SUP) but reflects the cap's
        # provenance.
        assert 0.80 < report.posterior < 0.90
        assert report.terminal_state == "completed"
        assert report.mode == "counting_fallback"
        # Provenance: cap fired, signal came via counting.
        assert "cycle cap" in report.explanation
        assert "counting_posterior pulled toward neutral" in report.explanation


class TestSciFactCase439V15Shape:
    """Cycle-capped, no integrated_assessment, small one-sided evidence
    pool. Reproduces the v15 case 439 regression: cluster weighting
    amplified 2 raw supports to weighted 4.20, producing posterior 0.985
    (confident SUP) on a gold-NEI claim. The counting-path cap penalty
    pulls this toward neutral so the posterior reflects the cap's
    provenance — still wrong direction (gold is NEI) but less
    confidently wrong, awaiting a future small-N dampener."""

    async def test_capped_small_pool_dampens_confidence(self, tmp_path: Path) -> None:
        store = DocumentStore.for_database("scifact_439_v15", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(
            description="Fz/PCP-dependent Pk localizes to the anterior membrane.",
            clarified_question=(
                "Fz/PCP-dependent Pk localizes to the anterior membrane."
            ),
            question_type="verificatory",
            claim_to_verify=("Fz/PCP-dependent Pk localizes to the anterior membrane."),
        )
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        claim = Claim(
            objective_id=obj.entity_id,
            statement=("Fz/PCP-dependent Pk localizes to the anterior membrane."),
            scope="zebrafish neurulation",
            stage=ClaimStage.HYPOTHESIS,
            cycle_capped=True,
            scrutiny_verdict="pass",
        )
        await repo.save(claim)

        # 2 supports / 0 contradicts (case 439 v15's exact shape, with
        # corroboration_count to mimic cluster weighting).
        ev_ids: list[str] = []
        for i in range(2):
            ev = Evidence(
                objective_id=obj.entity_id,
                source_type="open_targets",
                source_ref=f"https://opentargets/{i}",
                extracted_content="content",
                extracted=True,
                support_judgment="supports",
                corroboration_count=4,
            )
            await repo.save(ev)
            ev_ids.append(ev.entity_id)
        claim.evidence_ids = ev_ids
        claim.evidence_count = len(ev_ids)
        await repo.save(claim)

        report = await compute_posterior(repo, obj.entity_id)
        assert report is not None
        # Pre-counting-path-fix: posterior ≈ 0.985 (cluster weighting
        # amplified 2 supports to weighted 4.20, log_odds 4.20).
        # Post-fix: counting_posterior unchanged (~0.985), but pulled
        # by 0.7 → 0.5 + 0.485*0.7 = 0.840.
        assert 0.80 < report.posterior < 0.88
        assert report.terminal_state == "completed"
        assert report.mode == "counting_fallback"
        assert "cycle cap" in report.explanation


class TestGenuineOscillation:
    """Single capped claim with NO integrated_assessment AND essentially
    balanced counting (or empty evidence) is the case the original
    all-capped → 0.5 rule was trying to catch. The three-way rule still
    fires oscillation_detected here, just on much narrower grounds."""

    async def test_capped_no_verdict_no_evidence_emits_oscillation(
        self, tmp_path: Path
    ) -> None:
        store = DocumentStore.for_database("genuine_osc", db_dir=tmp_path)
        await store.initialize()
        repo = EpistemicRepository(store)
        obj = Objective(
            description="parent",
            clarified_question="parent",
            question_type="verificatory",
            claim_to_verify="claim X is true",
        )
        obj.objective_id = obj.entity_id
        await repo.save(obj)

        claim = Claim(
            objective_id=obj.entity_id,
            statement="claim X is true",
            scope="scope",
            stage=ClaimStage.HYPOTHESIS,
            cycle_capped=True,
            persistent_concerns=["unc-1", "unc-2"],
        )
        await repo.save(claim)

        report = await compute_posterior(repo, obj.entity_id)
        assert report is not None
        assert report.posterior == 0.5
        assert report.terminal_state == "oscillation_detected"
        assert "Oscillation detected" in report.explanation


class TestCapPenaltyEffect:
    """The same integrated verdict with vs without cycle_capped should
    yield different posteriors (the capped one closer to neutral),
    confirming the confidence penalty actually fires."""

    async def test_capped_pulls_toward_neutral(self, tmp_path: Path) -> None:
        async def _run(cycle_capped: bool, db_name: str) -> float:
            store = DocumentStore.for_database(db_name, db_dir=tmp_path)
            await store.initialize()
            repo = EpistemicRepository(store)
            obj = Objective(
                description="d",
                clarified_question="d",
                question_type="verificatory",
                claim_to_verify="claim X is true",
            )
            obj.objective_id = obj.entity_id
            await repo.save(obj)
            claim = Claim(
                objective_id=obj.entity_id,
                statement="claim X is true",
                scope="scope",
                stage=ClaimStage.SUPPORTED,
                cycle_capped=cycle_capped,
                integrated_assessment="supports",
                integrated_confidence=0.8,
            )
            await repo.save(claim)
            report = await compute_posterior(repo, obj.entity_id)
            assert report is not None
            return report.posterior

        p_uncapped = await _run(False, "cap_off")
        p_capped = await _run(True, "cap_on")

        # Uncapped: supports at 0.8 → 0.5 + 0.4 = 0.9.
        # Capped:   supports at 0.8 * 0.7 (penalty) = 0.56 → 0.5 + 0.28 = 0.78.
        assert p_uncapped > p_capped
        assert p_uncapped > 0.85
        assert 0.70 < p_capped < 0.85
