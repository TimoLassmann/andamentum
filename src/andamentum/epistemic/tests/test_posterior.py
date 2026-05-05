"""Tests for posterior P(Y) scoring via compute_posterior().

Covers:
- All supporting / all contradicting / balanced / no evidence
- Uninformative prior (no evidence → 0.5)
- Invalidated evidence excluded
- Corroborative evidence excluded (only representative counts)
- "no_bearing" evidence excluded
- Abandoned claims excluded
- Question type eligibility (explanatory, exploratory → None; verificatory, comparative, predictive → report)
- Evidence aggregated across multiple claims
"""

import math
import pytest

from ..confidence import PosteriorReport, compute_posterior
from ..entities import Claim, Evidence, Objective


OBJ_ID = "test-posterior-obj"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_objective(
    question_type: str | None = "verificatory",
    objective_id: str = OBJ_ID,
) -> Objective:
    return Objective(
        entity_id=objective_id,
        objective_id=objective_id,
        description="Test posterior question",
        question_type=question_type,
    )


def _make_claim(
    evidence_ids: list[str] | None = None,
    abandoned: bool = False,
    objective_id: str = OBJ_ID,
    integrated_assessment: str | None = None,
    integrated_confidence: float | None = None,
) -> Claim:
    """Build a test Claim.

    By default ``integrated_assessment=None`` — but the no-certified-
    verdict gate (added 2026-05-05) suspends the posterior to 0.5 when
    no claim has IA. Tests of the counting math should set IA so the
    gate doesn't fire AND assert on ``counting_posterior`` (the
    diagnostic), not on ``posterior`` (which now follows the
    integration verdict)."""
    return Claim(
        objective_id=objective_id,
        statement="Test claim for posterior",
        evidence_ids=evidence_ids or [],
        abandoned=abandoned,
        integrated_assessment=integrated_assessment,
        integrated_confidence=integrated_confidence,
    )


def _make_evidence(
    support_judgment: str | None = "supports",
    invalidated: bool = False,
    cluster_status: str = "representative",
    objective_id: str = OBJ_ID,
    entity_id: str | None = None,
) -> Evidence:
    kwargs: dict = dict(
        objective_id=objective_id,
        source_type="web_search",
        source_ref="https://example.com",
        extracted_content="Some content",
        extracted=True,
        support_judgment=support_judgment,
        invalidated=invalidated,
        cluster_status=cluster_status,
    )
    if entity_id is not None:
        kwargs["entity_id"] = entity_id
    return Evidence(**kwargs)


# =========================================================================
# Evidence direction tests
# =========================================================================


class TestPosteriorEvidenceDirection:
    """Tests that evidence direction drives posterior correctly."""

    async def test_all_supporting_evidence_high_diagnostic_counting(self, repo):
        """All supporting evidence should produce a high counting_posterior
        (the diagnostic). The headline posterior suspends to 0.5 because
        no claim has integrated_assessment — the no-certified-verdict
        gate (added 2026-05-05) requires IBE certification before
        committing to a directional posterior."""
        obj = _make_objective()
        await repo.save(obj)

        e1 = _make_evidence(support_judgment="supports", entity_id="e-s1")
        e2 = _make_evidence(support_judgment="supports", entity_id="e-s2")
        e3 = _make_evidence(support_judgment="supports", entity_id="e-s3")
        await repo.save(e1)
        await repo.save(e2)
        await repo.save(e3)

        claim = _make_claim(evidence_ids=["e-s1", "e-s2", "e-s3"])
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        # Headline posterior suspends — no IBE certification.
        assert report.posterior == 0.5
        assert report.terminal_state == "oscillation_detected"
        # Diagnostic counts still computed and exposed.
        assert report.supporting_count == 3
        assert report.contradicting_count == 0
        assert report.counting_posterior > 0.5

    async def test_all_contradicting_evidence_low_diagnostic_counting(self, repo):
        """All contradicting evidence — diagnostic counting reflects it
        even though the headline posterior suspends. See the analogous
        ``test_all_supporting_evidence_high_diagnostic_counting``."""
        obj = _make_objective()
        await repo.save(obj)

        e1 = _make_evidence(support_judgment="contradicts", entity_id="e-c1")
        e2 = _make_evidence(support_judgment="contradicts", entity_id="e-c2")
        e3 = _make_evidence(support_judgment="contradicts", entity_id="e-c3")
        await repo.save(e1)
        await repo.save(e2)
        await repo.save(e3)

        claim = _make_claim(evidence_ids=["e-c1", "e-c2", "e-c3"])
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        assert report.posterior == 0.5
        assert report.terminal_state == "oscillation_detected"
        assert report.supporting_count == 0
        assert report.contradicting_count == 3
        assert report.counting_posterior < 0.5

    async def test_balanced_evidence_posterior_near_half(self, repo):
        """Equal supporting and contradicting evidence should produce posterior near 0.5."""
        obj = _make_objective()
        await repo.save(obj)

        e1 = _make_evidence(support_judgment="supports", entity_id="e-b1")
        e2 = _make_evidence(support_judgment="contradicts", entity_id="e-b2")
        await repo.save(e1)
        await repo.save(e2)

        claim = _make_claim(evidence_ids=["e-b1", "e-b2"])
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        assert report.posterior == pytest.approx(0.5)
        assert report.supporting_count == 1
        assert report.contradicting_count == 1
        assert report.log_odds == 0

    async def test_no_evidence_posterior_exactly_half(self, repo):
        """No evidence at all should produce posterior exactly 0.5 (uninformative prior)."""
        obj = _make_objective()
        await repo.save(obj)

        claim = _make_claim(evidence_ids=[])
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        assert report.posterior == pytest.approx(0.5)
        assert report.supporting_count == 0
        assert report.contradicting_count == 0
        assert report.log_odds == 0


# =========================================================================
# Evidence filtering tests
# =========================================================================


class TestPosteriorEvidenceFiltering:
    """Tests that invalidated, corroborative, no_bearing evidence are excluded."""

    async def test_invalidated_evidence_excluded(self, repo):
        """Invalidated evidence should not count toward diagnostic
        counting (and would not count toward posterior if the
        no-certified-verdict gate weren't suspending the posterior
        in this no-IA fixture)."""
        obj = _make_objective()
        await repo.save(obj)

        e_valid = _make_evidence(support_judgment="supports", entity_id="e-v1")
        e_invalid = _make_evidence(
            support_judgment="contradicts", invalidated=True, entity_id="e-inv1"
        )
        await repo.save(e_valid)
        await repo.save(e_invalid)

        claim = _make_claim(evidence_ids=["e-v1", "e-inv1"])
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        # Only the valid supporting evidence is counted in the
        # diagnostic — invalidated is excluded.
        assert report.supporting_count == 1
        assert report.contradicting_count == 0

    async def test_corroborative_evidence_excluded(self, repo):
        """Corroborative (non-representative) evidence should not count."""
        obj = _make_objective()
        await repo.save(obj)

        e_rep = _make_evidence(
            support_judgment="supports",
            cluster_status="representative",
            entity_id="e-rep",
        )
        e_corr = _make_evidence(
            support_judgment="supports",
            cluster_status="corroborative",
            entity_id="e-corr",
        )
        e_def = _make_evidence(
            support_judgment="supports", cluster_status="deferred", entity_id="e-def"
        )
        await repo.save(e_rep)
        await repo.save(e_corr)
        await repo.save(e_def)

        claim = _make_claim(evidence_ids=["e-rep", "e-corr", "e-def"])
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        # Only the representative evidence should count
        assert report.supporting_count == 1
        assert report.contradicting_count == 0

    async def test_no_bearing_evidence_excluded(self, repo):
        """Evidence with support_judgment 'no_bearing' should not count
        toward the diagnostic supporting / contradicting tallies."""
        obj = _make_objective()
        await repo.save(obj)

        e_sup = _make_evidence(support_judgment="supports", entity_id="e-sup")
        e_nb = _make_evidence(support_judgment="no_bearing", entity_id="e-nb")
        await repo.save(e_sup)
        await repo.save(e_nb)

        claim = _make_claim(evidence_ids=["e-sup", "e-nb"])
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        assert report.supporting_count == 1
        assert report.contradicting_count == 0

    async def test_unjudged_evidence_excluded(self, repo):
        """Evidence with support_judgment None should be ignored."""
        obj = _make_objective()
        await repo.save(obj)

        e_sup = _make_evidence(support_judgment="supports", entity_id="e-sup2")
        e_none = _make_evidence(support_judgment=None, entity_id="e-none")
        await repo.save(e_sup)
        await repo.save(e_none)

        claim = _make_claim(evidence_ids=["e-sup2", "e-none"])
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        assert report.supporting_count == 1
        assert report.contradicting_count == 0


# =========================================================================
# Claim filtering tests
# =========================================================================


class TestPosteriorClaimFiltering:
    """Tests that abandoned claims are excluded."""

    async def test_abandoned_claims_excluded(self, repo):
        """Abandoned claims and their evidence should not count."""
        obj = _make_objective()
        await repo.save(obj)

        e_active = _make_evidence(support_judgment="supports", entity_id="e-act")
        e_abandoned = _make_evidence(support_judgment="contradicts", entity_id="e-abn")
        await repo.save(e_active)
        await repo.save(e_abandoned)

        active_claim = _make_claim(evidence_ids=["e-act"])
        abandoned_claim = _make_claim(evidence_ids=["e-abn"], abandoned=True)
        await repo.save(active_claim)
        await repo.save(abandoned_claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        # Only evidence from the active claim should count
        assert report.supporting_count == 1
        assert report.contradicting_count == 0


# =========================================================================
# Question type eligibility tests
# =========================================================================


class TestPosteriorQuestionTypeEligibility:
    """Tests that only eligible question types produce a PosteriorReport.

    Eligibility has two paths:
      (a) question_type is verificatory or predictive
      (b) the objective is in seed_claim mode (claim_to_verify is set)

    The non-seed cases below pin path (a)'s narrowness; the
    TestPosteriorSeedClaimMode class below pins path (b).
    """

    async def test_explanatory_without_seed_returns_none(self, repo):
        """Explanatory parent without seed_claim mode → None preserved."""
        obj = _make_objective(question_type="explanatory")
        await repo.save(obj)
        result = await compute_posterior(repo, OBJ_ID)
        assert result is None

    async def test_exploratory_without_seed_returns_none(self, repo):
        """Exploratory parent without seed_claim mode → None preserved."""
        obj = _make_objective(question_type="exploratory")
        await repo.save(obj)
        result = await compute_posterior(repo, OBJ_ID)
        assert result is None

    async def test_verificatory_returns_report(self, repo):
        """Verificatory questions should return a PosteriorReport."""
        obj = _make_objective(question_type="verificatory")
        await repo.save(obj)

        claim = _make_claim()
        await repo.save(claim)

        result = await compute_posterior(repo, OBJ_ID)
        assert isinstance(result, PosteriorReport)
        assert result.question_type == "verificatory"

    async def test_comparative_returns_none(self, repo):
        """Comparative questions have three outcomes (A better, B better, equivalent) — not binary."""
        obj = _make_objective(question_type="comparative")
        await repo.save(obj)

        result = await compute_posterior(repo, OBJ_ID)
        assert result is None

    async def test_predictive_returns_report(self, repo):
        """Predictive questions should return a PosteriorReport."""
        obj = _make_objective(question_type="predictive")
        await repo.save(obj)

        claim = _make_claim()
        await repo.save(claim)

        result = await compute_posterior(repo, OBJ_ID)
        assert isinstance(result, PosteriorReport)
        assert result.question_type == "predictive"


# =========================================================================
# Seed-claim mode eligibility (the smoke_v12_decompose case 54 fix)
# =========================================================================


class TestPosteriorSeedClaimMode:
    """When an objective is in seed_claim mode (claim_to_verify is set),
    compute_posterior must compute a posterior regardless of question_type.

    Bug context: smoke_v12_decompose case 54 was misclassified as
    explanatory; all 7 spawned children inherited that question_type,
    silently dropped their integration verdicts via the eligibility
    filter, and the harness saw posterior=None for every child despite
    valid IBE outcomes in the DB. The fix lets seed-claim children
    bypass the question_type filter — the seed claim's verification is
    binary by construction regardless of how the parent was classified.
    """

    async def test_explanatory_parent_seed_claim_child_returns_report(self, repo):
        """The headline case: explanatory question_type with claim_to_verify
        set must produce a posterior, not None."""
        obj = Objective(
            entity_id=OBJ_ID,
            objective_id=OBJ_ID,
            description="seed claim text",
            question_type="explanatory",
            claim_to_verify="seed claim text",
        )
        await repo.save(obj)
        # An IBE-style verdict on the underlying claim drives the posterior.
        claim = _make_claim()
        claim.integrated_assessment = "contradicts"
        claim.integrated_confidence = 0.75
        await repo.save(claim)

        result = await compute_posterior(repo, OBJ_ID)
        assert isinstance(result, PosteriorReport)
        # contradicts at 0.75 confidence → posterior = 0.5 - 0.75/2 = 0.125.
        assert result.posterior == pytest.approx(0.125)
        assert result.integration_verdict == "contradicts"

    async def test_comparative_parent_seed_claim_child_returns_report(self, repo):
        """A comparative parent decomposed into binary seed claims:
        each child verifies a specific claim, so seed-claim mode opens
        up posterior computation even though comparative parents
        themselves don't admit a P(Y)."""
        obj = Objective(
            entity_id=OBJ_ID,
            objective_id=OBJ_ID,
            description="seed claim",
            question_type="comparative",
            claim_to_verify="seed claim",
        )
        await repo.save(obj)
        claim = _make_claim()
        claim.integrated_assessment = "supports"
        claim.integrated_confidence = 0.6
        await repo.save(claim)

        result = await compute_posterior(repo, OBJ_ID)
        assert isinstance(result, PosteriorReport)
        # supports at 0.6 → posterior = 0.5 + 0.6/2 = 0.8.
        assert result.posterior == pytest.approx(0.8)

    async def test_no_question_type_with_seed_claim_returns_report(self, repo):
        """Edge: question_type is None (rare — classifier didn't run) but
        claim_to_verify is set. The seed-claim escape covers this; the
        report's question_type defaults to 'verificatory'."""
        obj = Objective(
            entity_id=OBJ_ID,
            objective_id=OBJ_ID,
            description="seed claim",
            question_type=None,
            claim_to_verify="seed claim",
        )
        await repo.save(obj)
        result = await compute_posterior(repo, OBJ_ID)
        assert isinstance(result, PosteriorReport)
        assert result.question_type == "verificatory"

    async def test_explanatory_without_seed_still_returns_none(self, repo):
        """Sanity: the seed-claim escape doesn't accidentally open the
        gate for non-seed explanatory parents — posterior P(Y) for a
        genuinely explanatory question is still semantically fuzzy and
        we keep returning None."""
        obj = Objective(
            entity_id=OBJ_ID,
            objective_id=OBJ_ID,
            description="why does X happen",
            question_type="explanatory",
            claim_to_verify=None,
        )
        await repo.save(obj)
        result = await compute_posterior(repo, OBJ_ID)
        assert result is None

    async def test_compute_posterior_honors_AND_rule(self, repo):
        """Multi-seed-claim with AND combination_rule: compute_posterior
        delegates to combine_claim_verdicts and returns the weakest-link
        bound (min over per-claim posteriors), not the confidence-weighted
        average. Matches what the user actually asked for via
        decomposition."""
        obj = Objective(
            entity_id=OBJ_ID,
            objective_id=OBJ_ID,
            description="parent",
            question_type="explanatory",
            decomposition={
                "sub_investigations": [
                    {"id": "A", "seed_claim": "alpha", "rationale": "ra"},
                    {"id": "B", "seed_claim": "beta", "rationale": "rb"},
                ],
                "combination_rule": "AND",
                "rationale": "both must hold",
            },
            combination_rule="AND",
        )
        await repo.save(obj)
        # Two claims with sub_investigation_id matching the decomposition.
        from andamentum.epistemic.entities.claim import ClaimStage as _CS

        claim_a = Claim(
            objective_id=OBJ_ID,
            statement="A",
            scope="ra",
            stage=_CS.SUPPORTED,
            sub_investigation_id="A",
            integrated_assessment="supports",
            integrated_confidence=0.8,
        )
        await repo.save(claim_a)
        claim_b = Claim(
            objective_id=OBJ_ID,
            statement="B",
            scope="rb",
            stage=_CS.SUPPORTED,
            sub_investigation_id="B",
            integrated_assessment="contradicts",
            integrated_confidence=0.7,
        )
        await repo.save(claim_b)

        result = await compute_posterior(repo, OBJ_ID)
        assert result is not None
        # AND over [supports@0.8 → 0.9, contradicts@0.7 → 0.15]
        # min = 0.15 → contradicts. Pre-fix: confidence-weighted average
        # would yield (0.9*0.8 + 0.15*0.7) / 1.5 ≈ 0.55 (insufficient).
        assert result.posterior == pytest.approx(0.15)
        assert result.integration_verdict == "contradicts"
        assert result.mode == "rule_aware_and"

    async def test_compute_posterior_honors_OR_rule(self, repo):
        """OR combination: max over per-claim posteriors."""
        obj = Objective(
            entity_id=OBJ_ID,
            objective_id=OBJ_ID,
            description="parent",
            question_type="verificatory",
            decomposition={
                "sub_investigations": [
                    {"id": "A", "seed_claim": "alpha", "rationale": "ra"},
                    {"id": "B", "seed_claim": "beta", "rationale": "rb"},
                ],
                "combination_rule": "OR",
                "rationale": "either suffices",
            },
            combination_rule="OR",
        )
        await repo.save(obj)
        from andamentum.epistemic.entities.claim import ClaimStage as _CS

        for sub_id, verdict, conf in (
            ("A", "contradicts", 0.5),
            ("B", "supports", 0.85),
        ):
            c = Claim(
                objective_id=OBJ_ID,
                statement=sub_id,
                scope="x",
                stage=_CS.SUPPORTED,
                sub_investigation_id=sub_id,
                integrated_assessment=verdict,
                integrated_confidence=conf,
            )
            await repo.save(c)

        result = await compute_posterior(repo, OBJ_ID)
        assert result is not None
        # OR over [contradicts@0.5 → 0.25, supports@0.85 → 0.925]
        # max = 0.925 → supports.
        assert result.posterior == pytest.approx(0.925)
        assert result.integration_verdict == "supports"
        assert result.mode == "rule_aware_or"

    async def test_no_combination_rule_uses_weighted_average(self, repo):
        """Sanity: when no combination_rule is set (open-research /
        ProposeClaims path), the rule-blind confidence-weighted average
        path remains in place."""
        obj = _make_objective(question_type="verificatory")
        await repo.save(obj)
        claim = _make_claim()
        claim.integrated_assessment = "supports"
        claim.integrated_confidence = 0.8
        await repo.save(claim)

        result = await compute_posterior(repo, OBJ_ID)
        assert result is not None
        assert result.mode == "abductive"

    async def test_explanatory_parent_with_decomposition_returns_report(self, repo):
        """Multi-seed-claim mode: parent classified explanatory but has
        decomposition with sub-investigations. is_verification_task()
        is True (decomposition is set with non-empty sub_investigations);
        eligibility should pass. This is the case-54 silent-loss bug
        replicated for multi-seed-claim — pre-fix, eligibility only
        checked claim_to_verify and would return None here."""
        obj = Objective(
            entity_id=OBJ_ID,
            objective_id=OBJ_ID,
            description="parent",
            question_type="explanatory",
            decomposition={
                "sub_investigations": [
                    {"id": "A", "seed_claim": "alpha", "rationale": "ra"},
                    {"id": "B", "seed_claim": "beta", "rationale": "rb"},
                ],
                "combination_rule": "AND",
                "rationale": "both must hold",
            },
        )
        await repo.save(obj)
        # A claim with an integration verdict so the report has content.
        claim = _make_claim()
        claim.integrated_assessment = "contradicts"
        claim.integrated_confidence = 0.75
        await repo.save(claim)

        result = await compute_posterior(repo, OBJ_ID)
        assert isinstance(result, PosteriorReport)
        # contradicts at 0.75 → posterior = 0.5 - 0.75/2 = 0.125.
        assert result.posterior == pytest.approx(0.125)
        assert result.integration_verdict == "contradicts"


# =========================================================================
# Multi-claim aggregation tests
# =========================================================================


class TestPosteriorMultiClaimAggregation:
    """Tests that evidence is aggregated across multiple active claims."""

    async def test_evidence_aggregated_across_claims(self, repo):
        """Evidence from multiple active claims should be summed
        together in the diagnostic counts. The headline posterior
        suspends because no claim has integrated_assessment."""
        obj = _make_objective()
        await repo.save(obj)

        e1 = _make_evidence(support_judgment="supports", entity_id="e-m1")
        e2 = _make_evidence(support_judgment="supports", entity_id="e-m2")
        e3 = _make_evidence(support_judgment="contradicts", entity_id="e-m3")
        await repo.save(e1)
        await repo.save(e2)
        await repo.save(e3)

        claim1 = _make_claim(evidence_ids=["e-m1", "e-m2"])
        claim2 = _make_claim(evidence_ids=["e-m3"])
        await repo.save(claim1)
        await repo.save(claim2)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        # 2 supporting from claim1 + 1 contradicting from claim2 — the
        # diagnostic counts aggregate across claims correctly.
        assert report.supporting_count == 2
        assert report.contradicting_count == 1
        # Diagnostic counting_posterior reflects the math:
        # 1 / (1 + exp(-1)) ≈ 0.731.
        assert report.counting_posterior == pytest.approx(
            1.0 / (1.0 + math.exp(-1)), abs=1e-4
        )
        # Headline posterior suspends — no IBE certification.
        assert report.posterior == 0.5


# =========================================================================
# Report structure tests
# =========================================================================


class TestPosteriorReportStructure:
    """Tests that PosteriorReport fields are correctly populated."""

    async def test_report_fields(self, repo):
        """PosteriorReport should have all expected fields correctly populated."""
        obj = _make_objective(question_type="verificatory")
        await repo.save(obj)

        e1 = _make_evidence(support_judgment="supports", entity_id="e-rf1")
        e2 = _make_evidence(support_judgment="contradicts", entity_id="e-rf2")
        await repo.save(e1)
        await repo.save(e2)

        claim = _make_claim(evidence_ids=["e-rf1", "e-rf2"])
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        assert isinstance(report, PosteriorReport)
        assert report.objective_id == OBJ_ID
        assert report.question_type == "verificatory"
        assert 0.0 <= report.posterior <= 1.0
        assert isinstance(report.log_odds, int)
        # supporting_count and contradicting_count are weighted (1 + log(cluster_size))
        # so they are floats, not ints.
        assert isinstance(report.supporting_count, float)
        assert isinstance(report.contradicting_count, float)
        assert isinstance(report.explanation, str)
        assert len(report.explanation) > 0

    async def test_sigmoid_calculation(self, repo):
        """Verify the sigmoid transform: posterior = 1/(1+exp(-log_odds))
        when an integrated_assessment is present (so counts can drive
        the diagnostic counting_posterior).

        After 2026-05-05's no-certified-verdict gate, counting alone
        does NOT drive the headline posterior — the gate suspends to
        0.5 if no claim has an integrated_assessment. To test the
        sigmoid math we set an integrated_assessment so the gate
        passes; the diagnostic counting_posterior then exposes the
        sigmoid value for inspection."""
        obj = _make_objective()
        await repo.save(obj)

        # 2 supporting, 0 contradicting → log_odds = 2
        e1 = _make_evidence(support_judgment="supports", entity_id="e-sig1")
        e2 = _make_evidence(support_judgment="supports", entity_id="e-sig2")
        await repo.save(e1)
        await repo.save(e2)

        claim = _make_claim(evidence_ids=["e-sig1", "e-sig2"])
        # Set IA so the no-certified-verdict gate doesn't fire.
        claim.integrated_assessment = "supports"
        claim.integrated_confidence = 0.4
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        expected = 1.0 / (1.0 + math.exp(-2))
        assert report.counting_posterior == pytest.approx(expected, abs=1e-4)


# =========================================================================
# Integration synthesis tests
# =========================================================================


class TestPosteriorIntegrationSynthesis:
    """Tests that the abduction agent's verdict drives the posterior.

    When `claim.integrated_assessment` is set, the posterior follows it
    directly: supports → 0.5 + c/2, contradicts → 0.5 - c/2, insufficient
    → 0.5. The per-item counting signal is reported as a diagnostic but
    does not enter the posterior. Counting only drives the answer when
    no claim received an integration verdict (counting_fallback mode).
    """

    async def test_insufficient_yields_neutral_posterior(self, repo):
        """Integration 'insufficient' → posterior 0.5 regardless of counts."""
        obj = _make_objective()
        await repo.save(obj)

        e1 = _make_evidence(support_judgment="supports", entity_id="e-i1")
        e2 = _make_evidence(support_judgment="supports", entity_id="e-i2")
        e3 = _make_evidence(support_judgment="supports", entity_id="e-i3")
        await repo.save(e1)
        await repo.save(e2)
        await repo.save(e3)

        claim = _make_claim(evidence_ids=["e-i1", "e-i2", "e-i3"])
        claim.integrated_assessment = "insufficient"
        claim.integrated_confidence = 0.5
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        assert report.mode == "abductive"
        # Even with 3 supporting items, abduction's "insufficient" wins.
        assert report.posterior == pytest.approx(0.5, abs=0.01)
        # Counting is still surfaced as a diagnostic.
        assert report.supporting_count == pytest.approx(3.0, abs=0.01)
        assert report.integration_verdict == "insufficient"

    async def test_supports_drives_posterior_up(self, repo):
        """Integration 'supports' at confidence c → posterior 0.5 + c/2."""
        obj = _make_objective()
        await repo.save(obj)

        e1 = _make_evidence(support_judgment="supports", entity_id="e-b1")
        e2 = _make_evidence(support_judgment="contradicts", entity_id="e-b2")
        await repo.save(e1)
        await repo.save(e2)

        claim = _make_claim(evidence_ids=["e-b1", "e-b2"])
        claim.integrated_assessment = "supports"
        claim.integrated_confidence = 0.8
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        assert report.mode == "abductive"
        # Posterior = 0.5 + 0.8/2 = 0.9, regardless of the balanced counts.
        assert report.posterior == pytest.approx(0.9, abs=0.01)
        # Counting is reported as a diagnostic but does not enter the result.
        assert report.counting_posterior == pytest.approx(0.5, abs=0.01)
        assert report.integration_verdict == "supports"

    async def test_contradicts_drives_posterior_down(self, repo):
        """Integration 'contradicts' overrides a counting-supports signal."""
        obj = _make_objective()
        await repo.save(obj)

        e1 = _make_evidence(support_judgment="supports", entity_id="e-d1")
        e2 = _make_evidence(support_judgment="supports", entity_id="e-d2")
        e3 = _make_evidence(support_judgment="supports", entity_id="e-d3")
        await repo.save(e1)
        await repo.save(e2)
        await repo.save(e3)

        claim = _make_claim(evidence_ids=["e-d1", "e-d2", "e-d3"])
        claim.integrated_assessment = "contradicts"
        claim.integrated_confidence = 0.9
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        assert report.mode == "abductive"
        # Counting points up (3 supports), abduction says contradicts at 0.9.
        # Posterior follows abduction: 0.5 - 0.9/2 = 0.05
        assert report.posterior == pytest.approx(0.05, abs=0.01)
        # Counting diagnostic is high — abduction overrode it.
        counting_p = 1.0 / (1.0 + math.exp(-3))
        assert report.counting_posterior == pytest.approx(counting_p, abs=1e-4)
        # Disagreement note appears so the reader can see the override.
        assert "disagree" in report.explanation.lower()

    async def test_counting_does_not_override_abduction(self, repo):
        """Many supporting items + abduction=contradicts → posterior follows abduction."""
        obj = _make_objective()
        await repo.save(obj)

        eids = []
        for i in range(20):
            eid = f"e-w{i}"
            eids.append(eid)
            e = _make_evidence(support_judgment="supports", entity_id=eid)
            await repo.save(e)

        claim = _make_claim(evidence_ids=eids)
        claim.integrated_assessment = "contradicts"
        claim.integrated_confidence = 0.9
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        # Counting alone would be ~1.0 (logistic(20)). Abduction overrides.
        assert report.posterior == pytest.approx(0.05, abs=0.01)
        assert report.mode == "abductive"

    async def test_no_evidence_integration_only(self, repo):
        """With no per-item evidence, abduction still fully determines posterior."""
        obj = _make_objective()
        await repo.save(obj)

        claim = _make_claim(evidence_ids=[])
        claim.integrated_assessment = "supports"
        claim.integrated_confidence = 0.8
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        assert report.mode == "abductive"
        # supports at 0.8 → 0.9
        assert report.posterior == pytest.approx(0.9, abs=0.01)

    async def test_no_integration_suspends_via_no_certified_verdict_gate(self, repo):
        """When no claim has an integration verdict, the
        no-certified-verdict gate (added 2026-05-05) suspends the
        posterior at 0.5 rather than falling back to counting.

        Pre-fix: counting drove the posterior on uncertified claims,
        which produced writer-vs-aggregator disagreements (the writer
        was simultaneously routed to SynthesizeInsufficient under the
        same condition). Now both signals derive from the same upstream
        condition: no IBE certification → no directional output. The
        diagnostic counting_posterior is still exposed so a reader can
        see WHAT counting WOULD have said, but it does not drive the
        posterior."""
        obj = _make_objective()
        await repo.save(obj)

        e1 = _make_evidence(support_judgment="supports", entity_id="e-fb1")
        e2 = _make_evidence(support_judgment="supports", entity_id="e-fb2")
        await repo.save(e1)
        await repo.save(e2)

        # Claim has NO integrated_assessment — abduction never ran.
        claim = _make_claim(evidence_ids=["e-fb1", "e-fb2"])
        claim.integrated_assessment = None
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        assert report.posterior == 0.5
        assert report.terminal_state == "oscillation_detected"
        assert report.mode == "counting_only"
        # Diagnostic counting still exposed for reader inspection.
        assert report.counting_posterior > 0.5  # 2 supports, 0 contradicts
        assert report.integration_verdict is None
        assert "No certified verdict" in report.explanation

    async def test_report_includes_new_fields(self, repo):
        """PosteriorReport surfaces all fields used by downstream consumers."""
        obj = _make_objective()
        await repo.save(obj)

        e1 = _make_evidence(support_judgment="supports", entity_id="e-nf1")
        await repo.save(e1)

        claim = _make_claim(evidence_ids=["e-nf1"])
        claim.integrated_assessment = "supports"
        claim.integrated_confidence = 0.7
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        assert hasattr(report, "counting_posterior")
        assert hasattr(report, "integration_verdict")
        assert hasattr(report, "integration_confidence")
        assert hasattr(report, "mode")
        assert report.integration_verdict == "supports"
        assert report.integration_confidence == pytest.approx(0.7, abs=0.01)
        assert report.mode == "abductive"
