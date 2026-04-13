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
) -> Claim:
    return Claim(
        objective_id=objective_id,
        statement="Test claim for posterior",
        evidence_ids=evidence_ids or [],
        abandoned=abandoned,
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
        source_type="web",
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

    async def test_all_supporting_evidence_high_posterior(self, repo):
        """All supporting evidence should produce a high posterior (> 0.5)."""
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
        assert report.posterior > 0.5
        assert report.supporting_count == 3
        assert report.contradicting_count == 0
        assert report.log_odds == 3

    async def test_all_contradicting_evidence_low_posterior(self, repo):
        """All contradicting evidence should produce a low posterior (< 0.5)."""
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
        assert report.posterior < 0.5
        assert report.supporting_count == 0
        assert report.contradicting_count == 3
        assert report.log_odds == -3

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
        """Invalidated evidence should not count toward posterior."""
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
        # Only the valid supporting evidence should count
        assert report.supporting_count == 1
        assert report.contradicting_count == 0
        assert report.log_odds == 1

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
        """Evidence with support_judgment 'no_bearing' should be ignored."""
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
        assert report.log_odds == 1

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
    """Tests that only eligible question types produce a PosteriorReport."""

    async def test_explanatory_returns_none(self, repo):
        """Explanatory questions should return None."""
        obj = _make_objective(question_type="explanatory")
        await repo.save(obj)
        result = await compute_posterior(repo, OBJ_ID)
        assert result is None

    async def test_exploratory_returns_none(self, repo):
        """Exploratory questions should return None."""
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
# Multi-claim aggregation tests
# =========================================================================


class TestPosteriorMultiClaimAggregation:
    """Tests that evidence is aggregated across multiple active claims."""

    async def test_evidence_aggregated_across_claims(self, repo):
        """Evidence from multiple active claims should be summed together."""
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
        # 2 supporting from claim1 + 1 contradicting from claim2
        assert report.supporting_count == 2
        assert report.contradicting_count == 1
        assert report.log_odds == 1
        # 1 / (1 + exp(-1)) ≈ 0.731
        assert report.posterior == pytest.approx(1.0 / (1.0 + math.exp(-1)), abs=1e-4)


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
        assert isinstance(report.supporting_count, int)
        assert isinstance(report.contradicting_count, int)
        assert isinstance(report.explanation, str)
        assert len(report.explanation) > 0

    async def test_sigmoid_calculation(self, repo):
        """Verify the sigmoid transform: posterior = 1/(1+exp(-log_odds))."""
        obj = _make_objective()
        await repo.save(obj)

        # 2 supporting, 0 contradicting → log_odds = 2
        e1 = _make_evidence(support_judgment="supports", entity_id="e-sig1")
        e2 = _make_evidence(support_judgment="supports", entity_id="e-sig2")
        await repo.save(e1)
        await repo.save(e2)

        claim = _make_claim(evidence_ids=["e-sig1", "e-sig2"])
        await repo.save(claim)

        report = await compute_posterior(repo, OBJ_ID)
        assert report is not None
        expected = 1.0 / (1.0 + math.exp(-2))
        assert report.posterior == pytest.approx(expected, abs=1e-4)
        assert report.log_odds == 2
