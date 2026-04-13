"""Tests for the investigation saturation check.

When investigation cycles stop producing useful information (all blocking
uncertainties resolved but verdict still needs_resolution), the claim should
be marked as saturated to prevent further wasteful investigation cycles.
"""

import pytest

from epistemic.storage import InMemoryStorageBackend
from epistemic.repository import EpistemicRepository
from epistemic.entities.objective import Objective
from epistemic.entities.claim import Claim
from epistemic.entities.evidence import Evidence
from epistemic.entities.uncertainty import Uncertainty, UncertaintyType
from epistemic.primitives import ClaimStage
from epistemic.operations import ScrutiniseClaimOperation
from epistemic.patterns import WorkItem, WORK_PATTERNS


# ---------------------------------------------------------------------------
# Helper: build a FakeAgentRunner that forces a specific scrutiny verdict
# ---------------------------------------------------------------------------


def _make_runner(verdict: str = "needs_resolution"):
    """Build a FakeAgentRunner override dict that produces the given verdict.

    The split-scrutiny path computes the verdict deterministically:
      - evidence_weight in (strong, moderate) -> pass
      - evidence_weight == conflicting          -> fail
      - anything else                           -> needs_resolution
    """
    if verdict == "pass":
        weight = "moderate"
    elif verdict == "fail":
        weight = "conflicting"
    else:
        weight = "weak"

    from tests.conftest import FakeAgentRunner

    return FakeAgentRunner(
        overrides={
            "epistemic_assess_evidence": {
                "claim_id": "c-1",
                "evidence_weight": weight,
                "confidence_estimate": 0.3,
                "justification": "test",
            },
        }
    )


# ---------------------------------------------------------------------------
# Saturation logic tests
# ---------------------------------------------------------------------------


class TestSaturationCheck:
    @pytest.fixture
    def backend(self):
        return InMemoryStorageBackend()

    @pytest.fixture
    def repo(self, backend):
        return EpistemicRepository(backend)

    @pytest.mark.asyncio
    async def test_saturated_when_no_unresolved_blocking(self, repo):
        """Claim should be saturated when investigation_count > 0,
        verdict=needs_resolution, and all blocking uncertainties are resolved."""
        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)

        claim = Claim(
            statement="test claim",
            objective_id=obj.entity_id,
            stage=ClaimStage.HYPOTHESIS,
            investigation_count=1,
        )
        ev = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="test evidence",
            quality_score=0.5,
        )
        await repo.save(ev)
        claim.evidence_ids = [ev.entity_id]
        claim.evidence_count = 1
        await repo.save(claim)

        # Create a resolved blocking uncertainty
        unc = Uncertainty(
            objective_id=obj.entity_id,
            uncertainty_type=UncertaintyType.UNKNOWN,
            description="Missing data",
            affected_claim_ids=[claim.entity_id],
            is_blocking=True,
        )
        unc.resolve("Unresolvable: acknowledged limitation")
        await repo.save(unc)

        runner = _make_runner(verdict="needs_resolution")
        op = ScrutiniseClaimOperation(repo, runner)
        work = WorkItem(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="scrutinise_claim",
        )
        await op.execute(work)

        updated = await repo.get("claim", claim.entity_id)
        assert updated.saturated is True

    @pytest.mark.asyncio
    async def test_not_saturated_when_blocking_unresolved(self, repo):
        """Claim should NOT be saturated if there are still unresolved
        blocking uncertainties."""
        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)

        claim = Claim(
            statement="test claim",
            objective_id=obj.entity_id,
            stage=ClaimStage.HYPOTHESIS,
            investigation_count=1,
        )
        ev = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="test evidence",
            quality_score=0.5,
        )
        await repo.save(ev)
        claim.evidence_ids = [ev.entity_id]
        claim.evidence_count = 1
        await repo.save(claim)

        # Create an UNRESOLVED blocking uncertainty
        unc = Uncertainty(
            objective_id=obj.entity_id,
            uncertainty_type=UncertaintyType.UNKNOWN,
            description="Missing data",
            affected_claim_ids=[claim.entity_id],
            is_blocking=True,
        )
        await repo.save(unc)

        runner = _make_runner(verdict="needs_resolution")
        op = ScrutiniseClaimOperation(repo, runner)
        work = WorkItem(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="scrutinise_claim",
        )
        await op.execute(work)

        updated = await repo.get("claim", claim.entity_id)
        assert updated.saturated is False

    @pytest.mark.asyncio
    async def test_not_saturated_on_first_scrutiny(self, repo):
        """First-time scrutiny (investigation_count=0) should never set saturated."""
        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)

        claim = Claim(
            statement="test claim",
            objective_id=obj.entity_id,
            stage=ClaimStage.HYPOTHESIS,
            investigation_count=0,
        )
        ev = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="test evidence",
            quality_score=0.5,
        )
        await repo.save(ev)
        claim.evidence_ids = [ev.entity_id]
        claim.evidence_count = 1
        await repo.save(claim)

        runner = _make_runner(verdict="needs_resolution")
        op = ScrutiniseClaimOperation(repo, runner)
        work = WorkItem(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="scrutinise_claim",
        )
        await op.execute(work)

        updated = await repo.get("claim", claim.entity_id)
        assert updated.saturated is False

    @pytest.mark.asyncio
    async def test_not_saturated_when_verdict_passes(self, repo):
        """If scrutiny passes, saturation check doesn't trigger."""
        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)

        claim = Claim(
            statement="test claim",
            objective_id=obj.entity_id,
            stage=ClaimStage.HYPOTHESIS,
            investigation_count=1,
        )
        ev = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="test evidence",
            quality_score=0.5,
        )
        await repo.save(ev)
        claim.evidence_ids = [ev.entity_id]
        claim.evidence_count = 1
        await repo.save(claim)

        runner = _make_runner(verdict="pass")
        op = ScrutiniseClaimOperation(repo, runner)
        work = WorkItem(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="scrutinise_claim",
        )
        await op.execute(work)

        updated = await repo.get("claim", claim.entity_id)
        assert updated.saturated is False

    @pytest.mark.asyncio
    async def test_not_saturated_with_only_nonblocking_unresolved(self, repo):
        """Non-blocking uncertainties (even if unresolved) should not prevent saturation."""
        obj = Objective(description="test", phase="claims_proposed")
        await repo.save(obj)

        claim = Claim(
            statement="test claim",
            objective_id=obj.entity_id,
            stage=ClaimStage.HYPOTHESIS,
            investigation_count=1,
        )
        ev = Evidence(
            objective_id=obj.entity_id,
            extracted=True,
            extracted_content="test evidence",
            quality_score=0.5,
        )
        await repo.save(ev)
        claim.evidence_ids = [ev.entity_id]
        claim.evidence_count = 1
        await repo.save(claim)

        # Create an unresolved NON-BLOCKING uncertainty
        unc = Uncertainty(
            objective_id=obj.entity_id,
            uncertainty_type=UncertaintyType.EVIDENCE_GAP,  # Non-blocking type
            description="Could use more evidence",
            affected_claim_ids=[claim.entity_id],
        )
        await repo.save(unc)

        runner = _make_runner(verdict="needs_resolution")
        op = ScrutiniseClaimOperation(repo, runner)
        work = WorkItem(
            entity_id=claim.entity_id,
            entity_type="claim",
            operation="scrutinise_claim",
        )
        await op.execute(work)

        updated = await repo.get("claim", claim.entity_id)
        # Non-blocking uncertainties don't count — claim should be saturated
        assert updated.saturated is True


# ---------------------------------------------------------------------------
# Pattern filter tests
# ---------------------------------------------------------------------------


class TestSaturationPatternFilter:
    def test_investigation_pattern_excludes_saturated(self):
        """Investigation patterns should have saturated=False filter."""
        investigate_patterns = [p for p in WORK_PATTERNS if p.operation == "investigate_claim"]
        assert len(investigate_patterns) >= 1
        for p in investigate_patterns:
            assert p.filters.get("saturated") is False, (
                f"Pattern '{p.description}' missing saturated=False filter"
            )

    def test_saturated_claim_does_not_match_investigation(self):
        """A saturated claim should not match investigation patterns."""
        investigate_patterns = [p for p in WORK_PATTERNS if p.operation == "investigate_claim"]
        claim = Claim(
            statement="test",
            objective_id="obj-1",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
            investigation_count=1,
            saturated=True,
        )
        for pattern in investigate_patterns:
            assert not pattern.matches(claim), (
                f"Saturated claim should not match pattern: {pattern.description}"
            )

    def test_unsaturated_claim_matches_investigation(self):
        """A non-saturated claim should still match investigation patterns."""
        needs_resolution_patterns = [
            p
            for p in WORK_PATTERNS
            if p.operation == "investigate_claim" and p.filters.get("scrutiny_verdict") == "needs_resolution"
        ]
        assert len(needs_resolution_patterns) >= 1
        claim = Claim(
            statement="test",
            objective_id="obj-1",
            stage=ClaimStage.HYPOTHESIS,
            scrutiny_verdict="needs_resolution",
            investigation_count=1,
            saturated=False,
        )
        for pattern in needs_resolution_patterns:
            assert pattern.matches(claim), (
                f"Non-saturated claim should match pattern: {pattern.description}"
            )
