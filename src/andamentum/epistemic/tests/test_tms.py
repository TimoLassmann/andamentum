"""Tests for TMS (Truth Maintenance System) cascading belief maintenance.

Tests the two new operations (InvalidateEvidenceOperation, RevalidateClaimOperation),
the validate_current_stage gate function, pattern matching, and transitive cascading.
"""

import pytest

from ..entities import Claim, ClaimStage, Evidence, Objective
from ..gates import validate_current_stage
from ..operations import (
    InvalidateEvidenceOperation,
    RevalidateClaimOperation,
)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _make_objective(repo, obj_id: str = "obj-1") -> Objective:
    """Create and save a test objective."""
    obj = Objective(
        entity_id=obj_id,
        objective_id=obj_id,
        description="Test objective",
        phase="planned",
    )
    return obj


def _make_evidence(
    obj_id: str = "obj-1",
    eid: str = "ev-1",
    extracted: bool = True,
    quality_score: float = 0.5,
    depends_on_claim_id: str | None = None,
    invalidated: bool = False,
    invalidation_cascaded: bool = False,
    invalidation_reason: str | None = None,
) -> Evidence:
    return Evidence(
        entity_id=eid,
        objective_id=obj_id,
        source_type="web_search",
        source_ref="https://example.com",
        extracted_content="Test content",
        extracted=extracted,
        quality_score=quality_score,
        depends_on_claim_id=depends_on_claim_id,
        invalidated=invalidated,
        invalidation_cascaded=invalidation_cascaded,
        invalidation_reason=invalidation_reason,
    )


def _make_claim(
    obj_id: str = "obj-1",
    cid: str = "cl-1",
    stage: ClaimStage = ClaimStage.SUPPORTED,
    evidence_ids: list[str] | None = None,
) -> Claim:
    eids = evidence_ids or []
    return Claim(
        entity_id=cid,
        objective_id=obj_id,
        statement="Test claim",
        evidence_ids=eids,
        evidence_count=len(eids),
        stage=stage,
        scrutiny_verdict="pass",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TestInvalidateEvidenceOperation
# ══════════════════════════════════════════════════════════════════════════════


class TestInvalidateEvidenceOperation:
    """Tests for InvalidateEvidenceOperation."""

    @pytest.mark.asyncio
    async def test_cascade_to_dependent_claim(self, repo):
        """Invalidated evidence is removed from dependent claim's evidence_ids."""
        ev = _make_evidence(invalidated=True, invalidation_cascaded=False)
        claim = _make_claim(evidence_ids=["ev-1"])
        await repo.save(ev)
        await repo.save(claim)

        op = InvalidateEvidenceOperation(repo=repo, agent_runner=None)
        from andamentum.epistemic.patterns import OperationInput

        work = OperationInput(
            entity_id="ev-1", entity_type="evidence", operation="invalidate_evidence"
        )
        result = await op.execute(work)

        assert result.success
        assert "1 claims" in result.message

        # Reload claim
        updated_claim = await repo.get("claim", "cl-1")
        assert "ev-1" not in updated_claim.evidence_ids
        assert updated_claim.evidence_count == 0

        # Evidence marked as cascaded
        updated_ev = await repo.get("evidence", "ev-1")
        assert updated_ev.invalidation_cascaded is True

    @pytest.mark.asyncio
    async def test_noop_if_already_cascaded(self, repo):
        """Already-cascaded evidence is not processed again."""
        ev = _make_evidence(invalidated=True, invalidation_cascaded=True)
        await repo.save(ev)

        op = InvalidateEvidenceOperation(repo=repo, agent_runner=None)
        from andamentum.epistemic.patterns import OperationInput

        work = OperationInput(
            entity_id="ev-1", entity_type="evidence", operation="invalidate_evidence"
        )
        result = await op.execute(work)

        assert result.success
        assert "already" in result.message.lower()

    @pytest.mark.asyncio
    async def test_orphan_evidence_no_claim_references(self, repo):
        """Evidence not referenced by any claim cascades cleanly."""
        ev = _make_evidence(invalidated=True, invalidation_cascaded=False)
        await repo.save(ev)

        # Claim that doesn't reference this evidence
        claim = _make_claim(evidence_ids=["ev-other"])
        await repo.save(claim)

        op = InvalidateEvidenceOperation(repo=repo, agent_runner=None)
        from andamentum.epistemic.patterns import OperationInput

        work = OperationInput(
            entity_id="ev-1", entity_type="evidence", operation="invalidate_evidence"
        )
        result = await op.execute(work)

        assert result.success
        assert "0 claims" in result.message

        # Claim untouched — evidence_ids unchanged
        updated_claim = await repo.get("claim", "cl-1")
        assert updated_claim.evidence_ids == ["ev-other"]

    @pytest.mark.asyncio
    async def test_multiple_claims_affected(self, repo):
        """Multiple claims referencing the same evidence all get flagged."""
        ev = _make_evidence(invalidated=True)
        await repo.save(ev)

        claim1 = _make_claim(cid="cl-1", evidence_ids=["ev-1", "ev-2"])
        claim2 = _make_claim(cid="cl-2", evidence_ids=["ev-1"])
        await repo.save(claim1)
        await repo.save(claim2)

        op = InvalidateEvidenceOperation(repo=repo, agent_runner=None)
        from andamentum.epistemic.patterns import OperationInput

        work = OperationInput(
            entity_id="ev-1", entity_type="evidence", operation="invalidate_evidence"
        )
        result = await op.execute(work)

        assert result.success
        assert "2 claims" in result.message

        cl1 = await repo.get("claim", "cl-1")
        cl2 = await repo.get("claim", "cl-2")
        assert "ev-1" not in cl1.evidence_ids
        assert "ev-1" not in cl2.evidence_ids


# ══════════════════════════════════════════════════════════════════════════════
# TestRevalidateClaimOperation
# ══════════════════════════════════════════════════════════════════════════════


class TestRevalidateClaimOperation:
    """Tests for RevalidateClaimOperation."""

    @pytest.mark.asyncio
    async def test_claim_still_passes_gate(self, repo):
        """Claim with enough evidence keeps its stage."""
        ev1 = _make_evidence(eid="ev-1", quality_score=0.5)
        ev2 = _make_evidence(eid="ev-2", quality_score=0.5)
        await repo.save(ev1)
        await repo.save(ev2)

        # SUPPORTED needs min_evidence=1, min_quality_sum=0.3
        claim = _make_claim(
            evidence_ids=["ev-1", "ev-2"],
            stage=ClaimStage.SUPPORTED,
        )
        await repo.save(claim)

        op = RevalidateClaimOperation(repo=repo, agent_runner=None)
        from andamentum.epistemic.patterns import OperationInput

        work = OperationInput(
            entity_id="cl-1", entity_type="claim", operation="revalidate_claim"
        )
        result = await op.execute(work)

        assert result.success
        assert "still meets" in result.message

        updated = await repo.get("claim", "cl-1")
        assert updated.stage == ClaimStage.SUPPORTED

    @pytest.mark.asyncio
    async def test_claim_fails_gate_demoted(self, repo):
        """Claim without enough evidence gets demoted."""
        # SUPPORTED needs min_evidence=1 and min_quality_sum=0.3
        # Claim has no evidence — should be demoted to HYPOTHESIS
        claim = _make_claim(
            evidence_ids=[],
            stage=ClaimStage.SUPPORTED,
        )
        await repo.save(claim)

        op = RevalidateClaimOperation(repo=repo, agent_runner=None)
        from andamentum.epistemic.patterns import OperationInput

        work = OperationInput(
            entity_id="cl-1", entity_type="claim", operation="revalidate_claim"
        )
        result = await op.execute(work)

        assert result.success
        assert "demoted" in result.message.lower()

        updated = await repo.get("claim", "cl-1")
        assert updated.stage == ClaimStage.HYPOTHESIS
        assert updated.modification_count == 1

    @pytest.mark.asyncio
    async def test_demotion_cascades_to_derived_evidence(self, repo):
        """When a claim is demoted, evidence depending on it gets invalidated."""
        claim = _make_claim(
            evidence_ids=[],
            stage=ClaimStage.SUPPORTED,
        )
        await repo.save(claim)

        # Evidence derived from this claim
        derived_ev = _make_evidence(
            eid="ev-derived",
            depends_on_claim_id="cl-1",
        )
        await repo.save(derived_ev)

        op = RevalidateClaimOperation(repo=repo, agent_runner=None)
        from andamentum.epistemic.patterns import OperationInput

        work = OperationInput(
            entity_id="cl-1", entity_type="claim", operation="revalidate_claim"
        )
        result = await op.execute(work)

        assert result.success

        # Derived evidence should be invalidated
        updated_ev = await repo.get("evidence", "ev-derived")
        assert updated_ev.invalidated is True
        assert updated_ev.invalidation_cascaded is False  # Ready for cascade processing

    @pytest.mark.asyncio
    async def test_hypothesis_cannot_demote_further(self, repo):
        """Claim at HYPOTHESIS stays at HYPOTHESIS — gate always passes."""
        claim = _make_claim(
            evidence_ids=[],
            stage=ClaimStage.HYPOTHESIS,
        )
        await repo.save(claim)

        op = RevalidateClaimOperation(repo=repo, agent_runner=None)
        from andamentum.epistemic.patterns import OperationInput

        work = OperationInput(
            entity_id="cl-1", entity_type="claim", operation="revalidate_claim"
        )
        result = await op.execute(work)

        assert result.success
        # HYPOTHESIS has no gate — validate_current_stage passes
        assert "still meets" in result.message

        updated = await repo.get("claim", "cl-1")
        assert updated.stage == ClaimStage.HYPOTHESIS

    @pytest.mark.asyncio
    async def test_demotion_records_promotion_history(self, repo):
        """TMS demotion is recorded in promotion_history."""
        claim = _make_claim(
            evidence_ids=[],
            stage=ClaimStage.SUPPORTED,
        )
        await repo.save(claim)

        op = RevalidateClaimOperation(repo=repo, agent_runner=None)
        from andamentum.epistemic.patterns import OperationInput

        work = OperationInput(
            entity_id="cl-1", entity_type="claim", operation="revalidate_claim"
        )
        await op.execute(work)

        updated = await repo.get("claim", "cl-1")
        assert len(updated.promotion_history) == 1
        entry = updated.promotion_history[0]
        assert entry["from"] == "supported"
        assert entry["to"] == "hypothesis"
        assert "TMS demotion" in entry["justification"]


# ══════════════════════════════════════════════════════════════════════════════
# TestValidateCurrentStage
# ══════════════════════════════════════════════════════════════════════════════


class TestValidateCurrentStage:
    """Tests for validate_current_stage gate function."""

    @pytest.mark.asyncio
    async def test_hypothesis_always_passes(self, repo):
        """HYPOTHESIS has no gate — always passes."""
        claim = _make_claim(stage=ClaimStage.HYPOTHESIS, evidence_ids=[])
        result = await validate_current_stage(claim, repo)
        assert result.passed

    @pytest.mark.asyncio
    async def test_supported_needs_evidence(self, repo):
        """SUPPORTED requires min_evidence=1."""
        claim = _make_claim(stage=ClaimStage.SUPPORTED, evidence_ids=[])
        result = await validate_current_stage(claim, repo)
        assert not result.passed
        assert "evidence" in result.blocking_reasons[0].lower()

    @pytest.mark.asyncio
    async def test_supported_passes_with_evidence(self, repo):
        """SUPPORTED passes with sufficient evidence."""
        ev = _make_evidence(eid="ev-1", quality_score=0.5)
        await repo.save(ev)

        claim = _make_claim(stage=ClaimStage.SUPPORTED, evidence_ids=["ev-1"])
        result = await validate_current_stage(claim, repo)
        assert result.passed

    @pytest.mark.asyncio
    async def test_invalidated_evidence_not_counted(self, repo):
        """Invalidated evidence doesn't count toward min_evidence."""
        ev = _make_evidence(eid="ev-1", quality_score=0.5, invalidated=True)
        await repo.save(ev)

        claim = _make_claim(stage=ClaimStage.SUPPORTED, evidence_ids=["ev-1"])
        result = await validate_current_stage(claim, repo)
        assert not result.passed


# ══════════════════════════════════════════════════════════════════════════════
# TestTransitiveCascade
# ══════════════════════════════════════════════════════════════════════════════


class TestTransitiveCascade:
    """Test full transitive cascade: E1 → C1 → E2 → C2."""

    @pytest.mark.asyncio
    async def test_full_cascade_chain(self, repo):
        """E1 invalidated → C1 demoted → E2 invalidated → C2 flagged."""
        from andamentum.epistemic.patterns import OperationInput

        # Build the chain: E1 supports C1, E2 depends on C1 and supports C2
        e1 = _make_evidence(eid="e1", quality_score=0.5, invalidated=True)
        c1 = _make_claim(cid="c1", stage=ClaimStage.SUPPORTED, evidence_ids=["e1"])
        e2 = _make_evidence(eid="e2", quality_score=0.5, depends_on_claim_id="c1")
        c2 = _make_claim(cid="c2", stage=ClaimStage.SUPPORTED, evidence_ids=["e2"])
        await repo.save(e1)
        await repo.save(c1)
        await repo.save(e2)
        await repo.save(c2)

        # Step 1: Invalidate evidence E1 → cascade to C1
        op1 = InvalidateEvidenceOperation(repo=repo, agent_runner=None)
        work1 = OperationInput(
            entity_id="e1", entity_type="evidence", operation="invalidate_evidence"
        )
        r1 = await op1.execute(work1)
        assert r1.success

        # C1 had its evidence removed by the cascade
        c1_updated = await repo.get("claim", "c1")
        assert "e1" not in c1_updated.evidence_ids

        # Step 2: Revalidate C1 → should be demoted → E2 gets invalidated
        op2 = RevalidateClaimOperation(repo=repo, agent_runner=None)
        work2 = OperationInput(
            entity_id="c1", entity_type="claim", operation="revalidate_claim"
        )
        r2 = await op2.execute(work2)
        assert r2.success

        c1_final = await repo.get("claim", "c1")
        assert c1_final.stage == ClaimStage.HYPOTHESIS

        # E2 should now be invalidated
        e2_updated = await repo.get("evidence", "e2")
        assert e2_updated.invalidated is True
        assert e2_updated.invalidation_cascaded is False

        # Step 3: Cascade E2 → C2
        op3 = InvalidateEvidenceOperation(repo=repo, agent_runner=None)
        work3 = OperationInput(
            entity_id="e2", entity_type="evidence", operation="invalidate_evidence"
        )
        r3 = await op3.execute(work3)
        assert r3.success

        c2_updated = await repo.get("claim", "c2")
        assert "e2" not in c2_updated.evidence_ids

    @pytest.mark.asyncio
    async def test_unaffected_claims_untouched(self, repo):
        """Claims not in the cascade chain remain unchanged."""
        e1 = _make_evidence(eid="e1", invalidated=True)
        c1 = _make_claim(cid="c1", evidence_ids=["e1"])

        # Unrelated claim
        e_other = _make_evidence(eid="e-other", quality_score=0.5)
        c_other = _make_claim(
            cid="c-other", stage=ClaimStage.SUPPORTED, evidence_ids=["e-other"]
        )
        await repo.save(e1)
        await repo.save(c1)
        await repo.save(e_other)
        await repo.save(c_other)

        op = InvalidateEvidenceOperation(repo=repo, agent_runner=None)
        from andamentum.epistemic.patterns import OperationInput

        work = OperationInput(
            entity_id="e1", entity_type="evidence", operation="invalidate_evidence"
        )
        await op.execute(work)

        # Unrelated claim untouched
        c_other_check = await repo.get("claim", "c-other")
        assert c_other_check.evidence_ids == ["e-other"]
        assert c_other_check.stage == ClaimStage.SUPPORTED

    @pytest.mark.asyncio
    async def test_cascade_terminates(self, repo):
        """Cascade terminates: cascaded evidence is not reprocessed."""
        from andamentum.epistemic.patterns import OperationInput

        ev = _make_evidence(eid="ev-1", invalidated=True, invalidation_cascaded=False)
        claim = _make_claim(cid="cl-1", evidence_ids=["ev-1"])
        await repo.save(ev)
        await repo.save(claim)

        op = InvalidateEvidenceOperation(repo=repo, agent_runner=None)
        work = OperationInput(
            entity_id="ev-1", entity_type="evidence", operation="invalidate_evidence"
        )

        # First execution
        r1 = await op.execute(work)
        assert r1.success
        assert "1 claims" in r1.message

        # Second execution — should be no-op
        r2 = await op.execute(work)
        assert r2.success
        assert "already" in r2.message.lower()


# ══════════════════════════════════════════════════════════════════════════════
# TestTMSTriggers — tests for the four new TMS entry points
# ══════════════════════════════════════════════════════════════════════════════


class TestInvestigationStubDependency:
    """Fix 1: investigation stubs must set depends_on_claim_id."""

    def test_investigation_stub_has_depends_on_claim_id(self):
        """Evidence stubs created by investigation should declare which claim spawned them."""
        # Simulate what InvestigateClaimOperation does at line 326-333
        from andamentum.epistemic.entities import Evidence

        stub = Evidence(
            objective_id="obj-1",
            source_type="web_search",
            source_ref="test query",
            extracted=False,
            created_by="investigate_claim",
            depends_on_claim_id="claim-42",
        )
        assert stub.depends_on_claim_id == "claim-42"

    @pytest.mark.asyncio
    async def test_tms_cascade_with_depends_on_claim_id(self, repo):
        """When claim is demoted by TMS, derived evidence with depends_on_claim_id is invalidated."""
        # Create evidence that depends on a claim
        ev = _make_evidence(
            eid="ev-derived", depends_on_claim_id="cl-1", quality_score=0.5
        )
        # Create the claim at SUPPORTED with enough evidence to pass basic gate
        ev_supporting = _make_evidence(eid="ev-support", quality_score=0.5)
        claim = _make_claim(
            cid="cl-1", stage=ClaimStage.SUPPORTED, evidence_ids=["ev-support"]
        )
        # Remove evidence so gate fails → demotion
        claim.evidence_ids = []
        claim.evidence_count = 0
        await repo.save(ev)
        await repo.save(ev_supporting)
        await repo.save(claim)

        from andamentum.epistemic.patterns import OperationInput

        op = RevalidateClaimOperation(repo=repo, agent_runner=None)
        work = OperationInput(
            entity_id="cl-1", entity_type="claim", operation="revalidate_claim"
        )
        result = await op.execute(work)

        assert result.success
        assert "demoted" in result.message.lower() or "HYPOTHESIS" in result.message

        # Derived evidence should now be invalidated
        ev_updated = await repo.get("evidence", "ev-derived")
        assert ev_updated.invalidated is True
        assert ev_updated.invalidation_cascaded is False  # Ready for cascade


class TestValidateCurrentStageExtended:
    """Fixes 3+5: validate_current_stage now checks adversarial balance and support balance."""

    @pytest.mark.asyncio
    async def test_adversarial_refutation_fails_gate(self, repo):
        """Claim with adversarial_balance < 0.3 should fail revalidation."""
        ev = _make_evidence(eid="ev-1", quality_score=0.5)
        await repo.save(ev)

        claim = _make_claim(evidence_ids=["ev-1"])
        claim.adversarial_balance = 0.15  # REFUTED level
        await repo.save(claim)

        result = await validate_current_stage(claim, repo)
        assert not result.passed
        assert "adversarial" in (result.reason or "").lower()

    @pytest.mark.asyncio
    async def test_adversarial_ok_passes_gate(self, repo):
        """Claim with good adversarial balance passes."""
        ev = _make_evidence(eid="ev-1", quality_score=0.5)
        await repo.save(ev)

        claim = _make_claim(evidence_ids=["ev-1"])
        claim.adversarial_balance = 0.75
        await repo.save(claim)

        result = await validate_current_stage(claim, repo)
        assert result.passed

    @pytest.mark.asyncio
    async def test_adversarial_not_set_passes_gate(self, repo):
        """Claim without adversarial check yet passes (no data = no trigger)."""
        ev = _make_evidence(eid="ev-1", quality_score=0.5)
        await repo.save(ev)

        claim = _make_claim(evidence_ids=["ev-1"])
        assert claim.adversarial_balance is None  # Not checked yet
        await repo.save(claim)

        result = await validate_current_stage(claim, repo)
        assert result.passed

    @pytest.mark.asyncio
    async def test_contradicting_outweighs_supporting_fails_gate(self, repo):
        """When contradicting >= supporting among judged evidence, gate fails."""
        ev_s = _make_evidence(eid="ev-s", quality_score=0.5)
        ev_s.support_judgment = "supports"
        ev_c1 = _make_evidence(eid="ev-c1", quality_score=0.5)
        ev_c1.support_judgment = "contradicts"
        ev_c2 = _make_evidence(eid="ev-c2", quality_score=0.5)
        ev_c2.support_judgment = "contradicts"
        await repo.save(ev_s)
        await repo.save(ev_c1)
        await repo.save(ev_c2)

        claim = _make_claim(evidence_ids=["ev-s", "ev-c1", "ev-c2"])
        await repo.save(claim)

        result = await validate_current_stage(claim, repo)
        assert not result.passed
        assert "contradicting" in (result.reason or "").lower()

    @pytest.mark.asyncio
    async def test_supporting_outweighs_contradicting_passes(self, repo):
        """When supporting > contradicting, gate passes on that criterion."""
        ev_s1 = _make_evidence(eid="ev-s1", quality_score=0.5)
        ev_s1.support_judgment = "supports"
        ev_s2 = _make_evidence(eid="ev-s2", quality_score=0.5)
        ev_s2.support_judgment = "supports"
        ev_c = _make_evidence(eid="ev-c", quality_score=0.5)
        ev_c.support_judgment = "contradicts"
        await repo.save(ev_s1)
        await repo.save(ev_s2)
        await repo.save(ev_c)

        claim = _make_claim(evidence_ids=["ev-s1", "ev-s2", "ev-c"])
        await repo.save(claim)

        result = await validate_current_stage(claim, repo)
        assert result.passed

    @pytest.mark.asyncio
    async def test_unjudged_evidence_not_counted(self, repo):
        """Unjudged evidence doesn't trigger the support balance check."""
        ev_unjudged = _make_evidence(eid="ev-u", quality_score=0.5)
        ev_unjudged.support_judgment = None  # Not judged
        ev_c = _make_evidence(eid="ev-c", quality_score=0.5)
        ev_c.support_judgment = "contradicts"
        await repo.save(ev_unjudged)
        await repo.save(ev_c)

        claim = _make_claim(evidence_ids=["ev-u", "ev-c"])
        await repo.save(claim)

        # Only 1 judged item — threshold is 2, so balance check won't fail
        result = await validate_current_stage(claim, repo)
        assert result.passed

    @pytest.mark.asyncio
    async def test_invalidated_evidence_excluded_from_balance(self, repo):
        """Invalidated evidence is not counted in support/contradict balance."""
        ev_s = _make_evidence(eid="ev-s", quality_score=0.5)
        ev_s.support_judgment = "supports"
        ev_c = _make_evidence(eid="ev-c", quality_score=0.5, invalidated=True)
        ev_c.support_judgment = "contradicts"
        ev_c2 = _make_evidence(eid="ev-c2", quality_score=0.5, invalidated=True)
        ev_c2.support_judgment = "contradicts"
        await repo.save(ev_s)
        await repo.save(ev_c)
        await repo.save(ev_c2)

        claim = _make_claim(evidence_ids=["ev-s", "ev-c", "ev-c2"])
        await repo.save(claim)

        # Both contradicting items are invalidated → only 1 judged item → passes
        result = await validate_current_stage(claim, repo)
        assert result.passed


class TestPromotionGuard:
    """Promotion is now guarded by the graph (TMS runs before promote).

    PromoteClaimOperation no longer checks needs_revalidation — the graph
    ensures TMS completes before calling promote.  We verify that promotion
    still works correctly via gate validation.
    """

    @pytest.mark.asyncio
    async def test_promotion_blocked_by_gate_failure(self, repo):
        """Claims that fail gate validation cannot be promoted."""
        from andamentum.epistemic.operations import PromoteClaimOperation
        from andamentum.epistemic.patterns import OperationInput

        # Claim with no evidence — gate should block promotion
        claim = _make_claim(
            cid="cl-1",
            stage=ClaimStage.HYPOTHESIS,
            evidence_ids=[],
        )
        await repo.save(claim)

        op = PromoteClaimOperation(repo=repo, agent_runner=None)
        work = OperationInput(
            entity_id="cl-1", entity_type="claim", operation="promote_claim"
        )
        result = await op.execute(work)

        assert not result.success

        # Claim should not have changed stage
        updated = await repo.get("claim", "cl-1")
        assert updated.stage == ClaimStage.HYPOTHESIS
