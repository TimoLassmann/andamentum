"""Tests for Peirce cycling triggered by uncertainty resolution.

After Clean Task 2 (move flow control from operations to graph nodes),
the ResolveUncertaintyOperation no longer resets scrutiny_verdict on
affected claims. That responsibility moved to the ResolveUncertainties
graph node, which populates state.claims_needing_rescrutiny instead.

These tests verify that the operation does NOT modify claims (P1/P5),
and that non-blocking / already-resolved uncertainties remain unchanged.
"""

import pytest

from ..entities import Claim, ClaimStage, Objective, Uncertainty, UncertaintyType
from ..operations.uncertainty import ResolveUncertaintyOperation
from ..patterns import WorkItem


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _make_objective(obj_id: str = "obj-1") -> Objective:
    return Objective(
        entity_id=obj_id,
        objective_id=obj_id,
        description="Test objective",
        phase="claims_done",
    )


def _make_claim(
    obj_id: str = "obj-1",
    cid: str = "cl-1",
    stage: ClaimStage = ClaimStage.SUPPORTED,
    scrutiny_verdict: str | None = "pass",
    investigation_count: int = 0,
    abandoned: bool = False,
) -> Claim:
    return Claim(
        entity_id=cid,
        objective_id=obj_id,
        statement="Test claim",
        evidence_ids=["ev-1"],
        evidence_count=1,
        stage=stage,
        scrutiny_verdict=scrutiny_verdict,
        investigation_count=investigation_count,
        abandoned=abandoned,
    )


def _make_uncertainty(
    obj_id: str = "obj-1",
    uid: str = "unc-1",
    uncertainty_type: UncertaintyType = UncertaintyType.CONTRADICTION,
    affected_claim_ids: list[str] | None = None,
) -> Uncertainty:
    return Uncertainty(
        entity_id=uid,
        objective_id=obj_id,
        uncertainty_type=uncertainty_type,
        description="Test uncertainty",
        affected_claim_ids=affected_claim_ids or ["cl-1"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tests: Blocking uncertainty resolution triggers Peirce cycling
# ══════════════════════════════════════════════════════════════════════════════


class TestBlockingResolutionNoScrutinyReset:
    """Operation no longer resets scrutiny — graph node handles this (P1/P5)."""

    @pytest.mark.asyncio
    async def test_contradiction_resolution_does_not_reset_scrutiny(self, repo, fake_runner):
        """Resolving a CONTRADICTION no longer resets scrutiny_verdict (moved to graph)."""
        await repo.save(_make_objective())
        claim = _make_claim(scrutiny_verdict="pass")
        unc = _make_uncertainty(uncertainty_type=UncertaintyType.CONTRADICTION)
        await repo.save(claim)
        await repo.save(unc)

        op = ResolveUncertaintyOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test-model"
        )
        work = WorkItem(entity_id="unc-1", entity_type="uncertainty", operation="resolve_uncertainty")
        result = await op.execute(work)

        assert result.success
        updated_claim = await repo.get("claim", "cl-1")
        assert updated_claim.scrutiny_verdict == "pass"  # unchanged by operation

    @pytest.mark.asyncio
    async def test_strong_counterevidence_does_not_reset_scrutiny(self, repo, fake_runner):
        """Resolving STRONG_COUNTEREVIDENCE no longer resets scrutiny (moved to graph)."""
        await repo.save(_make_objective())
        claim = _make_claim(scrutiny_verdict="pass")
        unc = _make_uncertainty(uncertainty_type=UncertaintyType.STRONG_COUNTEREVIDENCE)
        await repo.save(claim)
        await repo.save(unc)

        op = ResolveUncertaintyOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test-model"
        )
        work = WorkItem(entity_id="unc-1", entity_type="uncertainty", operation="resolve_uncertainty")
        await op.execute(work)

        updated_claim = await repo.get("claim", "cl-1")
        assert updated_claim.scrutiny_verdict == "pass"  # unchanged by operation

    @pytest.mark.asyncio
    async def test_unknown_type_does_not_reset_scrutiny(self, repo, fake_runner):
        """Resolving UNKNOWN (blocking) no longer resets scrutiny (moved to graph)."""
        await repo.save(_make_objective())
        claim = _make_claim(scrutiny_verdict="needs_resolution")
        unc = _make_uncertainty(uncertainty_type=UncertaintyType.UNKNOWN)
        await repo.save(claim)
        await repo.save(unc)

        op = ResolveUncertaintyOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test-model"
        )
        work = WorkItem(entity_id="unc-1", entity_type="uncertainty", operation="resolve_uncertainty")
        await op.execute(work)

        updated_claim = await repo.get("claim", "cl-1")
        assert updated_claim.scrutiny_verdict == "needs_resolution"  # unchanged by operation

    @pytest.mark.asyncio
    async def test_multiple_affected_claims_unchanged(self, repo, fake_runner):
        """All affected claims remain unchanged — operation does not touch them."""
        await repo.save(_make_objective())
        claim1 = _make_claim(cid="cl-1", scrutiny_verdict="pass")
        claim2 = _make_claim(cid="cl-2", scrutiny_verdict="fail")
        unc = _make_uncertainty(affected_claim_ids=["cl-1", "cl-2"])
        await repo.save(claim1)
        await repo.save(claim2)
        await repo.save(unc)

        op = ResolveUncertaintyOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test-model"
        )
        work = WorkItem(entity_id="unc-1", entity_type="uncertainty", operation="resolve_uncertainty")
        await op.execute(work)

        cl1 = await repo.get("claim", "cl-1")
        cl2 = await repo.get("claim", "cl-2")
        assert cl1.scrutiny_verdict == "pass"  # unchanged
        assert cl2.scrutiny_verdict == "fail"  # unchanged


# ══════════════════════════════════════════════════════════════════════════════
# Tests: Non-blocking uncertainties do NOT trigger cycling
# ══════════════════════════════════════════════════════════════════════════════


class TestNonBlockingResolutionNoReset:
    """Non-blocking uncertainty resolution should not reset scrutiny."""

    @pytest.mark.asyncio
    async def test_evidence_gap_no_reset(self, repo, fake_runner):
        """EVIDENCE_GAP is non-blocking — should not reset scrutiny."""
        await repo.save(_make_objective())
        claim = _make_claim(scrutiny_verdict="pass")
        unc = _make_uncertainty(uncertainty_type=UncertaintyType.EVIDENCE_GAP)
        await repo.save(claim)
        await repo.save(unc)

        op = ResolveUncertaintyOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test-model"
        )
        work = WorkItem(entity_id="unc-1", entity_type="uncertainty", operation="resolve_uncertainty")
        await op.execute(work)

        updated_claim = await repo.get("claim", "cl-1")
        assert updated_claim.scrutiny_verdict == "pass"

    @pytest.mark.asyncio
    async def test_assumption_no_reset(self, repo, fake_runner):
        """ASSUMPTION is non-blocking — should not reset scrutiny."""
        await repo.save(_make_objective())
        claim = _make_claim(scrutiny_verdict="pass")
        unc = _make_uncertainty(uncertainty_type=UncertaintyType.ASSUMPTION)
        await repo.save(claim)
        await repo.save(unc)

        op = ResolveUncertaintyOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test-model"
        )
        work = WorkItem(entity_id="unc-1", entity_type="uncertainty", operation="resolve_uncertainty")
        await op.execute(work)

        updated_claim = await repo.get("claim", "cl-1")
        assert updated_claim.scrutiny_verdict == "pass"


# ══════════════════════════════════════════════════════════════════════════════
# Tests: Safety caps prevent runaway loops
# ══════════════════════════════════════════════════════════════════════════════


class TestSafetyCaps:
    """Verify that Peirce cycling safety caps work correctly."""

    @pytest.mark.asyncio
    async def test_abandoned_claims_not_touched(self, repo, fake_runner):
        """Operation does not touch claims at all — abandoned or otherwise."""
        await repo.save(_make_objective())
        claim = _make_claim(scrutiny_verdict="fail", abandoned=True)
        unc = _make_uncertainty(uncertainty_type=UncertaintyType.CONTRADICTION)
        await repo.save(claim)
        await repo.save(unc)

        op = ResolveUncertaintyOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test-model"
        )
        work = WorkItem(entity_id="unc-1", entity_type="uncertainty", operation="resolve_uncertainty")
        await op.execute(work)

        updated_claim = await repo.get("claim", "cl-1")
        # Operation no longer touches claims — verdict unchanged
        assert updated_claim.scrutiny_verdict == "fail"

    @pytest.mark.asyncio
    async def test_investigation_count_preserved(self, repo, fake_runner):
        """Operation does not modify claims — investigation_count stays unchanged."""
        await repo.save(_make_objective())
        claim = _make_claim(scrutiny_verdict="pass", investigation_count=2)
        unc = _make_uncertainty(uncertainty_type=UncertaintyType.CONTRADICTION)
        await repo.save(claim)
        await repo.save(unc)

        op = ResolveUncertaintyOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test-model"
        )
        work = WorkItem(entity_id="unc-1", entity_type="uncertainty", operation="resolve_uncertainty")
        await op.execute(work)

        updated_claim = await repo.get("claim", "cl-1")
        # Operation no longer touches claims — both fields unchanged
        assert updated_claim.scrutiny_verdict == "pass"
        assert updated_claim.investigation_count == 2

    @pytest.mark.asyncio
    async def test_missing_claim_does_not_crash(self, repo, fake_runner):
        """If an affected_claim_id references a missing claim, resolution still succeeds."""
        await repo.save(_make_objective())
        unc = _make_uncertainty(
            uncertainty_type=UncertaintyType.CONTRADICTION,
            affected_claim_ids=["nonexistent-claim"],
        )
        await repo.save(unc)

        op = ResolveUncertaintyOperation(
            repo=repo, agent_runner=fake_runner, embedding_model="test-model"
        )
        work = WorkItem(entity_id="unc-1", entity_type="uncertainty", operation="resolve_uncertainty")
        result = await op.execute(work)

        # Should succeed — missing claim is skipped via except
        assert result.success


# ══════════════════════════════════════════════════════════════════════════════
# Tests: Unresolvable uncertainties do NOT trigger cycling
# ══════════════════════════════════════════════════════════════════════════════


class TestUnresolvableNoReset:
    """When the agent says can_resolve=False, the uncertainty is resolved
    as 'Unresolvable' but the operation does not touch claims."""

    @pytest.mark.asyncio
    async def test_unresolvable_does_not_reset_scrutiny(self, repo):
        """Unresolvable blocking uncertainty does not reset scrutiny (moved to graph)."""
        from .conftest import FakeAgentRunner

        runner = FakeAgentRunner(overrides={
            "epistemic_resolve_uncertainty": {
                "can_resolve": False,
                "resolution": "",
                "remaining_concerns": [],
            }
        })

        await repo.save(_make_objective())
        claim = _make_claim(scrutiny_verdict="pass")
        unc = _make_uncertainty(uncertainty_type=UncertaintyType.CONTRADICTION)
        await repo.save(claim)
        await repo.save(unc)

        op = ResolveUncertaintyOperation(
            repo=repo, agent_runner=runner, embedding_model="test-model"
        )
        work = WorkItem(entity_id="unc-1", entity_type="uncertainty", operation="resolve_uncertainty")
        await op.execute(work)

        # Uncertainty is marked as "Unresolvable" — still a resolution
        updated_unc = await repo.get("uncertainty", "unc-1")
        assert updated_unc.resolution is not None  # "Unresolvable: acknowledged limitation"

        # Operation no longer touches claims — scrutiny unchanged
        updated_claim = await repo.get("claim", "cl-1")
        assert updated_claim.scrutiny_verdict == "pass"


class TestAlreadyResolvedNoop:
    """Already-resolved uncertainties should not re-trigger cycling."""

    @pytest.mark.asyncio
    async def test_already_resolved_is_noop(self, repo):
        """Pre-resolved uncertainty does not re-reset scrutiny."""
        await repo.save(_make_objective())
        claim = _make_claim(scrutiny_verdict="pass")
        unc = _make_uncertainty(uncertainty_type=UncertaintyType.CONTRADICTION)
        unc.resolve("Previously resolved")
        await repo.save(claim)
        await repo.save(unc)

        op = ResolveUncertaintyOperation(
            repo=repo, agent_runner=None, embedding_model="test-model"
        )
        work = WorkItem(entity_id="unc-1", entity_type="uncertainty", operation="resolve_uncertainty")
        result = await op.execute(work)

        assert result.success
        assert "Already resolved" in result.message

        # Scrutiny NOT reset — no new resolution happened
        updated_claim = await repo.get("claim", "cl-1")
        assert updated_claim.scrutiny_verdict == "pass"
