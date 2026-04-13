"""Gate validators for epistemic stage transitions.

Validates claim promotions through stage gates and skepticism requirements:
- check_promotion: Verify claim can advance to proposed stage
- check_skepticism_invariant: Ensure skeptic review completed before promotion
- get_gate_requirements: Human-readable gate descriptions

Architecture: Layer 1 (framework-agnostic, no model calls)
"""

from typing import Dict, Any, Optional, Set

from ..primitives import WorkItemType, ClaimStage, Claim, WorkItem
from ..gates import check_promotion_gate, get_gate, GateResult, describe_all_gates
from .types import ValidationResult


class GateValidators:
    """Validates stage gate transitions and skepticism requirements.

    Enforces the Controlled Commitment invariant from epistemic philosophy:
    beliefs advance through explicit stages with gate requirements.
    """

    def check_promotion(
        self,
        claim: Claim,
        proposed_stage: ClaimStage,
        evidence_count: int,
        has_uncertainties: bool,
        has_skeptic_review: bool,
        justification_links: Dict[str, Any],
    ) -> GateResult:
        """Check if a claim promotion is allowed.

        Wrapper around gates.check_promotion_gate for convenience.

        Args:
            claim: The claim to promote
            proposed_stage: Target stage for promotion
            evidence_count: Number of evidence items linked to claim
            has_uncertainties: Whether claim has uncertainty annotations
            has_skeptic_review: Whether claim has been reviewed by skeptic
            justification_links: Additional justification data

        Returns:
            GateResult with pass/fail and reasoning
        """
        return check_promotion_gate(
            claim_id=claim.claim_id,
            current_stage=claim.stage,
            proposed_stage=proposed_stage,
            evidence_count=evidence_count,
            has_uncertainties_listed=has_uncertainties,
            has_skeptic_review=has_skeptic_review,
            justification_links=justification_links,
        )

    def check_skepticism_invariant(
        self, workitem: WorkItem, completed_workitem_ids: Set[str]
    ) -> ValidationResult:
        """Check that skepticism requirements are met for promotion.

        From philosophy: Skepticism is NOT optional and not personality-based.
        It is enforced procedurally via mandatory skeptic phases.

        Note: The actual scrutiny verdict check is done in the orchestrator's
        PROMOTE_CLAIM handler using get_claim_scrutiny_verdict(). This check
        is advisory - it verifies workitem dependencies are properly set up.

        Args:
            workitem: The workitem to check
            completed_workitem_ids: Set of workitem IDs that have completed

        Returns:
            ValidationResult with warnings if skepticism not properly set up
        """
        result = ValidationResult(valid=True)

        # Promotion requires prior scrutiny
        if workitem.operation_type == WorkItemType.PROMOTE_CLAIM:
            # Check if any dependency is in completed workitem IDs
            # Note: Dependencies are UUIDs, not strings like "scrutinise_claim"
            # The PROMOTE workitem should depend on the SCRUTINISE workitem that
            # was created during fan-out (see orchestrator._apply_state_transitions)
            completed_deps = [
                dep for dep in workitem.dependencies if dep in completed_workitem_ids
            ]
            if not completed_deps:
                result.add_warning(
                    "SKEPT_001",
                    "Promotion workitem has no completed dependencies - "
                    "scrutinise_claim should complete before promote_claim",
                    workitem_id=workitem.workitem_id,
                )

        return result

    def get_gate_requirements(
        self, from_stage: ClaimStage, to_stage: ClaimStage
    ) -> Optional[str]:
        """Get human-readable gate requirements for a transition.

        Args:
            from_stage: Current claim stage
            to_stage: Target claim stage

        Returns:
            Human-readable description of requirements, or None if no gate
        """
        gate = get_gate(from_stage, to_stage)
        if gate:
            return gate.describe()
        return None

    def describe_all_requirements(self) -> str:
        """Get description of all stage gate requirements.

        Returns:
            Multi-line string describing all gates
        """
        return describe_all_gates()
