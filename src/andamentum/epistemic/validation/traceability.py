"""Traceability validators for epistemic claims and artefacts.

Validates the Traceability invariant from epistemic philosophy:
- Every belief must be traceable to evidence and assumptions
- Artefacts may simplify but must never invent beliefs

Validators:
- validate_claim_traceability: Check claim has valid evidence/uncertainty links
- validate_artefact_traceability: Check artefact references valid snapshot claims
- validate_artefact_trace_completeness: Check all paragraphs have trace entries

Architecture: Layer 1 (framework-agnostic, no model calls)
"""

from typing import Dict, List, Set

from ..primitives import ClaimStage, Claim, Artefact, Snapshot
from .types import ValidationResult


class TraceabilityValidators:
    """Validates traceability requirements for claims and artefacts.

    Enforces the Traceability invariant: every belief must be traceable
    to evidence and assumptions.
    """

    def validate_claim_traceability(
        self,
        claim: Claim,
        available_evidence_ids: Set[str],
        available_uncertainty_ids: Set[str]
    ) -> ValidationResult:
        """Validate that a claim has proper traceability.

        From philosophy: Every belief must be traceable to evidence and assumptions.

        Args:
            claim: The claim to validate
            available_evidence_ids: Set of valid evidence IDs in the system
            available_uncertainty_ids: Set of valid uncertainty IDs in the system

        Returns:
            ValidationResult with errors if claim has broken links
        """
        result = ValidationResult(valid=True)

        # Check evidence links exist
        for eid in claim.evidence_ids:
            if eid not in available_evidence_ids:
                result.add_error("TRACE_001", f"Claim references unknown evidence: {eid}", claim_id=claim.claim_id)

        # Check uncertainty links exist
        for uid in claim.uncertainty_ids:
            if uid not in available_uncertainty_ids:
                result.add_error("TRACE_002", f"Claim references unknown uncertainty: {uid}", claim_id=claim.claim_id)

        # Higher stages require evidence
        if claim.stage != ClaimStage.HYPOTHESIS and not claim.evidence_ids:
            result.add_error(
                "TRACE_003",
                f"Claim at stage {claim.stage.value} must have evidence links",
                claim_id=claim.claim_id,
            )

        return result

    def validate_artefact_traceability(
        self,
        artefact: Artefact,
        snapshot: Snapshot,
        available_claim_ids: Set[str]
    ) -> ValidationResult:
        """Validate that an artefact is properly traceable to its snapshot.

        From philosophy: Artefacts may simplify but must never invent beliefs.

        Args:
            artefact: The artefact to validate
            snapshot: The snapshot the artefact is based on
            available_claim_ids: Set of all valid claim IDs in the system

        Returns:
            ValidationResult with errors if artefact has broken links
        """
        result = ValidationResult(valid=True)

        # Artefact must reference valid snapshot
        if artefact.snapshot_id != snapshot.snapshot_id:
            result.add_error(
                "ARTF_TRACE_001",
                f"Artefact snapshot_id mismatch: {artefact.snapshot_id} vs {snapshot.snapshot_id}",
            )

        # All claim refs in trace must be in snapshot
        snapshot_claims = set(snapshot.claim_ids)
        for para_id, claim_ids in artefact.trace.items():
            for cid in claim_ids:
                if cid not in snapshot_claims:
                    result.add_error(
                        "ARTF_TRACE_002",
                        f"Artefact references claim {cid} not in snapshot",
                        paragraph=para_id,
                    )

        return result

    def validate_artefact_trace_completeness(
        self,
        artefact_trace: Dict[str, List[str]],
        paragraph_count: int,
        snapshot_claim_ids: Set[str],
    ) -> ValidationResult:
        """Validate trace coverage and claim reference validity.

        This is called during artefact compilation to ensure:
        1. All paragraphs have trace entries (SOFT: warning if missing)
        2. All claim references are valid (HARD: error if referencing unknown claims)

        Args:
            artefact_trace: Dict mapping para_N to list of claim_ids
            paragraph_count: Number of paragraphs in the artefact
            snapshot_claim_ids: Set of claim_ids in the snapshot

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult(valid=True)

        # Check all paragraphs have trace entries
        for i in range(paragraph_count):
            para_key = f"para_{i}"
            if para_key not in artefact_trace:
                result.add_warning(
                    "TRACE_COMPLETE_001",
                    f"Paragraph {i} has no trace entry - cannot trace back to claims"
                )
            elif not artefact_trace[para_key]:
                result.add_warning(
                    "TRACE_COMPLETE_002",
                    f"Paragraph {i} has empty trace entry - no claim references"
                )

        # Check all claim references are valid
        for para_key, claim_ids in artefact_trace.items():
            for cid in claim_ids:
                if cid not in snapshot_claim_ids:
                    result.add_error(
                        "TRACE_COMPLETE_010",
                        f"{para_key} references claim {cid} which is not in the snapshot"
                    )

        return result
