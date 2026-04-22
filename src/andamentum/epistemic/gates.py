"""Stage Gates - Validation requirements for claim stage transitions.

Each stage transition has deterministic requirements. Gates are checked
inside the promote_claim operation to ensure claims only advance when
all conditions are met.

Features:
- Minimum evidence counts per stage
- Verification track requirements (adversarial, convergence, deductive)
- Blocking uncertainty detection
- Degeneracy detection (Lakatos methodology)

Architecture: Layer 1 (framework-agnostic)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

from .entities import Claim, ClaimStage

if TYPE_CHECKING:
    from .repository import EpistemicRepository


logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE HIERARCHY
# ══════════════════════════════════════════════════════════════════════════════

STAGE_HIERARCHY: dict[ClaimStage, int] = {
    ClaimStage.HYPOTHESIS: 0,
    ClaimStage.SUPPORTED: 1,
    ClaimStage.PROVISIONAL: 2,
    ClaimStage.ROBUST: 3,
    ClaimStage.ACTIONABLE: 4,
}


def is_at_or_above_threshold(stage: ClaimStage, min_stage: ClaimStage) -> bool:
    """Check if a claim stage meets or exceeds a threshold.

    Args:
        stage: The stage to check
        min_stage: The minimum required stage

    Returns:
        True if stage >= min_stage in the hierarchy
    """
    return STAGE_HIERARCHY[stage] >= STAGE_HIERARCHY[min_stage]


# ══════════════════════════════════════════════════════════════════════════════
# GATE DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class StageGate:
    """Requirements for advancing to a claim stage.

    Attributes:
        target_stage: The stage this gate validates entry to
        min_evidence: Minimum number of evidence items required
        min_quality_sum: Minimum sum of quality_score across evidence (0.0 = disabled)
        requires_scrutiny: Whether scrutiny must pass
        requires_adversarial: Whether adversarial search must complete
        requires_convergence: Whether convergence check must complete
        requires_deductive: Whether deductive validation must complete
        requires_computational: Whether computational verification must complete
        blocks_on_uncertainties: Whether blocking uncertainties prevent advancement
        custom_check: Optional async function for additional validation
    """

    target_stage: ClaimStage
    min_evidence: int
    min_quality_sum: float  # Min sum of quality_score across stage evidence; promotion requires at least this total (0.0 = disabled)
    requires_scrutiny: bool
    requires_adversarial: bool
    requires_convergence: bool
    requires_deductive: bool
    requires_computational: bool
    blocks_on_uncertainties: bool
    min_supporting_sources: int = (
        0  # Log-odds: independent clusters judged "supports" (0 = disabled)
    )
    adversarial_balance_threshold: float = (
        0.0  # 0.0 = disabled; claim must have balance >= this
    )
    custom_check: Optional[
        Callable[["Claim", "EpistemicRepository"], Awaitable[bool]]
    ] = None

    def describe(self) -> str:
        """Return human-readable description of gate requirements."""
        reqs: list[str] = []
        reqs.append(f"≥{self.min_evidence} evidence")
        if self.min_quality_sum > 0:
            reqs.append(f"quality sum ≥{self.min_quality_sum}")
        if self.requires_scrutiny:
            reqs.append("scrutiny passed")
        if self.requires_adversarial:
            reqs.append("adversarial checked")
        if self.requires_convergence:
            reqs.append("convergence checked")
        if self.requires_deductive:
            reqs.append("deductive validated")
        if self.requires_computational:
            reqs.append("computationally verified")
        if self.blocks_on_uncertainties:
            reqs.append("no blocking uncertainties")
        if self.adversarial_balance_threshold > 0:
            reqs.append(f"adversarial balance ≥{self.adversarial_balance_threshold}")
        if self.custom_check:
            reqs.append("custom check")
        return f"To reach {self.target_stage.value}: {', '.join(reqs)}"


@dataclass
class GateResult:
    """Result of gate validation.

    Attributes:
        passed: Whether all requirements were met
        blocking_reasons: List of reasons why gate didn't pass
        warnings: Non-blocking issues to be aware of
    """

    passed: bool
    blocking_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reason: Optional[str] = None  # Legacy compatibility

    def __bool__(self) -> bool:
        return self.passed


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM GATE CHECKS
# ══════════════════════════════════════════════════════════════════════════════


async def check_independent_evidence_lines(
    claim: "Claim", repo: "EpistemicRepository"
) -> bool:
    """Check that claim has independent evidence from different domains.

    ROBUST stage requires evidence from at least 2 different domains
    with domain distance > 0.5.

    Args:
        claim: The claim being promoted
        repo: Repository for loading evidence

    Returns:
        True if independent evidence lines exist
    """
    if claim.evidence_count < 3:
        return False

    # Load evidence for this claim
    evidence_list = await repo.query(
        "evidence",
        objective_id=claim.objective_id,
    )

    # Filter to evidence linked to this claim via evidence_ids
    claim_evidence = [e for e in evidence_list if e.entity_id in claim.evidence_ids]

    if len(claim_evidence) < 3:
        return False

    # Check domain diversity using source_type (Evidence has no "domain" field)
    domains: set[str] = set()
    for e in claim_evidence:
        domains.add(e.source_type)

    # Require at least 2 different domains
    return len(domains) >= 2


async def check_decision_criteria(claim: "Claim", repo: "EpistemicRepository") -> bool:
    """Check that claim has decision criteria defined.

    ACTIONABLE stage requires "what would change my mind" criteria.

    Args:
        claim: The claim being promoted
        repo: Repository for loading related entities

    Returns:
        True if decision criteria are present
    """
    # Claim has no "reversal_criteria" field — check assumptions as decision criteria proxy
    if claim.assumptions and len(claim.assumptions) > 0:
        return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
# STAGE GATE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════


STAGE_GATES: dict[ClaimStage, StageGate] = {
    ClaimStage.SUPPORTED: StageGate(
        target_stage=ClaimStage.SUPPORTED,
        min_evidence=1,
        min_quality_sum=0.3,
        requires_scrutiny=True,
        requires_adversarial=False,
        requires_convergence=False,
        requires_deductive=False,
        requires_computational=False,
        blocks_on_uncertainties=True,
        min_supporting_sources=1,
    ),
    ClaimStage.PROVISIONAL: StageGate(
        target_stage=ClaimStage.PROVISIONAL,
        min_evidence=2,
        min_quality_sum=0.5,
        requires_scrutiny=True,
        requires_adversarial=True,
        requires_convergence=True,
        requires_deductive=True,
        requires_computational=False,
        blocks_on_uncertainties=True,
        min_supporting_sources=2,
        adversarial_balance_threshold=0.4,
    ),
    ClaimStage.ROBUST: StageGate(
        target_stage=ClaimStage.ROBUST,
        min_evidence=3,
        min_quality_sum=1.5,
        requires_scrutiny=True,
        requires_adversarial=True,
        requires_convergence=True,
        requires_deductive=True,
        requires_computational=False,
        blocks_on_uncertainties=True,
        min_supporting_sources=3,
        custom_check=check_independent_evidence_lines,
    ),
    ClaimStage.ACTIONABLE: StageGate(
        target_stage=ClaimStage.ACTIONABLE,
        min_evidence=3,
        min_quality_sum=1.5,
        requires_scrutiny=True,
        requires_adversarial=True,
        requires_convergence=True,
        requires_deductive=True,
        requires_computational=False,
        blocks_on_uncertainties=True,
        min_supporting_sources=3,
        custom_check=check_decision_criteria,
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# DEGENERACY DETECTION (Lakatos Methodology)
# ══════════════════════════════════════════════════════════════════════════════


def count_modifications_in_window(
    modification_timestamps: list[str],
    hours: int = 24,
) -> int:
    """Count modifications within a time window.

    Args:
        modification_timestamps: List of ISO-format timestamp strings when claim was modified
        hours: Window size in hours

    Returns:
        Number of modifications in the window
    """
    if not modification_timestamps:
        return 0

    cutoff = datetime.now() - timedelta(hours=hours)
    return sum(
        1 for ts in modification_timestamps if datetime.fromisoformat(ts) > cutoff
    )


class DegeneracyCodes:
    """Degeneracy detection codes from Lakatos methodology.

    A research program is degenerative when it:
    - Repeatedly modifies claims without new evidence (ad hoc patches)
    - Shows rapid oscillation between states
    - Accumulates modifications without convergence
    """

    # Too many total modifications
    DEGEN_001 = "DEGEN_001"
    DEGEN_001_MSG = "Excessive modifications ({count} > 3 total)"

    # Recent burst of modifications
    DEGEN_003 = "DEGEN_003"
    DEGEN_003_MSG = "Modification burst ({count} in {hours}h window)"


def check_degeneracy(claim: "Claim") -> list[str]:
    """Check claim for degenerative research patterns.

    Args:
        claim: The claim to check

    Returns:
        List of degeneracy warning codes with messages
    """
    warnings = []

    # DEGEN_001: Total modification count
    if claim.modification_count > 3:
        warnings.append(
            f"{DegeneracyCodes.DEGEN_001}: "
            f"{DegeneracyCodes.DEGEN_001_MSG.format(count=claim.modification_count)}"
        )

    # DEGEN_003: Recent burst
    modification_timestamps = claim.modification_timestamps
    recent_mods = count_modifications_in_window(modification_timestamps, hours=24)
    if recent_mods >= 3:
        warnings.append(
            f"{DegeneracyCodes.DEGEN_003}: "
            f"{DegeneracyCodes.DEGEN_003_MSG.format(count=recent_mods, hours=24)}"
        )

    return warnings


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY-WEIGHTED EVIDENCE
# ══════════════════════════════════════════════════════════════════════════════


async def quality_weighted_evidence_sum(
    claim: "Claim", repo: "EpistemicRepository"
) -> float:
    """Sum of quality_score for all scored evidence supporting this claim.

    Evidence without a quality_score is skipped (not counted toward the sum).
    This avoids inflating quality sums with hardcoded defaults.

    Args:
        claim: The claim to check
        repo: Repository for loading evidence

    Returns:
        Sum of quality scores across all scored, non-invalidated evidence
    """
    total = 0.0
    for eid in claim.evidence_ids:
        try:
            evidence = await repo.get("evidence", eid)
            if evidence.invalidated:
                continue
            if getattr(evidence, "cluster_status", "unclustered") in (
                "corroborative",
                "deferred",
            ):
                continue
            if evidence.quality_score is not None:
                total += evidence.quality_score
        except Exception as e:
            logger.warning(
                "quality_weighted_evidence_sum: failed to load evidence %s: %s", eid, e
            )
    return total


async def _any_evidence_judged(claim: "Claim", repo: "EpistemicRepository") -> bool:
    """Check if any evidence linked to this claim has been judged."""
    for eid in claim.evidence_ids:
        try:
            evidence = await repo.get("evidence", eid)
            if getattr(evidence, "support_judgment", None) is not None:
                return True
        except Exception as e:
            logger.warning(
                "_any_evidence_judged: failed to load evidence %s: %s", eid, e
            )
    return False


async def count_supporting_sources(claim: "Claim", repo: "EpistemicRepository") -> int:
    """Count independent evidence clusters judged as 'supports' for this claim.

    Only counts evidence that is:
    - Not invalidated
    - Representative (not corroborative or deferred)
    - Judged as 'supports' by the evidence judge

    This is the domain-independent gate criterion: how many independent
    sources actively support this claim?

    Args:
        claim: The claim to check
        repo: Repository for loading evidence

    Returns:
        Count of supporting representative evidence items
    """
    count = 0
    for eid in claim.evidence_ids:
        try:
            evidence = await repo.get("evidence", eid)
            if evidence.invalidated:
                continue
            if getattr(evidence, "cluster_status", "unclustered") in (
                "corroborative",
                "deferred",
            ):
                continue
            if getattr(evidence, "support_judgment", None) == "supports":
                count += 1
        except Exception as e:
            logger.warning(
                "count_supporting_sources: failed to load evidence %s: %s", eid, e
            )
    return count


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE SCORE COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════


# Per-claim stage-based confidence used by PromoteClaimOperation.
# This is a pipeline-internal score for gate decisions, separate from
# the post-hoc posterior confidence (confidence.py:compute_posterior).
_STAGE_CONFIDENCE: dict["ClaimStage", tuple[float, float]] = {
    ClaimStage.HYPOTHESIS: (0.1, 0.0),
    ClaimStage.SUPPORTED: (0.3, 0.1),
    ClaimStage.PROVISIONAL: (0.5, 0.15),
    ClaimStage.ROBUST: (0.7, 0.15),
    ClaimStage.ACTIONABLE: (0.85, 0.1),
}


def compute_confidence_score(
    stage: "ClaimStage",
    avg_quality: float,
    adversarial_balance: Optional[float] = None,
) -> float:
    """Compute confidence score from stage, evidence quality, and adversarial balance.

    Args:
        stage: Current claim stage
        avg_quality: Average quality_score of supporting evidence (0.0-1.0)
        adversarial_balance: Optional adversarial balance score (0.0-1.0)

    Returns:
        Confidence score 0.0-1.0
    """
    base, bonus_weight = _STAGE_CONFIDENCE.get(stage, (0.1, 0.0))
    score = base + avg_quality * bonus_weight
    # Adversarial balance penalty: if claim is challenged, reduce confidence
    if adversarial_balance is not None and adversarial_balance < 0.6:
        penalty = (0.6 - adversarial_balance) * 0.3  # Max penalty ~0.18
        score -= penalty
    return max(0.0, min(1.0, score))


# ══════════════════════════════════════════════════════════════════════════════
# GATE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════


async def validate_promotion(
    claim: "Claim",
    target_stage: ClaimStage,
    repo: "EpistemicRepository",
    question_type: Optional[str] = None,
) -> GateResult:
    """Check if claim can be promoted to target stage.

    Args:
        claim: The claim to validate
        target_stage: The stage to promote to
        repo: Repository for loading related entities
        question_type: Optional question type for routing-aware gate thresholds.
            When provided, overrides default thresholds using the routing config.
            Falls back to default thresholds if unknown or unavailable.

    Returns:
        GateResult with pass/fail and reasons
    """
    gate = STAGE_GATES.get(target_stage)
    if not gate:
        return GateResult(
            passed=False,
            blocking_reasons=[f"Unknown target stage: {target_stage}"],
            warnings=[],
        )

    # Apply question-type routing overrides
    overrides: dict[str, object] = {}
    if question_type:
        try:
            from .routing import get_routing_profile

            profile = get_routing_profile(question_type)
            stage_key = target_stage.value.lower()  # e.g., "supported", "provisional"
            overrides = profile.gate_thresholds.get(stage_key, {})
        except (KeyError, ImportError):
            pass  # Fall back to default thresholds

    reasons: list[str] = []
    warnings: list[str] = []

    # Evidence count — use override if available
    min_evidence = int(overrides.get("min_evidence_weighted", gate.min_evidence))  # type: ignore[arg-type]
    if claim.evidence_count < min_evidence:
        reasons.append(f"Need {min_evidence} evidence, have {claim.evidence_count}")

    # Quality-weighted evidence sum — use override if available
    min_quality = float(overrides.get("min_quality_mean", gate.min_quality_sum))  # type: ignore[arg-type]
    if min_quality > 0:
        try:
            quality_sum = await quality_weighted_evidence_sum(claim, repo)
            if quality_sum < min_quality:
                reasons.append(
                    f"Quality sum {quality_sum:.2f} < {min_quality:.1f} required"
                )
        except Exception as e:
            warnings.append(f"Could not compute quality sum: {e}")

    # Supporting sources OR adversarial survival (Popper corroboration).
    # Direct supporting evidence is the primary path. But when adversarial
    # search has actively looked for counterevidence and found none
    # (high adversarial balance), that survival counts as corroboration.
    if gate.min_supporting_sources > 0:
        try:
            supporting = await count_supporting_sources(claim, repo)
            any_judged = await _any_evidence_judged(claim, repo)

            adversarial_survived = (
                claim.adversarial_checked
                and claim.adversarial_balance is not None
                and claim.adversarial_balance >= 0.7
            )

            if (
                any_judged
                and supporting < gate.min_supporting_sources
                and not adversarial_survived
            ):
                reasons.append(
                    f"Need {gate.min_supporting_sources} supporting sources, have {supporting}"
                )
        except Exception as e:
            warnings.append(f"Could not count supporting sources: {e}")

    # Scrutiny
    if gate.requires_scrutiny and claim.scrutiny_verdict != "pass":
        reasons.append(f"Scrutiny not passed (verdict: {claim.scrutiny_verdict})")

    # Verification tracks — routing-aware.
    # Only require tracks that are PRIMARY or SECONDARY for this question type.
    # SKIP and IF_APPLICABLE tracks are not required for promotion.
    _TRACK_TO_GATE_FLAG = {
        "adversarial": (gate.requires_adversarial, "adversarial_checked"),
        "convergence": (gate.requires_convergence, "convergence_checked"),
        "deductive": (gate.requires_deductive, "deductive_checked"),
        "computational": (gate.requires_computational, "computational_checked"),
    }
    _TRACK_TO_LABEL = {
        "adversarial": "Adversarial search not complete",
        "convergence": "Convergence check not complete",
        "deductive": "Deductive validation not complete",
        "computational": "Computational verification not complete",
    }

    if question_type:
        try:
            from .routing import get_routing_profile, TrackActivation

            routing = get_routing_profile(question_type)
            for track_name, (gate_required, claim_field) in _TRACK_TO_GATE_FLAG.items():
                if not gate_required:
                    continue
                activation = routing.tracks.get(track_name, TrackActivation.SKIP)
                if activation in (TrackActivation.PRIMARY, TrackActivation.SECONDARY):
                    if not getattr(claim, claim_field, False):
                        reasons.append(_TRACK_TO_LABEL[track_name])
                # SKIP and IF_APPLICABLE: do not require for promotion
        except (KeyError, ImportError):
            # Fallback: use hardcoded gate requirements (backward compat)
            for track_name, (gate_required, claim_field) in _TRACK_TO_GATE_FLAG.items():
                if gate_required and not getattr(claim, claim_field, False):
                    reasons.append(_TRACK_TO_LABEL[track_name])
    else:
        # No question_type: fall back to hardcoded gate requirements
        for track_name, (gate_required, claim_field) in _TRACK_TO_GATE_FLAG.items():
            if gate_required and not getattr(claim, claim_field, False):
                reasons.append(_TRACK_TO_LABEL[track_name])

    # Adversarial balance threshold — use override if available
    balance_threshold = float(  # type: ignore[arg-type]
        overrides.get("min_adversarial_balance", gate.adversarial_balance_threshold)  # type: ignore[arg-type]
    )
    if balance_threshold > 0:
        balance = claim.adversarial_balance
        if balance is not None and balance < balance_threshold:
            reasons.append(
                f"Adversarial balance {balance:.2f} < {balance_threshold} required"
            )

    # Routing-specific requirements
    if overrides.get("requires_falsification_criteria"):
        if not claim.predictions_generated:
            reasons.append(
                "Falsification criteria required but no predictions generated"
            )
        elif not any(p.get("failure_criteria") for p in claim.predictions):
            reasons.append("Predictions exist but none have falsification criteria")

    if overrides.get("requires_fact_value_separation"):
        # For normative questions — advisory check until claim tagging is implemented (Phase 3)
        warnings.append(
            "Fact-value separation check: advisory — full implementation pending"
        )

    # Blocking uncertainties
    if gate.blocks_on_uncertainties:
        try:
            blocking = await repo.query(
                "uncertainty",
                affected_claim_ids__contains=claim.entity_id,
                resolution=None,
            )
            # Filter to only blocking uncertainties
            blocking = [u for u in blocking if u.is_blocking]
            if blocking:
                reasons.append(f"{len(blocking)} blocking uncertainties unresolved")
        except Exception as e:
            # If we can't verify no blocking uncertainties exist, we must block.
            # Allowing promotion when the safety check itself fails is dangerous —
            # claims could bypass uncertainty validation due to a transient error.
            reasons.append(f"Could not check blocking uncertainties: {e}")

    # Degeneracy detection
    degeneracy_warnings = check_degeneracy(claim)
    for dw in degeneracy_warnings:
        # Degeneracy warnings are blocking for promotions
        reasons.append(dw)

    # Custom check
    if gate.custom_check:
        try:
            custom_ok = await gate.custom_check(claim, repo)
            if not custom_ok:
                reasons.append(f"Custom gate check failed for {target_stage.value}")
        except Exception as e:
            # If the custom check crashes, block promotion. A failing safety check
            # must not silently allow claims through.
            reasons.append(f"Custom gate check error for {target_stage.value}: {e}")

    return GateResult(
        passed=len(reasons) == 0,
        blocking_reasons=reasons,
        warnings=warnings,
        reason=reasons[0] if reasons else None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TMS: CURRENT STAGE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════


async def validate_current_stage(
    claim: "Claim",
    repo: "EpistemicRepository",
) -> GateResult:
    """Check if claim still meets requirements for its CURRENT stage.

    Unlike validate_promotion (which checks the NEXT stage), this checks
    whether a claim should remain at its current stage after evidence changes
    or new adversarial/contradictory information.

    Checks:
    1. min_evidence — enough non-invalidated evidence remaining
    2. min_quality_sum — quality-weighted evidence still sufficient
    3. adversarial_balance — if adversarial search found refutation (< 0.3)
    4. support_balance — if contradicting evidence outweighs supporting

    Checks 3-4 only apply when the relevant data exists (adversarial has
    been run, evidence has been judged). This avoids failing on data that
    hasn't been collected yet.

    HYPOTHESIS always passes (no gate below it).

    Args:
        claim: The claim to validate
        repo: Repository for loading evidence

    Returns:
        GateResult with pass/fail and reasons
    """
    gate = STAGE_GATES.get(claim.stage)
    if not gate:
        # HYPOTHESIS has no gate — always passes
        return GateResult(passed=True)

    reasons: list[str] = []

    # Count non-invalidated evidence
    valid_evidence_count = 0
    for eid in claim.evidence_ids:
        try:
            evidence = await repo.get("evidence", eid)
            if not evidence.invalidated:
                valid_evidence_count += 1
        except Exception as e:
            logger.warning(
                "validate_current_stage: failed to load evidence %s: %s", eid, e
            )

    if valid_evidence_count < gate.min_evidence:
        reasons.append(
            f"Need {gate.min_evidence} evidence, have {valid_evidence_count} (after invalidation)"
        )

    # Quality-weighted sum (already filters invalidated in quality_weighted_evidence_sum)
    if gate.min_quality_sum > 0:
        try:
            quality_sum = await quality_weighted_evidence_sum(claim, repo)
            if quality_sum < gate.min_quality_sum:
                reasons.append(
                    f"Quality sum {quality_sum:.2f} < {gate.min_quality_sum:.1f} required (after invalidation)"
                )
        except Exception as e:
            logger.warning("validate_current_stage: quality sum check failed: %s", e)

    # Adversarial balance — if adversarial search has run and found severe refutation,
    # the claim shouldn't remain at its current stage. 0.3 is the same threshold
    # used in AdversarialSearchOperation to flag severe challenges.
    adversarial_balance = getattr(claim, "adversarial_balance", None)
    if adversarial_balance is not None and adversarial_balance < 0.3:
        reasons.append(
            f"Adversarial balance {adversarial_balance:.2f} indicates refutation (threshold 0.3)"
        )

    # Support/contradict balance — if judged contradicting evidence outweighs supporting,
    # the claim's epistemic foundation is undermined. Only check when evidence has been
    # judged (support_judgment is not None) to avoid false triggers on unjudged evidence.
    supporting = 0
    contradicting = 0
    for eid in claim.evidence_ids:
        ev = await repo.get("evidence", eid)
        if ev.invalidated:
            continue
        judgment = getattr(ev, "support_judgment", None)
        if judgment == "supports":
            supporting += 1
        elif judgment == "contradicts":
            contradicting += 1
    # Only fail if we have at least 2 judged items and contradicts >= supports
    if (supporting + contradicting) >= 2 and contradicting >= supporting:
        reasons.append(
            f"Contradicting evidence ({contradicting}) >= supporting ({supporting})"
        )

    return GateResult(
        passed=len(reasons) == 0,
        blocking_reasons=reasons,
        reason=reasons[0] if reasons else None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STAGE NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════


def get_next_stage(current_stage: ClaimStage) -> Optional[ClaimStage]:
    """Get the next stage in the promotion sequence.

    Args:
        current_stage: Current claim stage

    Returns:
        Next stage or None if at max
    """
    sequence = [
        ClaimStage.HYPOTHESIS,
        ClaimStage.SUPPORTED,
        ClaimStage.PROVISIONAL,
        ClaimStage.ROBUST,
        ClaimStage.ACTIONABLE,
    ]

    try:
        idx = sequence.index(current_stage)
        if idx < len(sequence) - 1:
            return sequence[idx + 1]
    except ValueError:
        pass

    return None


def get_previous_stage(current_stage: ClaimStage) -> Optional[ClaimStage]:
    """Get the previous stage for demotion.

    Args:
        current_stage: Current claim stage

    Returns:
        Previous stage or None if at min
    """
    sequence = [
        ClaimStage.HYPOTHESIS,
        ClaimStage.SUPPORTED,
        ClaimStage.PROVISIONAL,
        ClaimStage.ROBUST,
        ClaimStage.ACTIONABLE,
    ]

    try:
        idx = sequence.index(current_stage)
        if idx > 0:
            return sequence[idx - 1]
    except ValueError:
        pass

    return None


def can_demote(current_stage: ClaimStage) -> bool:
    """Check if demotion is possible from current stage.

    From philosophy: demotion is allowed and expected.
    Claims can be demoted to any lower stage.
    """
    return current_stage != ClaimStage.HYPOTHESIS


def get_demotion_targets(current_stage: ClaimStage) -> list[ClaimStage]:
    """Get valid demotion targets from current stage."""
    stage_order = sorted(STAGE_HIERARCHY, key=lambda s: STAGE_HIERARCHY[s])
    current_idx = stage_order.index(current_stage)
    return stage_order[:current_idx]


def get_valid_promotions(current_stage: ClaimStage) -> list[ClaimStage]:
    """Get list of valid promotion targets from current stage."""
    next_stage = get_next_stage(current_stage)
    return [next_stage] if next_stage else []


def describe_all_gates() -> str:
    """Get human-readable description of all stage gates."""
    lines = ["Stage Gate Requirements:", ""]
    for stage, gate in STAGE_GATES.items():
        reqs = []
        reqs.append(f"≥{gate.min_evidence} evidence")
        if gate.requires_scrutiny:
            reqs.append("scrutiny passed")
        if gate.requires_adversarial:
            reqs.append("adversarial checked")
        if gate.requires_convergence:
            reqs.append("convergence checked")
        if gate.requires_deductive:
            reqs.append("deductive validated")
        if gate.requires_computational:
            reqs.append("computationally verified")
        if gate.blocks_on_uncertainties:
            reqs.append("no blocking uncertainties")
        if gate.adversarial_balance_threshold > 0:
            reqs.append(f"adversarial balance ≥{gate.adversarial_balance_threshold}")
        if gate.custom_check:
            reqs.append("custom check")
        lines.append(f"  → {stage.value}: {', '.join(reqs)}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY SHIMS
# These functions provide backward compatibility with the old gates API
# used by the validation module.
# ══════════════════════════════════════════════════════════════════════════════


def get_gate(from_stage: ClaimStage, to_stage: ClaimStage) -> Optional[StageGate]:
    """Get the stage gate for a transition.

    Legacy compatibility shim. The new API uses STAGE_GATES directly.

    Args:
        from_stage: Current claim stage
        to_stage: Target claim stage

    Returns:
        StageGate or None if transition is not valid
    """
    # Only valid if to_stage is the next stage
    next_stage = get_next_stage(from_stage)
    if next_stage != to_stage:
        return None
    return STAGE_GATES.get(to_stage)


def check_promotion_gate(
    claim_id: str,
    current_stage: ClaimStage,
    proposed_stage: ClaimStage,
    evidence_count: int,
    has_uncertainties_listed: bool,
    has_skeptic_review: bool,
    justification_links: Optional[dict[str, Any]] = None,
) -> GateResult:
    """Check if a claim can be promoted through a stage gate.

    Legacy compatibility shim. The new API uses validate_promotion with
    a Claim object and repository.

    This synchronous version performs basic checks without repository access.
    For full validation including blocking uncertainties, use validate_promotion.

    Args:
        claim_id: ID of the claim to promote
        current_stage: Current claim stage
        proposed_stage: Target stage for promotion
        evidence_count: Number of evidence items linked to claim
        has_uncertainties_listed: Whether claim has uncertainty annotations
        has_skeptic_review: Whether claim has been reviewed by skeptic

    Returns:
        GateResult with pass/fail and reasoning
    """
    gate = STAGE_GATES.get(proposed_stage)
    if not gate:
        return GateResult(
            passed=False,
            blocking_reasons=[f"No gate defined for stage: {proposed_stage}"],
        )

    # Verify sequential progression
    next_stage = get_next_stage(current_stage)
    if next_stage != proposed_stage:
        return GateResult(
            passed=False,
            blocking_reasons=[
                f"Cannot promote from {current_stage.value} to {proposed_stage.value} - "
                f"must go to {next_stage.value if next_stage else 'none'} first"
            ],
        )

    reasons: list[str] = []

    # Evidence count
    if evidence_count < gate.min_evidence:
        reasons.append(f"Need {gate.min_evidence} evidence, have {evidence_count}")

    # Scrutiny check
    if gate.requires_scrutiny and not has_skeptic_review:
        reasons.append("Scrutiny not passed")

    return GateResult(
        passed=len(reasons) == 0,
        blocking_reasons=reasons,
        reason=reasons[0] if reasons else None,
    )
