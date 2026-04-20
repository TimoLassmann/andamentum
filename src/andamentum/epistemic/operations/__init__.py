"""Operations - Validated transformations on epistemic entities.

Operations are the atomic units of epistemic work. Each operation:
1. Loads the target entity
2. Runs an agent with adapter normalization
3. Validates gate requirements
4. Updates entity state
5. Checks phase transitions

This package splits operations by pipeline phase. All public names are
re-exported here so that ``from andamentum.epistemic.operations import X`` continues
to work for every X that was previously importable from the monolithic
``operations.py`` module.

Architecture: Layer 1 (framework-agnostic)
"""

from typing import Optional, TYPE_CHECKING

# ── Base layer ───────────────────────────────────────────────────────────
from .base import (
    DEDUP_SIMILARITY_THRESHOLD,
    MAX_INVESTIGATION_ATTEMPTS,
    MAX_UNCERTAINTY_DEPTH,
    AgentRunner,
    BaseOperation,
    DefaultValidator,
    EvidenceGatherer,
    GatheredEvidence,
    OperationResult,
    OperationValidator,
    QualityScore,
    QualityScorer,
    WorkItem,
    _truncate_for_trace,
)

# ── Preplanning (Phases 0-2) ────────────────────────────────────────────
from .preplanning import (
    ClarifyQuestionOperation,
    ClassifyQuestionOperation,
    ConceptualAnalysisOperation,
    PlanTaskOperation,
)

# ── Claims ──────────────────────────────────────────────────────────────
from .claims import (
    EVIDENCE_TOP_K,
    ProposeClaimsOperation,
    select_top_k_evidence,
)

# ── Evidence ────────────────────────────────────────────────────────────
from .evidence import ExtractEvidenceOperation

# ── Scrutiny ────────────────────────────────────────────────────────────
from .scrutiny import ScrutiniseClaimOperation

# ── Stage management ────────────────────────────────────────────────────
from .stage_management import (
    DemoteClaimOperation,
    PromoteClaimOperation,
)

# ── Verification ────────────────────────────────────────────────────────
from .verification import (
    AdversarialSearchOperation,
    AssessConvergenceOperation,
    ValidateDeductivelyOperation,
    VerifyComputationallyOperation,
)

# ── Uncertainty ─────────────────────────────────────────────────────────
from .uncertainty import ResolveUncertaintyOperation

# ── Concern dedup ──────────────────────────────────────────────────────
from .concerns import DeduplicateConcernsOperation

# ── Synthesis ──────────────────────────────────────────────────────────
from .synthesis import (
    FreezeSnapshotOperation,
    SynthesizeReportOperation,
)

# ── Integration ─────────────────────────────────────────────────────────
from .integration import AbductiveIntegrationOperation

# ── Analysis ─────────────────────────────────────────────────────────────
from .analysis import (
    AnalyzeArgumentOperation,
    ContrastiveEvaluationOperation,
    CrossClaimConsistencyOperation,
)

# ── Investigation ────────────────────────────────────────────────────────
from .investigation import (
    GeneratePredictionOperation,
    InvestigateClaimOperation,
    RecordDecisionOperation,
)
from .seed_claim import SeedClaimOperation

# ── Cleanup ─────────────────────────────────────────────────────────────
from .cleanup import AbandonStaleClaimOperation

# ── Belief maintenance (TMS) ────────────────────────────────────────────
from .belief_maintenance import (
    InvalidateEvidenceOperation,
    RevalidateClaimOperation,
    SetRoutingDefaultsOperation,
)

if TYPE_CHECKING:
    from ..repository import EpistemicRepository

# ══════════════════════════════════════════════════════════════════════════════
# OPERATION REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

OPERATION_CLASSES: dict[str, type[BaseOperation]] = {
    # Preplanning
    "clarify_question": ClarifyQuestionOperation,
    "classify_question": ClassifyQuestionOperation,
    "conceptual_analysis": ConceptualAnalysisOperation,
    "plan_task": PlanTaskOperation,
    "propose_claims": ProposeClaimsOperation,
    "seed_claim": SeedClaimOperation,
    # Evidence
    "extract_evidence": ExtractEvidenceOperation,
    # Scrutiny, investigation, and verification
    "scrutinise_claim": ScrutiniseClaimOperation,
    "investigate_claim": InvestigateClaimOperation,
    "adversarial_search": AdversarialSearchOperation,
    "assess_convergence": AssessConvergenceOperation,
    "validate_deductively": ValidateDeductivelyOperation,
    "verify_computationally": VerifyComputationallyOperation,
    # Integration
    "integrate_evidence": AbductiveIntegrationOperation,
    # Argument analysis
    "analyze_argument": AnalyzeArgumentOperation,
    # Pairwise claim operations
    "contrastive_evaluation": ContrastiveEvaluationOperation,
    "cross_claim_consistency": CrossClaimConsistencyOperation,
    # Routing defaults (deterministic, no LLM)
    "set_routing_defaults": SetRoutingDefaultsOperation,
    # Stage management
    "promote_claim": PromoteClaimOperation,
    "demote_claim": DemoteClaimOperation,
    # Uncertainty
    "resolve_uncertainty": ResolveUncertaintyOperation,
    "deduplicate_concerns": DeduplicateConcernsOperation,
    # Prediction
    "generate_prediction": GeneratePredictionOperation,
    # Decision
    "record_decision": RecordDecisionOperation,
    # Cleanup
    "abandon_stale_claim": AbandonStaleClaimOperation,
    # TMS: Belief maintenance
    "invalidate_evidence": InvalidateEvidenceOperation,
    "revalidate_claim": RevalidateClaimOperation,
    # Synthesis
    "freeze_snapshot": FreezeSnapshotOperation,
    "synthesize_report": SynthesizeReportOperation,
}


def create_operations(
    repo: "EpistemicRepository",
    agent_runner: Optional[AgentRunner] = None,
    evidence_gatherer: Optional[EvidenceGatherer] = None,
    quality_scorer: Optional[QualityScorer] = None,
    model: Optional[str] = None,
    providers: Optional[dict] = None,
    embedding_model: Optional[str] = None,
) -> dict[str, BaseOperation]:
    """Create all operation instances.

    When evidence_gatherer is not provided, attempts to create a default
    WebSearchGatherer using deep_research (if installed). Pass
    evidence_gatherer=None explicitly and model=None to disable this.

    Args:
        repo: Repository for entity CRUD
        agent_runner: Optional agent execution protocol
        evidence_gatherer: Optional evidence gathering protocol
        quality_scorer: Optional quality scoring protocol
        model: LLM model string for default evidence gatherer auto-creation
        providers: Optional dict of named evidence providers (e.g., from
            ``andamentum.epistemic.providers.get_biomedical_providers()``).
        embedding_model: Embedding model for similarity/clustering operations.

    Returns:
        Dict mapping operation name to instance
    """
    # Auto-create evidence gatherer if not provided and deep_research is available
    if evidence_gatherer is None and model is not None:
        from ..evidence_gathering import get_default_gatherer

        evidence_gatherer = get_default_gatherer(model=model, providers=providers)

    return {
        name: cls(
            repo,
            agent_runner,
            evidence_gatherer=evidence_gatherer,
            quality_scorer=quality_scorer,
            embedding_model=embedding_model,
        )
        for name, cls in OPERATION_CLASSES.items()
    }


__all__ = [
    # Base
    "BaseOperation",
    "OperationResult",
    "GatheredEvidence",
    "EvidenceGatherer",
    "QualityScorer",
    "QualityScore",
    "AgentRunner",
    "OperationValidator",
    "DefaultValidator",
    "DEDUP_SIMILARITY_THRESHOLD",
    "MAX_INVESTIGATION_ATTEMPTS",
    "MAX_UNCERTAINTY_DEPTH",
    "WorkItem",
    "_truncate_for_trace",
    # Claims
    "EVIDENCE_TOP_K",
    "select_top_k_evidence",
    # Operations
    "ClarifyQuestionOperation",
    "ClassifyQuestionOperation",
    "ConceptualAnalysisOperation",
    "PlanTaskOperation",
    "ProposeClaimsOperation",
    "ExtractEvidenceOperation",
    "ScrutiniseClaimOperation",
    "PromoteClaimOperation",
    "DemoteClaimOperation",
    "AdversarialSearchOperation",
    "AssessConvergenceOperation",
    "ValidateDeductivelyOperation",
    "VerifyComputationallyOperation",
    "ResolveUncertaintyOperation",
    "FreezeSnapshotOperation",
    "SynthesizeReportOperation",
    "AnalyzeArgumentOperation",
    "ContrastiveEvaluationOperation",
    "CrossClaimConsistencyOperation",
    "GeneratePredictionOperation",
    "RecordDecisionOperation",
    "InvestigateClaimOperation",
    "InvalidateEvidenceOperation",
    "RevalidateClaimOperation",
    "SetRoutingDefaultsOperation",
    "AbandonStaleClaimOperation",
    "AbductiveIntegrationOperation",
    # Registry
    "OPERATION_CLASSES",
    "create_operations",
]
