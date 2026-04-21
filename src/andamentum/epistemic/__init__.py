"""Epistemic — Formal epistemology for AI research.

Evidence-based claims with traceability, deterministic stage gates,
and pattern-driven scheduling.

Top-level exports cover the essential public API. For specialised
modules (adversarial search, convergence detection, prediction),
import from the submodule directly::

    from andamentum.epistemic.adversarial_query_generator import generate_adversarial_queries
    from andamentum.epistemic.convergence_detector import detect_convergence
"""

__version__ = "0.1.0"

# ── Core Entities ──────────────────────────────────────────────────────────
from .entities import (
    EpistemicEntity,
    ENTITY_CLASSES,
    Objective,
    Evidence,
    Claim,
    ClaimStage,
    Uncertainty,
    UncertaintyType,
    UncertaintyScope,
    BLOCKING_TYPES,
    Decision,
    Snapshot,
    Artefact,
)

# ── Repository ────────────────────────────────────────────────────────────
from .repository import (
    EpistemicRepository,
    EntityNotFoundError,
)

# ── Gates (novel contribution) ────────────────────────────────────────────
from .gates import (
    StageGate,
    GateResult,
    STAGE_GATES,
    STAGE_HIERARCHY,
    validate_promotion,
    get_next_stage,
    get_previous_stage,
    can_demote,
    check_degeneracy,
    DegeneracyCodes,
    quality_weighted_evidence_sum,
    compute_confidence_score,
)

# ── Patterns ──────────────────────────────────────────────────────────────
from .patterns import OperationInput, WorkItem  # WorkItem is backward compat alias

# ── Pipeline result ──────────────────────────────────────────────────────
from .operations_runner import PipelineResult
from .graph.quarantine import QuarantineRecord

# ── Operations ────────────────────────────────────────────────────────────
from .operations import (
    BaseOperation,
    OperationResult,
    AgentRunner,
    GatheredEvidence,
    EvidenceGatherer,
    QualityScorer,
)

# ── Agents (Python-native definitions) ───────────────────────────────────
from .agents import AgentDefinition, AGENT_REGISTRY

# ── Confidence ────────────────────────────────────────────────────────────
from .confidence import (
    compute_posterior,
    PosteriorReport,
)

# ── Preflight ─────────────────────────────────────────────────────────────
from .preflight import CheckResult, PreflightResult, HealthCheckable, preflight

# ── Provider routing (DEPRECATED — embedding-based, replaced by LLM agent) ──
# Kept for optional fast-path use. Primary routing is now via the
# epistemic_select_provider focused agent in PlanTaskOperation.
from .provider_routing import (
    ProviderScore,
    rank_providers,
    select_providers,
)

# ── Runner (standalone execution) ────────────────────────────────────────
# Lazy import to keep pydantic-ai off the critical import path:
#   from andamentum.epistemic.runner import DefaultAgentRunner

__all__ = [
    # Core Entities
    "EpistemicEntity",
    "ENTITY_CLASSES",
    "Objective",
    "Evidence",
    "Claim",
    "ClaimStage",
    "Uncertainty",
    "UncertaintyType",
    "UncertaintyScope",
    "BLOCKING_TYPES",
    "Decision",
    "Snapshot",
    "Artefact",
    # Repository
    "EpistemicRepository",
    "EntityNotFoundError",
    # Gates
    "StageGate",
    "GateResult",
    "STAGE_GATES",
    "STAGE_HIERARCHY",
    "validate_promotion",
    "get_next_stage",
    "get_previous_stage",
    "can_demote",
    "check_degeneracy",
    "DegeneracyCodes",
    "quality_weighted_evidence_sum",
    "compute_confidence_score",
    # Patterns
    "OperationInput",
    "WorkItem",  # backward compat alias
    # Pipeline result
    "PipelineResult",
    "QuarantineRecord",
    # Operations
    "BaseOperation",
    "OperationResult",
    "AgentRunner",
    "GatheredEvidence",
    "EvidenceGatherer",
    "QualityScorer",
    # Agents
    "AgentDefinition",
    "AGENT_REGISTRY",
    # Confidence
    "compute_posterior",
    "PosteriorReport",
    # Preflight
    "CheckResult",
    "PreflightResult",
    "HealthCheckable",
    "preflight",
    # Provider routing
    "ProviderScore",
    "rank_providers",
    "select_providers",
]
