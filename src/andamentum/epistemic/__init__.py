"""Epistemic — Formal epistemology for AI research.

Evidence-based claims with traceability, deterministic stage gates,
and pattern-driven scheduling.

Top-level exports cover the essential public API. For specialised
modules (adversarial search, convergence detection, prediction),
import from the submodule directly::

    from epistemic.adversarial_query_generator import generate_adversarial_queries
    from epistemic.convergence_detector import detect_convergence
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

# ── Storage & Repository ──────────────────────────────────────────────────
from .storage import (
    StorageBackend,
    InMemoryStorageBackend,
    StoredDocument,
    DocumentRef,
    DocumentMetadata,
)
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

# ── Patterns (novel contribution) ─────────────────────────────────────────
from .patterns import (
    Pattern,
    WorkItem,
    PatternScheduler,
    WORK_PATTERNS,
    DEFAULT_OPERATION_BUDGETS,
    SYNTHESIS_OPS,
    MAX_ENTITY_ATTEMPTS,
)

# ── Operations ────────────────────────────────────────────────────────────
from .operations import (
    BaseOperation,
    OperationResult,
    AgentRunner,
    GatheredEvidence,
    EvidenceGatherer,
    QualityScorer,
    OPERATION_CLASSES,
    create_operations,
)

# ── Agents (Python-native definitions) ───────────────────────────────────
from .agents import AgentDefinition, AGENT_REGISTRY

# ── Confidence ────────────────────────────────────────────────────────────
from .confidence import (
    compute_answer_confidence,
    AnswerConfidenceReport,
    compute_posterior,
    PosteriorReport,
)

# ── Preflight ─────────────────────────────────────────────────────────────
from .preflight import CheckResult, PreflightResult, HealthCheckable, preflight

# ── Runner (standalone execution, requires [llm] extra) ─────────────────
# Lazy import to avoid hard dependency on pydantic-ai:
#   from epistemic.runner import DefaultAgentRunner

__all__ = [
    # Core Entities
    "EpistemicEntity", "ENTITY_CLASSES",
    "Objective", "Evidence", "Claim", "ClaimStage",
    "Uncertainty", "UncertaintyType", "UncertaintyScope", "BLOCKING_TYPES",
    "Decision", "Snapshot", "Artefact",
    # Storage & Repository
    "StorageBackend", "InMemoryStorageBackend",
    "StoredDocument", "DocumentRef", "DocumentMetadata",
    "EpistemicRepository", "EntityNotFoundError",
    # Gates
    "StageGate", "GateResult", "STAGE_GATES", "STAGE_HIERARCHY",
    "validate_promotion", "get_next_stage", "get_previous_stage", "can_demote",
    "check_degeneracy", "DegeneracyCodes",
    "quality_weighted_evidence_sum", "compute_confidence_score",
    # Patterns
    "Pattern", "WorkItem", "PatternScheduler", "WORK_PATTERNS",
    "DEFAULT_OPERATION_BUDGETS", "SYNTHESIS_OPS", "MAX_ENTITY_ATTEMPTS",
    # Operations
    "BaseOperation", "OperationResult", "AgentRunner",
    "GatheredEvidence", "EvidenceGatherer", "QualityScorer",
    "OPERATION_CLASSES", "create_operations",
    # Agents
    "AgentDefinition", "AGENT_REGISTRY",
    # Confidence
    "compute_answer_confidence", "AnswerConfidenceReport",
    "compute_posterior", "PosteriorReport",
    # Preflight
    "CheckResult", "PreflightResult", "HealthCheckable", "preflight",
]
