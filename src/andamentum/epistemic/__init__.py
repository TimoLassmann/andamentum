"""Epistemic — Formal epistemology for AI research.

Evidence-based claims with traceability, deterministic stage gates,
and pattern-driven scheduling.

Top-level exports cover the essential public API. For specialised
modules (adversarial search, convergence detection, prediction),
import from the submodule directly::

    from andamentum.epistemic.adversarial_query_generator import generate_adversarial_queries
    from andamentum.epistemic.convergence_detector import detect_convergence
"""

# === Functions you can wrap as agent tools ===
# `EpistemicRepository`, `BaseOperation`, `EvidenceGatherer`, `QualityScorer`
# are classes — wrap their methods (or subclass them and wrap subclass methods).
# To run a full pipeline, use `from andamentum.epistemic.runner import
# DefaultAgentRunner` (lazy-imported to keep pydantic-ai off the critical path).
from .confidence import compute_posterior
from .gates import (
    can_demote,
    check_degeneracy,
    compute_confidence_score,
    get_next_stage,
    get_previous_stage,
    quality_weighted_evidence_sum,
    validate_promotion,
)
from .operations import (
    AgentRunner,
    BaseOperation,
    EvidenceGatherer,
    QualityScorer,
)
from .preflight import preflight
from .repository import EpistemicRepository

# === Result/data types (entities, configs, return values; not tools themselves) ===
from .agents import AGENT_REGISTRY, AgentDefinition
from .confidence import PosteriorReport
from .entities import (
    BLOCKING_TYPES,
    ENTITY_CLASSES,
    Artefact,
    Claim,
    ClaimStage,
    Decision,
    EpistemicEntity,
    Evidence,
    Objective,
    Snapshot,
    Uncertainty,
    UncertaintyScope,
    UncertaintyType,
)
from .gates import (
    STAGE_GATES,
    STAGE_HIERARCHY,
    DegeneracyCodes,
    GateResult,
    StageGate,
)
from .graph.quarantine import QuarantineRecord
from .operations import GatheredEvidence, OperationResult
from .operations.base import OperationInput
from .operations_runner import PipelineResult
from .preflight import CheckResult, HealthCheckable, PreflightResult
from .repository import EntityNotFoundError

__all__ = [
    # Functions / callables
    "AgentRunner",
    "BaseOperation",
    "EpistemicRepository",
    "EvidenceGatherer",
    "QualityScorer",
    "can_demote",
    "check_degeneracy",
    "compute_confidence_score",
    "compute_posterior",
    "get_next_stage",
    "get_previous_stage",
    "preflight",
    "quality_weighted_evidence_sum",
    "validate_promotion",
    # Data types
    "AGENT_REGISTRY",
    "AgentDefinition",
    "Artefact",
    "BLOCKING_TYPES",
    "CheckResult",
    "Claim",
    "ClaimStage",
    "Decision",
    "DegeneracyCodes",
    "ENTITY_CLASSES",
    "EntityNotFoundError",
    "EpistemicEntity",
    "Evidence",
    "GateResult",
    "GatheredEvidence",
    "HealthCheckable",
    "Objective",
    "OperationInput",
    "OperationResult",
    "PipelineResult",
    "PosteriorReport",
    "PreflightResult",
    "QuarantineRecord",
    "STAGE_GATES",
    "STAGE_HIERARCHY",
    "Snapshot",
    "StageGate",
    "Uncertainty",
    "UncertaintyScope",
    "UncertaintyType",
]
