"""Epistemic — Formal epistemology for AI research.

Evidence-based claims with traceability, deterministic stage gates,
and pattern-driven scheduling.

The entry points are :func:`ask` (a research question) and :func:`verify`
(one claim). Both are async, take keyword-only arguments and return one typed
:class:`AskResult`.

Everything else (entities, gates, operations, providers, the repository) stays
importable from its submodule, and from this package for older callers, but is
not part of the public surface. For specialised modules (adversarial search,
convergence detection, prediction), import from the submodule directly::

    from andamentum.epistemic.adversarial_query_generator import generate_adversarial_queries
    from andamentum.epistemic.convergence_detector import detect_convergence
"""

# ruff: noqa: F401  (Names imported here but not in __all__ stay importable from the package for older callers.)

# === Functions you can wrap as agent tools ===
# `EpistemicRepository`, `BaseOperation`, `EvidenceGatherer`, `QualityScorer`
# are classes — wrap their methods (or subclass them and wrap subclass methods).
# To run a full pipeline, use `from andamentum.epistemic.runner import
# DefaultAgentRunner` (lazy-imported to keep pydantic-ai off the critical path).
from .ask_verify import ask, verify
from .confidence import build_verdict, compute_posterior
from .errors import EpistemicError
from .graph import DegenerateQuestionError
from .graph.stages import StageInvariantError
from .result_models import AskResult
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
from .entities.verdict import (
    ClaimVerdict,
    EpistemicVerdict,
    EvidenceRef,
    VerdictProvenance,
    render_verdict_line,
)
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
    # Entry points
    "ask",
    "verify",
    # Results
    "AskResult",
    "PipelineResult",
    "EpistemicVerdict",
    "ClaimVerdict",
    "EvidenceRef",
    "VerdictProvenance",
    "PosteriorReport",
    # Persistence, for reading a finished run
    "EpistemicRepository",
    # Errors
    "EpistemicError",
    "DegenerateQuestionError",
    "EntityNotFoundError",
    "StageInvariantError",
]
