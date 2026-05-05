"""Foundation layer for all epistemic operations.

Contains protocols, result types, constants, helper functions, and the
abstract BaseOperation class that every concrete operation inherits from.
All operation modules in this package depend on this module.

Architecture: Layer 1 (framework-agnostic)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json as _json
from typing import Any, Mapping, Optional, Protocol, TYPE_CHECKING

from ..adapters import ADAPTERS

if TYPE_CHECKING:
    from ..repository import EpistemicRepository
    from .identifier_extraction import Identifiers


# ══════════════════════════════════════════════════════════════════════════════
# TYPE ALIASES
# ══════════════════════════════════════════════════════════════════════════════

#: Registry mapping provider name → gatherer duck-typed instance.
#: Each value must implement ``gather(query: str) -> list[GatheredEvidence]``.
ProviderRegistry = Mapping[str, Any]


# ══════════════════════════════════════════════════════════════════════════════
# OPERATION INPUT
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class OperationInput:
    """Input for an epistemic operation.

    Specifies which entity to process and which operation to run.
    """

    entity_id: str
    entity_type: str
    operation: str
    metadata: dict[str, Any] = field(
        default_factory=dict
    )  # TODO(theme-5): per-operation TypedDicts when feasible


# Backward compatibility alias
WorkItem = OperationInput

# Cosine similarity threshold for deduplication across all sites.
# The embedding model (embeddinggemma) produces within-group similarities
# of 0.6-0.75 for texts that are clearly about the same topic, so 0.7
# catches most true duplicates while staying above noise.
# The LLM validation pass (validate_groups) is the safety net for
# any false merges this threshold produces.
DEDUP_SIMILARITY_THRESHOLD = 0.7


# ══════════════════════════════════════════════════════════════════════════════
# RESULT TYPES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class OperationResult:
    """Result from executing an operation.

    Attributes:
        success: Whether operation completed successfully
        entity_id: ID of the primary entity affected
        message: Human-readable status message
        created_entities: IDs of any new entities created
        validation_errors: Gate or validation errors that blocked success
        did_work: Whether this call performed real work. False when an
            operation short-circuited because the requested change was
            already in place (e.g. uncertainty already resolved). Graph
            nodes use this to avoid treating idempotent no-ops as fresh
            state changes that should re-trigger downstream work.
    """

    success: bool
    entity_id: str
    message: str = ""
    created_entities: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    did_work: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE GATHERER PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class GatheredEvidence:
    """Raw evidence returned by an EvidenceGatherer.

    Attributes:
        content: Human-readable summary (always present, used by LLM agents)
        source_ref: Primary identifier (DOI, URL, NCT number, ChEMBL ID)
        source_type: Provider name ("pubmed", "biorxiv", "chembl", etc.)
        evidence_kind: What type of evidence ("literature", "preprint",
            "clinical_trial", "bioactivity", "genetic_association", "web_page")
        identifiers: Structured identifiers for dedup and cross-reference
            (e.g. {"doi": "...", "pmid": "...", "nct_id": "..."})
        structured_data: Provider-specific structured fields preserved as-is
            (e.g. trial phase, IC50 values, association scores)
        limitations: Provider-reported caveats
        quality_score: Pre-populated source quality 0.0-1.0 (None if unscored)
        quality_metadata: Raw quality assessment data for traceability
    """

    content: str
    source_ref: str
    source_type: str
    evidence_kind: str = "unknown"
    identifiers: dict[str, str] = field(default_factory=dict)
    structured_data: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    quality_score: Optional[float] = None
    quality_metadata: Optional[dict[str, Any]] = None


class EvidenceGatherer(Protocol):
    """Gathers raw evidence from external sources.

    The epistemic system passes natural language intent. Each provider
    implementation handles its own query construction - the epistemic
    system does not learn provider-specific APIs.
    """

    async def gather(self, source_type: str, query: str) -> list[GatheredEvidence]:
        """Gather evidence from external sources.

        Args:
            source_type: Type of source to query (e.g., "web_search", "knowledge_sources")
            query: Natural language query describing what evidence to find

        Returns:
            List of gathered evidence items
        """
        ...


@dataclass
class QualityScore:
    """Source quality assessment result.

    Attributes:
        score: Composite quality score 0.0-1.0
        source: Which scorer produced this ("openalex", "heuristic", etc.)
        raw_metadata: Full assessment data for traceability
    """

    score: float
    source: str
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class QualityScorer(Protocol):
    """Scores evidence source quality via a bibliometric resolver.

    Currently implemented by ``OpenAlexQualityScorer``. Future
    Crossref or Semantic Scholar implementations would slot in via
    the same Protocol.

    Injected via ``create_operations()``. When the scorer returns
    None (no identifiers found, or lookup failed), evidence quality
    falls through to LLM-based assessment in
    ``ExtractEvidenceOperation._score_evidence`` Path 2.
    """

    async def score(
        self,
        identifiers: "Identifiers",
        source_ref: str,
        source_type: str,
    ) -> QualityScore:
        """Score a source's quality from pre-extracted identifiers.

        Phase 3 of the efficiency plan: identifiers are extracted
        upstream (via ``operations.identifier_extraction.extract_identifiers``)
        from both the source_ref and the evidence content body, so the
        scorer has more context than just the source_ref. The scorer
        no longer does identifier extraction itself.

        Args:
            identifiers: Pre-extracted DOI / PMID / arXiv identifiers.
                When all are None, the scorer should return None.
            source_ref: Source reference (kept for logging context).
            source_type: Source type, e.g. "openalex", "web_search"
                (kept for logging context).

        Returns:
            QualityScore with composite score and metadata, or None
            when no identifier resolved.
        """
        ...


# NOTE on Peirce-cycling caps. Two constants used to live here —
# ``MAX_INVESTIGATION_ATTEMPTS`` and ``MAX_UNCERTAINTY_DEPTH``. They
# were never referenced; the actual caps were inline ``3``s in
# ``graph/nodes.py``. As of 2026-05-05 the canonical Peirce-cycling
# constant is ``epistemic.thresholds.PEIRCE_CYCLE_CAP``, which all
# three Peirce-grounded loops (investigation, scrutiny↔resolve,
# uncertainty depth) share. See that module for the philosophical
# basis.


# ══════════════════════════════════════════════════════════════════════════════
# AGENT RUNNER PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════


class AgentRunner(Protocol):
    """Protocol for running agents.

    This abstracts the actual agent execution, which may be:
    - PydanticAI agents
    - Mock agents for testing
    - MCP tool calls
    """

    async def run(self, agent_name: str, **kwargs: Any) -> Any:
        """Run an agent and return its output."""
        ...


class OperationValidator(Protocol):
    """Protocol for validating operations."""

    async def validate_pre(self, operation: str, entity: Any) -> tuple[bool, list[str]]:
        """Validate before operation runs."""
        ...

    async def validate_post(
        self, operation: str, entity: Any, result: Any
    ) -> tuple[bool, list[str]]:
        """Validate after operation runs."""
        ...


class DefaultValidator:
    """Default validator that always passes."""

    async def validate_pre(self, operation: str, entity: Any) -> tuple[bool, list[str]]:
        return True, []

    async def validate_post(
        self, operation: str, entity: Any, result: Any
    ) -> tuple[bool, list[str]]:
        return True, []


# ══════════════════════════════════════════════════════════════════════════════
# TRACE HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _truncate_for_trace(value: Any, max_length: int = 5000) -> str:
    """Convert a value to string for trace storage, truncating if needed."""
    if value is None:
        return "null"
    if isinstance(value, str):
        if len(value) > max_length:
            return value[:max_length] + f"\n... [truncated, {len(value)} chars total]"
        return value
    if hasattr(value, "model_dump"):
        try:
            s = _json.dumps(value.model_dump(mode="json"), indent=2, default=str)
        except Exception:
            s = str(value)
    elif isinstance(value, (dict, list)):
        try:
            s = _json.dumps(value, indent=2, default=str)
        except Exception:
            s = str(value)
    else:
        s = str(value)
    if len(s) > max_length:
        return s[:max_length] + f"\n... [truncated, {len(s)} chars total]"
    return s


# ══════════════════════════════════════════════════════════════════════════════
# BASE OPERATION
# ══════════════════════════════════════════════════════════════════════════════


class BaseOperation(ABC):
    """Base class for all epistemic operations.

    Operations are validated transformations that:
    1. Load entities from repository
    2. Run agents with adapter normalization
    3. Validate gate requirements
    4. Update entity state
    5. Check phase transitions

    Subclasses implement the execute() method.
    """

    entity_type: str = "unknown"

    def __init__(
        self,
        repo: "EpistemicRepository",
        agent_runner: Optional[AgentRunner] = None,
        validator: Optional[OperationValidator] = None,
        evidence_gatherer: Optional[EvidenceGatherer] = None,
        quality_scorer: Optional[QualityScorer] = None,
        embedding_model: Optional[str] = None,
    ):
        """Initialize operation.

        Args:
            repo: Repository for entity CRUD
            agent_runner: Optional agent execution protocol
            validator: Optional validation protocol
            evidence_gatherer: Optional evidence gathering protocol
            quality_scorer: Optional quality scoring protocol
            embedding_model: Embedding model for similarity/clustering operations.
        """
        self.repo = repo
        self.agent_runner = agent_runner
        self.validator = validator or DefaultValidator()
        self.evidence_gatherer = evidence_gatherer
        self.quality_scorer = quality_scorer
        self.embedding_model = embedding_model
        self._agent_calls: list[dict[str, Any]] = []

    @abstractmethod
    async def execute(self, work: OperationInput) -> OperationResult:
        """Execute the operation.

        Args:
            work: Operation input describing what to do

        Returns:
            OperationResult with success/failure status
        """
        ...

    async def run_agent(
        self,
        agent_name: str,
        **kwargs: Any,
    ) -> Any:
        """Run agent with adapter normalization.

        Args:
            agent_name: Name of the agent to run
            **kwargs: Arguments to pass to agent

        Returns:
            Adapted agent output
        """
        if not self.agent_runner:
            raise RuntimeError("No agent runner configured")

        raw = await self.agent_runner.run(agent_name, **kwargs)

        # Capture agent I/O for execution trace
        self._agent_calls.append(
            {
                "agent_name": agent_name,
                "input": {k: _truncate_for_trace(v) for k, v in kwargs.items()},
                "raw_output": _truncate_for_trace(raw),
            }
        )

        adapter = ADAPTERS.get(agent_name)
        if not adapter:
            return raw

        return adapter(raw)

    async def log_event(
        self,
        event_type: str,
        target_id: str,
        details: dict[str, Any],
    ) -> None:
        """Log an epistemic event.

        Args:
            event_type: Type of event (e.g., "claim_promoted")
            target_id: ID of the entity affected
            details: Additional event details
        """
        # For now, just log to the standard logger
        # In full implementation, would save EpistemicEvent entity
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"[{event_type}] {target_id}: {details}")
