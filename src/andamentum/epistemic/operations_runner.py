"""Operations Runner - Entry point for epistemic operations.

Delegates to the pydantic-graph DAG scheduler in
``andamentum.epistemic.graph``. The graph makes operation dependencies
explicit and type-checked, replacing the old pattern-based scheduler.

Architecture: Layer 2 (pydantic-graph)
"""

import logging
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .confidence import PosteriorReport
from .graph.quarantine import QuarantineRecord

# Type for progress callback: (operation_type, workitem_id, success, message, outputs) -> None
ProgressCallback = Callable[[str, str, bool, str, dict[str, Any]], None]

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULER RESULT
# ══════════════════════════════════════════════════════════════════════════════


class PipelineResult(BaseModel):
    """Result from an epistemic pipeline run.

    The single typed exit of ``run_epistemic_graph`` / ``run_research_question``
    (dialect Law 9). Provides graph-scheduler native fields plus the
    ``success`` predicate used by CLI handlers.
    """

    # PosteriorReport is a Pydantic model; QuarantineRecord too. No arbitrary
    # types needed — but the field defaults use factories, so keep validation on.
    model_config = ConfigDict(arbitrary_types_allowed=False)

    objective_id: str
    iterations: int
    successful: int
    failed: int
    status: str
    errors: list[str] = Field(default_factory=list)
    posterior: Optional[PosteriorReport] = None
    quarantined: list[QuarantineRecord] = Field(default_factory=list)
    retrieval_failed: bool = False

    @property
    def success(self) -> bool:
        """Whether the scheduler produced useful results.

        A run succeeds if it completed at least one operation successfully.
        Individual operation failures (e.g., agent rate limits, inapplicable
        verification) are expected noise in a multi-agent pipeline — they
        don't invalidate the findings that were produced.
        """
        return self.successful > 0


# ══════════════════════════════════════════════════════════════════════════════
# RESEARCH RUNNER
# ══════════════════════════════════════════════════════════════════════════════


async def run_research_question(
    question: str,
    database_name: str = "epistemic_research",
    mode: Literal["verify", "research"] = "research",
    verbose: bool = False,
    skip_preplanning: bool = False,
    model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
    provider: str = "all",
    providers: Optional[dict[str, Any]] = None,
    quality_scorer: Optional[Any] = None,
    db_dir: Optional[str] = None,
) -> PipelineResult:
    """Run a research question through the epistemic pipeline.

    Delegates to the pydantic-graph DAG scheduler. The graph makes
    operation dependencies explicit and type-checked.

    Two modes (``mode`` parameter):

    * ``"research"`` (default): ``question`` is a research question. The
      graph attempts decomposition; if the decomposer produces no usable
      sub-investigations, the ``MultiSeedClaim → ProposeClaims`` fallback
      in ``CreateClaims`` routes to the open-research path.
    * ``"verify"``: ``question`` is a single claim to verify (SciFact-
      style). The graph skips decomposition and seeds exactly one Claim
      from the user-provided text.

    Full pipeline:
        PrepareObjective -> Decompose -> PlanEvidence -> ExtractEvidence
        -> CreateClaims -> Scrutinize -> [investigation cycle] ->
        PromoteToSupported -> RunVerification -> ResolveUncertainties ->
        IntegrateEvidence -> PromoteSupported -> Synthesize

    Args:
        question: The research question or (in verify mode) the claim text
        database_name: Name of the database to use
        mode: "research" attempts decomposition; "verify" treats `question`
            as a single claim and skips decomposition.
        verbose: Print detailed output
        skip_preplanning: Skip CLARIFY_QUESTION and CONCEPTUAL_ANALYSIS
        model: Optional LLM model to use (e.g., "openai:gpt-4o-mini")
        progress_callback: Optional callback for progress updates.
            Signature: (operation_type, entity_id, success, message, outputs) -> None
        provider: Evidence provider to use: "web_search" or "all"
        providers: Optional dict of named EvidenceProvider instances to inject.
            Overrides the default provider setup when given.
        quality_scorer: Optional QualityScorer instance for evidence quality scoring.
        db_dir: Custom directory for database files. When provided, the database
            is written here instead of ~/.config/andamentum/databases/.

    Returns:
        PipelineResult with execution summary
    """
    from .graph import run_epistemic_graph

    return await run_epistemic_graph(
        question=question,
        database_name=database_name,
        mode=mode,
        verbose=verbose,
        skip_preplanning=skip_preplanning,
        model=model,
        embedding_model=embedding_model,
        progress_callback=progress_callback,
        provider=provider,
        providers=providers,
        quality_scorer=quality_scorer,
        db_dir=db_dir,
    )
