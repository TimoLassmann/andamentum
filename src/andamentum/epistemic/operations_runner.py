"""Operations Runner - Entry point for epistemic operations.

Delegates to the pydantic-graph DAG scheduler in
``andamentum.epistemic.graph``. The graph makes operation dependencies
explicit and type-checked, replacing the old pattern-based scheduler.

Architecture: Layer 2 (pydantic-graph)
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .confidence import PosteriorReport

from .graph.quarantine import QuarantineRecord

# Type for progress callback: (operation_type, workitem_id, success, message, outputs) -> None
ProgressCallback = Callable[[str, str, bool, str, dict[str, Any]], None]

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULER RESULT
# ══════════════════════════════════════════════════════════════════════════════


class PipelineResult:
    """Result from an epistemic pipeline run.

    Provides both graph-scheduler native fields and compatibility
    properties used by CLI handlers.
    """

    def __init__(
        self,
        objective_id: str,
        iterations: int,
        successful: int,
        failed: int,
        status: str,
        errors: Optional[list[str]] = None,
        posterior: Optional["PosteriorReport"] = None,
        quarantined: Optional[list[QuarantineRecord]] = None,
    ):
        self.objective_id = objective_id
        self.iterations = iterations
        self.successful = successful
        self.failed = failed
        self.status = status
        self.errors = errors or []
        self.posterior = posterior
        self.quarantined: list[QuarantineRecord] = quarantined or []

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
    operation dependencies explicit and type-checked, replacing the
    pattern-based scheduler.

    Full pipeline:
        PrepareObjective -> PlanEvidence -> ExtractEvidence ->
        CreateClaims -> Scrutinize -> [investigation cycle] ->
        PromoteToSupported -> RunVerification -> ResolveUncertainties ->
        IntegrateEvidence -> PromoteSupported -> Synthesize

    Args:
        question: The research question to investigate
        database_name: Name of the database to use
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
