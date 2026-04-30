"""Top-down decomposed orchestrator (Phase 3 of the unified architecture).

Wraps ``run_epistemic_graph`` with a coordinator that:

  1. Creates the parent objective in the database.
  2. Runs preplanning operations on the parent (clarify → classify →
     conceptual analysis), the same triple ``PrepareObjective`` runs.
  3. Runs ``DecomposeQuestionOperation`` to produce 2-5 sub-investigations.
  4. Runs ``SpawnSubObjectivesOperation`` to materialize one child Objective
     per sub-investigation. Each child is in seed_claim mode (its
     ``claim_to_verify`` is the sub-investigation's seed claim).
  5. Dispatches each child through the existing graph via
     ``run_epistemic_graph(objective_id=child_id, skip_preplanning=True)``.
  6. Combines child posteriors into a single combined verdict via
     ``combine_sub_verdicts`` honoring the decomposition's
     combination_rule (AND / OR / WEIGHTED_AND / UNION).

The graph itself is unchanged. Bypass paths still work: when
``decompose=False`` the orchestrator delegates straight to
``run_epistemic_graph`` (existing seed_claim or open-research run).

Design principle: each sub-objective is one independent graph run. State
between sub-investigations is intentionally not shared in Phase 3 — we
keep their evidence and reasoning isolated so the combination is over
verdicts, not internals. Phase 4-5 will add reflection-driven gap
detection across siblings.

Combination semantics for Phase 3 are conservative bounds, not joint
probabilities:
  * AND          → min of child posteriors (weakest-link bound)
  * OR           → max of child posteriors (best-evidence bound)
  * WEIGHTED_AND → mean of child posteriors (Phase 5 will add weights)
  * UNION        → posterior=None; combined view is set-collection
                   (Phase 5 will define semantics)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .confidence import PosteriorReport
from .graph.quarantine import QuarantineRecord
from .operations_runner import PipelineResult, ProgressCallback

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# COMBINATION HELPER
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class CombinedVerdict:
    """The aggregated outcome across sub-investigations.

    Carries enough to render a final answer for the parent question while
    keeping the per-child posteriors visible for diagnostics.
    """

    posterior: float | None
    verdict: str  # "supports" | "contradicts" | "insufficient" | "no_data"
    combination_rule: str
    child_posteriors: list[float | None]
    terminal_state: str  # "completed" | "retrieval_failed"
    explanation: str


def _verdict_label(p: float) -> str:
    """Map a combined posterior to a verbal verdict using the same
    breakpoints the per-claim integration uses."""
    if p > 0.66:
        return "supports"
    if p < 0.34:
        return "contradicts"
    return "insufficient"


def combine_sub_verdicts(
    child_results: list[PipelineResult], combination_rule: str
) -> CombinedVerdict:
    """Aggregate per-child posteriors into a combined verdict.

    Children whose ``posterior`` is None are excluded from the numeric
    combination but recorded in ``child_posteriors`` as None so the
    diagnostic remains complete.

    For AND / OR / WEIGHTED_AND, when no child contributed a numeric
    posterior the combined verdict is "no_data" with posterior=None. For
    UNION, posterior is intentionally None — Phase 5 will define
    set-collection semantics.

    If any child terminated with ``retrieval_failed``, the combined
    terminal_state is ``retrieval_failed``.
    """
    rule = combination_rule.upper()

    # Collect per-child posteriors and terminal-state propagation.
    child_posteriors: list[float | None] = []
    any_retrieval_failed = False
    for r in child_results:
        if r.posterior is None:
            child_posteriors.append(None)
            continue
        child_posteriors.append(r.posterior.posterior)
        if r.posterior.terminal_state == "retrieval_failed":
            any_retrieval_failed = True

    numeric = [p for p in child_posteriors if p is not None]
    terminal_state = "retrieval_failed" if any_retrieval_failed else "completed"

    if rule == "UNION":
        # Phase 5 will define set-collection semantics. For now we
        # surface a no-numeric-combination signal so callers can render
        # children individually.
        return CombinedVerdict(
            posterior=None,
            verdict="union",
            combination_rule="UNION",
            child_posteriors=child_posteriors,
            terminal_state=terminal_state,
            explanation=(
                "UNION combination is a Phase 5 stub. Render the "
                f"{len(child_posteriors)} child verdicts individually."
            ),
        )

    if not numeric:
        return CombinedVerdict(
            posterior=None,
            verdict="no_data",
            combination_rule=rule,
            child_posteriors=child_posteriors,
            terminal_state=terminal_state,
            explanation="No child produced a numeric posterior.",
        )

    if rule == "AND":
        combined = min(numeric)
        method = "min (weakest-link bound on conjunction)"
    elif rule == "OR":
        combined = max(numeric)
        method = "max (best-evidence bound on disjunction)"
    elif rule == "WEIGHTED_AND":
        combined = sum(numeric) / len(numeric)
        method = "mean (Phase 5 will add explicit weights)"
    else:
        # Unknown rule: be loud rather than silently picking a default.
        raise ValueError(
            f"Unknown combination_rule {combination_rule!r}; "
            "expected AND / OR / WEIGHTED_AND / UNION"
        )

    return CombinedVerdict(
        posterior=combined,
        verdict=_verdict_label(combined),
        combination_rule=rule,
        child_posteriors=child_posteriors,
        terminal_state=terminal_state,
        explanation=(
            f"{rule} over {len(numeric)} child posteriors via {method}: "
            f"{[round(p, 3) for p in numeric]} → {round(combined, 3)}"
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# DECOMPOSED PIPELINE RESULT
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DecomposedPipelineResult:
    """Result of a decomposed run.

    Wraps the per-child :class:`PipelineResult` instances and the combined
    verdict produced by ``combine_sub_verdicts``. The aggregate counters
    (``successful``, ``failed``, ``errors``, ``quarantined``) are sums
    over children so downstream code that doesn't care about
    decomposition can treat this like a single PipelineResult.
    """

    parent_objective_id: str
    sub_results: list[PipelineResult]
    combined: CombinedVerdict

    @property
    def successful(self) -> int:
        return sum(r.successful for r in self.sub_results)

    @property
    def failed(self) -> int:
        return sum(r.failed for r in self.sub_results)

    @property
    def errors(self) -> list[str]:
        out: list[str] = []
        for r in self.sub_results:
            out.extend(r.errors)
        return out

    @property
    def quarantined(self) -> list[QuarantineRecord]:
        out: list[QuarantineRecord] = []
        for r in self.sub_results:
            out.extend(r.quarantined)
        return out

    @property
    def status(self) -> str:
        return self.combined.verdict

    @property
    def success(self) -> bool:
        return self.successful > 0

    @property
    def posterior(self) -> Optional[PosteriorReport]:
        """Synthesize a parent-level PosteriorReport from the combined verdict.

        Returned shape mirrors per-child PosteriorReport so reporting code
        that consumes ``.posterior.posterior`` works uniformly. Returns
        None when the combiner produced no numeric posterior (UNION or
        all-children-missing).
        """
        if self.combined.posterior is None:
            return None
        # Pick a representative child to inherit question_type from.
        qt = "verificatory"
        for r in self.sub_results:
            if r.posterior is not None:
                qt = r.posterior.question_type
                break
        return PosteriorReport(
            posterior=self.combined.posterior,
            log_odds=0,
            supporting_count=0,
            contradicting_count=0,
            counting_posterior=self.combined.posterior,
            integration_verdict=self.combined.verdict,
            integration_confidence=None,
            mode="decomposed",
            terminal_state="retrieval_failed"
            if self.combined.terminal_state == "retrieval_failed"
            else "completed",
            objective_id=self.parent_objective_id,
            question_type=qt,
            explanation=self.combined.explanation,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════


# Type alias for the inner runner so tests can substitute a stub.
InnerRunner = Callable[..., Any]


async def run_research_question_decomposed(
    question: str,
    *,
    database_name: str = "epistemic_research",
    verbose: bool = False,
    model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
    provider: str = "all",
    providers: Optional[dict[str, Any]] = None,
    quality_scorer: Optional[Any] = None,
    db_dir: Optional[str] = None,
    decompose: bool = True,
    _inner_runner: Optional[InnerRunner] = None,
) -> DecomposedPipelineResult | PipelineResult:
    """Run a research question through top-down decomposition.

    When ``decompose=True`` (default):
        Creates a parent Objective, runs preplanning + decomposition +
        spawning, then runs each spawned child through
        ``run_epistemic_graph`` and combines the results.

    When ``decompose=False``:
        Delegates directly to ``run_epistemic_graph`` for an
        undecomposed run (returns PipelineResult unchanged).

    Bypass cases that return PipelineResult directly:
        - ``decompose=False``
        - decomposition produced no sub-investigations (empty children)
          — falls back to a single graph run on the parent

    Args:
        question: The research question.
        database_name, verbose, model, embedding_model, progress_callback,
        provider, providers, quality_scorer, db_dir: forwarded to
            ``run_epistemic_graph``.
        decompose: Set False to disable decomposition entirely.
        _inner_runner: Test-injection point for the inner graph runner.
            Defaults to ``run_epistemic_graph``.

    Returns:
        DecomposedPipelineResult on success; PipelineResult when the
        decomposition path is bypassed.
    """
    from andamentum.document_store import DocumentStore

    from .entities import Objective
    from .operations.preplanning import (
        ClarifyQuestionOperation,
        ClassifyQuestionOperation,
        ConceptualAnalysisOperation,
        DecomposeQuestionOperation,
        SpawnSubObjectivesOperation,
    )
    from .operations.base import OperationInput
    from .repository import EpistemicRepository
    from .runner import DefaultAgentRunner

    if _inner_runner is None:
        from .graph import run_epistemic_graph

        _inner_runner = run_epistemic_graph

    if not decompose:
        # Bypass path: caller explicitly opted out of decomposition.
        return await _inner_runner(
            question=question,
            database_name=database_name,
            verbose=verbose,
            model=model,
            embedding_model=embedding_model,
            progress_callback=progress_callback,
            provider=provider,
            providers=providers,
            quality_scorer=quality_scorer,
            db_dir=db_dir,
        )

    if not model:
        raise ValueError(
            "model is required for run_research_question_decomposed. "
            "Pass model= or set ANDAMENTUM_MAIN_LLM_MODEL."
        )

    # Set up repo + agent runner ONCE for the parent.
    store = DocumentStore.for_database(database_name, db_dir=db_dir)
    await store.initialize()
    repo = EpistemicRepository(store)
    agent_runner = DefaultAgentRunner(model=model)

    # Resolve embedding model now so child calls can pass the same string.
    if not embedding_model:
        from andamentum.core.models import resolve_embedding_model_from_args

        embedding_model = resolve_embedding_model_from_args()

    # Create (or resume) the parent objective.
    existing = await repo.query("objective")
    parent_objective = None
    for obj in existing:
        if obj.parent_objective_id is None and obj.claim_to_verify is None:
            parent_objective = obj
            break
    if parent_objective is None:
        oid = f"obj_{uuid.uuid4().hex[:12]}"
        parent_objective = Objective(
            entity_id=oid,
            objective_id=oid,
            description=question,
            phase="new",
        )
        await repo.save(parent_objective)
    parent_id = parent_objective.entity_id

    if verbose:
        logger.info("Decomposed run on parent objective %s", parent_id)

    # Helper to instantiate and run a preplanning op on the parent.
    async def _run(op_class: type, operation: str) -> Any:
        op = op_class(
            repo=repo,
            agent_runner=agent_runner,
            embedding_model=embedding_model,
        )
        return await op.execute(
            OperationInput(
                entity_id=parent_id,
                entity_type="objective",
                operation=operation,
            )
        )

    # 1. Preplanning on the parent.
    await _run(ClarifyQuestionOperation, "clarify_question")
    await _run(ClassifyQuestionOperation, "classify_question")
    await _run(ConceptualAnalysisOperation, "conceptual_analysis")

    # 2. Decompose. If the agent returns no decomposition (e.g. seed_claim
    # mode bypass — shouldn't happen here since we created the parent
    # without claim_to_verify, but be defensive), fall back to a single
    # undecomposed graph run.
    decompose_result = await _run(DecomposeQuestionOperation, "decompose_question")
    parent_objective = await repo.get("objective", parent_id)
    if parent_objective.decomposition is None:
        if verbose:
            logger.info(
                "No decomposition produced (%s); falling back to undecomposed run",
                decompose_result.message,
            )
        return await _inner_runner(
            question=question,
            database_name=database_name,
            verbose=verbose,
            model=model,
            embedding_model=embedding_model,
            progress_callback=progress_callback,
            provider=provider,
            providers=providers,
            quality_scorer=quality_scorer,
            db_dir=db_dir,
            objective_id=parent_id,
        )

    # 3. Spawn sub-objectives.
    await _run(SpawnSubObjectivesOperation, "spawn_sub_objectives")
    parent_objective = await repo.get("objective", parent_id)
    if not parent_objective.sub_objective_ids:
        if verbose:
            logger.info(
                "Decomposition spawned no children; falling back to undecomposed run"
            )
        return await _inner_runner(
            question=question,
            database_name=database_name,
            verbose=verbose,
            model=model,
            embedding_model=embedding_model,
            progress_callback=progress_callback,
            provider=provider,
            providers=providers,
            quality_scorer=quality_scorer,
            db_dir=db_dir,
            objective_id=parent_id,
        )

    # 4. Run the graph on each child. Children are in seed_claim mode and
    # phase=analyzed, so PrepareObjective runs the no-op preplanning ops
    # and CreateClaims uses SeedClaimOperation. We pass skip_preplanning
    # for clarity; the per-op idempotence guards would catch it anyway.
    sub_results: list[PipelineResult] = []
    for child_id in parent_objective.sub_objective_ids:
        if verbose:
            logger.info("Running graph on sub-objective %s", child_id)
        child_result = await _inner_runner(
            question=question,  # not used when objective_id is set
            database_name=database_name,
            verbose=verbose,
            skip_preplanning=True,
            model=model,
            embedding_model=embedding_model,
            progress_callback=progress_callback,
            provider=provider,
            providers=providers,
            quality_scorer=quality_scorer,
            db_dir=db_dir,
            objective_id=child_id,
        )
        sub_results.append(child_result)

    # 5. Combine.
    rule = parent_objective.combination_rule or "AND"
    combined = combine_sub_verdicts(sub_results, rule)
    if verbose:
        logger.info(
            "Combined verdict over %d children (%s): %s",
            len(sub_results),
            rule,
            combined.explanation,
        )

    return DecomposedPipelineResult(
        parent_objective_id=parent_id,
        sub_results=sub_results,
        combined=combined,
    )


__all__ = [
    "CombinedVerdict",
    "DecomposedPipelineResult",
    "combine_sub_verdicts",
    "run_research_question_decomposed",
]
