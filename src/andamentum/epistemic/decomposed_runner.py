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

Combination semantics are conservative bounds, not joint probabilities:
  * AND          → min of child posteriors (weakest-link bound)
  * OR           → max of child posteriors (best-evidence bound)
  * WEIGHTED_AND → weighted mean of child posteriors using per-sub
                   weights from the decomposition. With all-equal weights
                   this degenerates to a simple mean. Decomposer guidance
                   (in DECOMPOSE_QUESTION_PROMPT): only assign non-equal
                   weights when there is genuine differential importance;
                   otherwise pick AND or OR.
  * UNION        → posterior=None (no scalar verdict); the combined view
                   is the set of children's verdicts. Used for
                   exploratory / compositional questions where each
                   sub-investigation contributes a facet rather than a
                   value to be averaged. Renderers / callers iterate
                   ``DecomposedPipelineResult.sub_results`` for the full
                   structured answer.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

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
    child_results: list[PipelineResult],
    combination_rule: str,
    weights: list[float] | None = None,
) -> CombinedVerdict:
    """Aggregate per-child posteriors into a combined verdict.

    Children whose ``posterior`` is None are excluded from the numeric
    combination but recorded in ``child_posteriors`` as None so the
    diagnostic remains complete.

    For AND / OR / WEIGHTED_AND, when no child contributed a numeric
    posterior the combined verdict is "no_data" with posterior=None. For
    UNION, posterior is None by design (set-collection semantics).

    If any child terminated with ``retrieval_failed``, the combined
    terminal_state is ``retrieval_failed``.

    Args:
        child_results: per-child PipelineResults in decomposition order.
        combination_rule: AND / OR / WEIGHTED_AND / UNION (case-insensitive).
        weights: per-child weights, same length as ``child_results``, only
            consumed by WEIGHTED_AND. None or all-equal weights make
            WEIGHTED_AND degenerate to a simple mean. Weights for
            children with posterior=None are dropped from the
            normalization. Negative weights raise ValueError.
    """
    rule = combination_rule.upper()

    # Collect per-child posteriors and terminal-state propagation.
    # Children with terminal_state in {retrieval_failed,
    # oscillation_detected} contribute their numeric posterior (0.5) to
    # the diagnostic ``child_posteriors`` list but are excluded from the
    # numeric combination — both states represent uninformative outcomes.
    # Terminal-state precedence: retrieval_failed (no evidence at all)
    # outranks oscillation_detected (evidence but no convergence) when
    # multiple children fail differently. Either still beats completed.
    child_posteriors: list[float | None] = []
    any_retrieval_failed = False
    any_oscillation = False
    excluded_indices: set[int] = set()
    for idx, r in enumerate(child_results):
        if r.posterior is None:
            child_posteriors.append(None)
            excluded_indices.add(idx)
            continue
        child_posteriors.append(r.posterior.posterior)
        if r.posterior.terminal_state == "retrieval_failed":
            any_retrieval_failed = True
            excluded_indices.add(idx)
        elif r.posterior.terminal_state == "oscillation_detected":
            any_oscillation = True
            excluded_indices.add(idx)

    numeric = [
        p
        for i, p in enumerate(child_posteriors)
        if p is not None and i not in excluded_indices
    ]
    if any_retrieval_failed:
        terminal_state = "retrieval_failed"
    elif any_oscillation:
        terminal_state = "oscillation_detected"
    else:
        terminal_state = "completed"

    if rule == "UNION":
        # Set-collection semantics: each child contributes a facet rather
        # than a value to be averaged. The scalar posterior is None by
        # design; callers iterate sub_results for the full answer.
        return CombinedVerdict(
            posterior=None,
            verdict="union",
            combination_rule="UNION",
            child_posteriors=child_posteriors,
            terminal_state=terminal_state,
            explanation=(
                f"UNION over {len(child_posteriors)} children: render each "
                "child's findings individually; there is no scalar verdict."
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
        combined, method = _weighted_mean(child_posteriors, weights)
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


def _weighted_mean(
    child_posteriors: list[float | None], weights: list[float] | None
) -> tuple[float, str]:
    """Compute a weighted mean over the numeric child posteriors.

    Children with posterior=None are dropped along with their weight.
    If weights is None, falls back to a simple mean. If all weights for
    numeric children are zero, also falls back to a simple mean (the
    decomposer signaled no preference).

    Returns (combined, method-description-for-explanation).
    """
    if weights is None:
        numeric = [p for p in child_posteriors if p is not None]
        return sum(numeric) / len(numeric), "mean (no weights provided)"

    if len(weights) != len(child_posteriors):
        raise ValueError(
            f"weights length {len(weights)} does not match "
            f"child_results length {len(child_posteriors)}"
        )
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")

    paired = [
        (p, w)
        for p, w in zip(child_posteriors, weights, strict=True)
        if p is not None
    ]
    weight_sum = sum(w for _, w in paired)
    if weight_sum == 0.0:
        # Decomposer assigned all-zero weights to numeric children. Fall
        # back to a simple mean rather than dividing by zero.
        numeric = [p for p, _ in paired]
        return sum(numeric) / len(numeric), "mean (all weights zero)"

    weighted = sum(p * w for p, w in paired) / weight_sum
    return (
        weighted,
        f"weighted mean (weights={[round(w, 2) for _, w in paired]})",
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
        # Propagate the terminal_state from the combiner so callers can
        # distinguish converged decomposed answers from "one or more
        # children failed retrieval / hit the cycle cap". The Literal
        # narrowing satisfies pydantic's PosteriorReport schema.
        ts = self.combined.terminal_state
        if ts == "retrieval_failed":
            propagated_ts: Literal[
                "completed", "retrieval_failed", "oscillation_detected"
            ] = "retrieval_failed"
        elif ts == "oscillation_detected":
            propagated_ts = "oscillation_detected"
        else:
            propagated_ts = "completed"
        return PosteriorReport(
            posterior=self.combined.posterior,
            log_odds=0,
            supporting_count=0,
            contradicting_count=0,
            counting_posterior=self.combined.posterior,
            integration_verdict=self.combined.verdict,
            integration_confidence=None,
            mode="decomposed",
            terminal_state=propagated_ts,
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
    max_reflection_rounds: int = 1,
    _inner_runner: Optional[InnerRunner] = None,
) -> DecomposedPipelineResult | PipelineResult:
    """Run a research question through top-down decomposition.

    When ``decompose=True`` (default):
        Creates a parent Objective, runs preplanning + decomposition +
        spawning, then runs each spawned child through
        ``run_epistemic_graph`` and combines the results. Optionally
        runs a corrective reflection loop (Phase 4) after combination.

    When ``decompose=False``:
        Delegates directly to ``run_epistemic_graph`` for an
        undecomposed run (returns PipelineResult unchanged).

    Bypass cases that return PipelineResult directly:
        - ``decompose=False``
        - decomposition produced no sub-investigations (empty children)
          — falls back to a single graph run on the parent

    Reflection loop (Phase 4):
        After the initial combine, while ``reflection_rounds <
        max_reflection_rounds`` AND the verdict is non-decisive
        (combined.verdict in {insufficient, no_data} OR
        terminal_state=retrieval_failed), runs
        ReflectOnGapsOperation. The op may declare sufficiency (loop
        exits) or append new sub-investigations to the decomposition.
        If new sub-investigations were added, the orchestrator delta-
        spawns them, runs the new children, and re-combines.

    Args:
        question: The research question.
        database_name, verbose, model, embedding_model, progress_callback,
        provider, providers, quality_scorer, db_dir: forwarded to
            ``run_epistemic_graph``.
        decompose: Set False to disable decomposition entirely.
        max_reflection_rounds: Cap on corrective reflection rounds.
            Default 1 (one corrective pass max). Set 0 to disable
            reflection entirely.
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
        ReflectOnGapsOperation,
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

    async def _run_unrun_children_and_combine() -> CombinedVerdict:
        """Run any spawned-but-unrun children and re-combine. Reused by
        the initial pass and by each reflection round."""
        nonlocal sub_results
        latest_parent = await repo.get("objective", parent_id)
        already_run = {r.objective_id for r in sub_results}
        for child_id in latest_parent.sub_objective_ids:
            if child_id in already_run:
                continue
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
        rule = latest_parent.combination_rule or "AND"
        weights = _extract_weights(latest_parent, len(sub_results))
        return combine_sub_verdicts(sub_results, rule, weights=weights)

    combined = await _run_unrun_children_and_combine()
    if verbose:
        logger.info(
            "Combined verdict over %d children: %s",
            len(sub_results),
            combined.explanation,
        )

    # 6. Reflection loop (Phase 4). Trigger only when the combined
    # verdict is non-decisive — clear supports/contradicts skip
    # reflection. Cap at max_reflection_rounds (default 1).
    for _ in range(max_reflection_rounds):
        needs_reflection = (
            combined.verdict in ("insufficient", "no_data")
            or combined.terminal_state == "retrieval_failed"
        )
        if not needs_reflection:
            break
        if verbose:
            logger.info(
                "Reflecting on gaps (verdict=%s, terminal=%s)",
                combined.verdict,
                combined.terminal_state,
            )
        reflection_result = await _run(ReflectOnGapsOperation, "reflect_on_gaps")
        if not reflection_result.success or not reflection_result.did_work:
            # Agent declared sufficiency, proposed no additions, or the op
            # failed. Either way, no new children to run.
            if verbose:
                logger.info("Reflection terminated: %s", reflection_result.message)
            break
        # New sub-investigations were appended to parent.decomposition.
        # Delta-spawn fills in the missing children.
        await _run(SpawnSubObjectivesOperation, "spawn_sub_objectives")
        combined = await _run_unrun_children_and_combine()
        if verbose:
            logger.info(
                "Post-reflection combined verdict over %d children: %s",
                len(sub_results),
                combined.explanation,
            )

    return DecomposedPipelineResult(
        parent_objective_id=parent_id,
        sub_results=sub_results,
        combined=combined,
    )


def _extract_weights(parent_objective: Any, expected_count: int) -> list[float] | None:
    """Pull per-sub weights from the parent's decomposition in spawn order.

    Returns None if the decomposition is missing weights or its length
    doesn't match the spawned children — the combiner then falls back to
    an unweighted mean for WEIGHTED_AND. The shape mismatch case is
    permissive on purpose: a future reflection round may grow the
    decomposition, and a length-aware fallback is safer than crashing
    here.
    """
    decomposition = getattr(parent_objective, "decomposition", None)
    if not decomposition:
        return None
    subs = decomposition.get("sub_investigations") or []
    if len(subs) != expected_count:
        return None
    weights = [float(s.get("weight", 1.0)) for s in subs]
    return weights


__all__ = [
    "CombinedVerdict",
    "DecomposedPipelineResult",
    "combine_sub_verdicts",
    "run_research_question_decomposed",
]
