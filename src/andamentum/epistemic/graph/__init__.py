"""Epistemic pipeline as a pydantic-graph DAG.

Replaces the pattern-based scheduler with explicit node dependencies.
Every scheduling decision is a typed return value, not a pattern match.
The graph makes the workflow visible, testable, and provably correct.

Usage::

    from andamentum.epistemic.graph import run_epistemic_graph

    result = await run_epistemic_graph(
        question="Does metformin reduce cancer risk?",
        database_name="my_research",
        model="openai:gpt-4o-mini",
    )
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def run_epistemic_graph(
    question: str,
    database_name: str = "epistemic_research",
    verbose: bool = False,
    skip_preplanning: bool = False,
    model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    progress_callback: Optional[Any] = None,
    provider: str = "all",
    providers: Optional[dict[str, Any]] = None,
    quality_scorer: Optional[Any] = None,
    db_dir: Optional[str] = None,
    objective_id: Optional[str] = None,
    decompose: bool = False,
    stop_after: Optional[type] = None,
) -> Any:
    """Run a research question through the epistemic graph pipeline.

    Same interface as the old run_research_question, but uses a
    pydantic-graph DAG instead of the pattern scheduler.

    Args:
        objective_id: Target a specific existing objective (e.g. a
            sub-objective spawned by the decomposed runner). When None,
            falls back to resume-first-objective-or-create behaviour
            from ``question``.

    Returns:
        PipelineResult (backward compatible)
    """
    from andamentum.document_store import DocumentStore

    from ..entities import Objective
    from ..evidence_gathering import get_default_gatherer
    from ..repository import EpistemicRepository
    from ..runner import DefaultAgentRunner
    from .deps import EpistemicDeps
    from .nodes import PrepareObjective, epistemic_graph
    from .state import EpistemicGraphState

    # Initialize database and repository
    store = DocumentStore.for_database(database_name, db_dir=db_dir)
    await store.initialize()
    repo = EpistemicRepository(store)

    if objective_id is not None:
        # Targeted run: load the specified objective directly. Used by the
        # decomposed runner to dispatch each spawned sub-objective through
        # the graph without colliding on the resume-first-objective rule.
        target = await repo.get("objective", objective_id)
        if target is None:
            raise ValueError(f"Objective {objective_id} not found in {database_name}")
        if verbose:
            phase = getattr(target, "phase", "unknown")
            logger.info(f"Targeted objective: {objective_id} (phase={phase})")
    else:
        # Resume if an objective already exists, otherwise create one
        existing_objectives = await repo.query("objective")
        if existing_objectives:
            objective_id = existing_objectives[0].objective_id
            if verbose:
                phase = getattr(existing_objectives[0], "phase", "unknown")
                logger.info(f"Resuming objective: {objective_id} (phase={phase})")
        else:
            objective_id = f"obj_{uuid.uuid4().hex[:12]}"
            starting_phase = "analyzed" if skip_preplanning else "new"
            objective = Objective(
                entity_id=objective_id,
                objective_id=objective_id,
                description=question,
                phase=starting_phase,
            )
            await repo.save(objective)
            if verbose:
                logger.info(
                    f"Created objective: {objective_id} (phase={starting_phase})"
                )

    # Create agent runner and evidence gatherer
    if not model:
        raise ValueError(
            "model is required for run_epistemic_graph. "
            "Pass --model or set ANDAMENTUM_MAIN_LLM_MODEL."
        )
    if not embedding_model:
        from andamentum.core.models import resolve_embedding_model_from_args

        embedding_model = resolve_embedding_model_from_args()
    agent_runner = DefaultAgentRunner(model=model)

    # Auto-load providers
    if providers is None and provider == "all":
        from ..providers import get_biomedical_providers

        providers = get_biomedical_providers()

    evidence_gatherer = (
        get_default_gatherer(
            model=model, providers=providers, embedding_model=embedding_model
        )
        if model
        else None
    )

    # Build graph state and deps
    state = EpistemicGraphState(
        objective_id=objective_id,
        question=question,
        skip_preplanning=skip_preplanning,
        decompose=decompose,
    )
    deps = EpistemicDeps(
        repo=repo,
        agent_runner=agent_runner,
        evidence_gatherer=evidence_gatherer,
        quality_scorer=quality_scorer,
        embedding_model=embedding_model,
        provider=provider,
        verbose=verbose,
        progress_callback=progress_callback,
    )

    # Run the graph
    if verbose:
        logger.info(f"Starting epistemic graph for objective {objective_id}")

    if stop_after is None:
        graph_result = await epistemic_graph.run(
            PrepareObjective(),
            state=state,
            deps=deps,
        )
        result = graph_result.output
    else:
        # Stage-runner mode: drive the graph node-by-node and break
        # after the requested node's run() completes. The DB is the
        # checkpoint; callers resume by passing start_at on a later
        # invocation. See docs/superpowers/plans/2026-05-03-stage-runners.md.
        async with epistemic_graph.iter(
            PrepareObjective(), state=state, deps=deps
        ) as run:
            while run.result is None:
                next_node = run.next_node
                if next_node is None:
                    break
                node_class = type(next_node)
                await run.next()
                if node_class is stop_after:
                    break
        result = run.result.output if run.result is not None else None

    if verbose:
        if result is not None:
            logger.info(
                f"Graph complete: {result.successful} successful, {result.failed} failed"
            )
        else:
            assert stop_after is not None
            logger.info(f"Graph stopped after {stop_after.__name__}")

    # Compute posterior confidence only on a complete graph traversal.
    posterior_report = None
    if result is not None and result.successful > 0:
        from ..confidence import compute_posterior

        posterior_report = await compute_posterior(
            repo, objective_id, retrieval_failed=result.retrieval_failed
        )

    from ..operations_runner import PipelineResult

    if result is None:
        assert stop_after is not None
        return PipelineResult(
            objective_id=objective_id,
            iterations=len(state.operations_log),
            successful=0,
            failed=0,
            status=f"stopped_after:{stop_after.__name__}",
            errors=[],
            posterior=None,
            quarantined=[],
            retrieval_failed=False,
        )
    return PipelineResult(
        objective_id=objective_id,
        iterations=len(state.operations_log),
        successful=result.successful,
        failed=result.failed,
        status=result.status,
        errors=result.errors,
        posterior=posterior_report,
        quarantined=result.quarantined,
        retrieval_failed=result.retrieval_failed,
    )
