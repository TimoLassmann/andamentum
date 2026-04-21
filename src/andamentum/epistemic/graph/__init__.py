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
) -> Any:
    """Run a research question through the epistemic graph pipeline.

    Same interface as the old run_research_question, but uses a
    pydantic-graph DAG instead of the pattern scheduler.

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
            logger.info(f"Created objective: {objective_id} (phase={starting_phase})")

    # Create agent runner and evidence gatherer
    if not model:
        raise ValueError(
            "model is required for run_epistemic_graph. "
            "Pass --model or set ANDAMENTUM_MAIN_LLM_MODEL."
        )
    if not embedding_model:
        import os as _os

        embedding_model = _os.environ.get(
            "ANDAMENTUM_EMBEDDING_MODEL", "embeddinggemma:latest"
        )
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

    graph_result = await epistemic_graph.run(
        PrepareObjective(),
        state=state,
        deps=deps,
    )

    result = graph_result.output

    if verbose:
        logger.info(
            f"Graph complete: {result.successful} successful, {result.failed} failed"
        )

    # Compute posterior confidence (deterministic, no LLM).
    # No fallback: if posterior raises, the caller sees the real error.
    posterior_report = None
    if result.successful > 0:
        from ..confidence import compute_posterior

        posterior_report = await compute_posterior(repo, objective_id)

    # Return backward-compatible result
    from ..operations_runner import PipelineResult

    return PipelineResult(
        objective_id=objective_id,
        iterations=len(state.operations_log),
        successful=result.successful,
        failed=result.failed,
        status=result.status,
        errors=result.errors,
        posterior=posterior_report,
        quarantined=result.quarantined,
    )
