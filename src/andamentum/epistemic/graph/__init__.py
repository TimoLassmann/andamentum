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

import json
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from ..operations_runner import PipelineResult

logger = logging.getLogger(__name__)


class DegenerateQuestionError(ValueError):
    """Raised when the question can't plausibly be a research question.

    K5 from the 2026-05-03 freeze sheet: ``--question Q`` was accepted
    silently, producing 4 sub-investigations of nothing meaningful and
    burning API budget on a degenerate decomposition. Loud refusal at
    the entry point prevents that whole class of accidental run.

    The validator is intentionally permissive — it rejects what is
    obviously NOT a research question, not what is suboptimal. Borderline
    inputs ("Why?", "Is X?") are accepted; the user is the one judging
    whether their question is good enough to investigate. We just refuse
    obvious garbage.
    """


def _validate_research_question(question: str) -> None:
    """Reject obviously degenerate research questions.

    Rules (intentionally minimal):
      * non-empty after stripping whitespace
      * at least 10 characters
      * at least 2 words

    A "real" research question almost always exceeds all three; "Q",
    "?", "x y" all fail at least one. Borderline inputs like "Is X
    safe?" pass — the system isn't a question-quality grader, it's a
    guard against accidentally typing a single character into the CLI
    and getting a 4-investigation decomposition out the other side.
    """
    stripped = (question or "").strip()
    if not stripped:
        raise DegenerateQuestionError(
            "Question is empty. Pass a real research question."
        )
    if len(stripped) < 10:
        raise DegenerateQuestionError(
            f"Question too short ({len(stripped)} chars; need ≥10): {stripped!r}. "
            "Pass a real research question, not a placeholder."
        )
    if len(stripped.split()) < 2:
        raise DegenerateQuestionError(
            f"Question is a single token: {stripped!r}. "
            "A research question needs at least a subject and a predicate."
        )


async def _check_stage_invariant(
    exit_node: type,
    state: Any,
    repo: Any,
) -> None:
    """If ``exit_node`` is a known stage exit, enforce that stage's
    invariant. Failing invariant = loud crash, by design — see
    docs/superpowers/plans/2026-05-03-stage-runners.md (the
    "unforeseen edge cases" mitigation)."""
    from .stages import StageInvariantError, stage_for_exit_node

    stage = stage_for_exit_node(exit_node)
    if stage is None:
        return
    ok = await stage.exit_invariant(state, repo)
    if not ok:
        raise StageInvariantError(
            f"Stage {stage.name!r} exited at {exit_node.__name__} but "
            "its invariant is unsatisfied. The boundary is leaky — "
            "downstream stages would inherit half-finished work. See "
            f"src/andamentum/epistemic/graph/stages.py for the contract."
        )


async def _emit_artifacts(
    output_dir: Path,
    visits: list[dict[str, Any]],
    repo: Any,
    objective_id: str,
) -> None:
    """Write run.jsonl, diff.json, timing.txt to output_dir.

    The DB at the end of the run is the canonical state — these files
    are derived views of it, intended to be greppable, diffable, and
    committable as test fixtures. See
    docs/superpowers/plans/2026-05-03-stage-runners.md.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "run.jsonl").write_text(
        "\n".join(json.dumps(v) for v in visits) + "\n"
    )

    objs = await repo.query("objective")
    claims = await repo.query("claim")
    evidence = await repo.query("evidence")
    obj = next((o for o in objs if o.objective_id == objective_id), None)
    decomp = obj.decomposition if obj is not None else None
    cv = decomp.combined_verdict if decomp is not None else None
    diff = {
        "objective_id": objective_id,
        "claims": len(claims),
        "evidence": len(evidence),
        "evidence_with_content": sum(
            1
            for e in evidence
            if (e.extracted_content or "") and len(e.extracted_content) > 200
        ),
        "decomposition_present": decomp is not None,
        "combined_verdict": cv.verdict if cv is not None else None,
        "combined_posterior": cv.posterior if cv is not None else None,
        "claims_terminal": sum(1 for c in claims if c.cycle_capped or c.abandoned),
        "claims_with_integrated_assessment": sum(
            1 for c in claims if c.integrated_assessment is not None
        ),
    }
    (output_dir / "diff.json").write_text(json.dumps(diff, indent=2) + "\n")

    by_node: dict[str, float] = {}
    for v in visits:
        by_node[v["node"]] = by_node.get(v["node"], 0.0) + v["ms"]
    total_ms = sum(v["ms"] for v in visits)
    lines = [f"Total: {total_ms / 1000:.2f}s ({len(visits)} node visits)\n"]
    for n, ms in sorted(by_node.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {n}: {ms / 1000:.2f}s\n")
    (output_dir / "timing.txt").write_text("".join(lines))


Mode = Literal["verify", "research"]


async def run_epistemic_graph(
    question: str,
    database_name: str = "epistemic_research",
    mode: Mode = "research",
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
    stop_after: Optional[type] = None,
    start_at: Optional[type] = None,
    output_dir: Optional[Path] = None,
    ibe_agreement_k: Optional[int] = None,
) -> "PipelineResult":
    """Run a research question through the epistemic graph pipeline.

    Two modes, picked by ``mode``:

    * ``"research"`` (default): ``question`` is a research question. The
      graph attempts decomposition; if the decomposer produces no usable
      sub-investigations, the ``MultiSeedClaim → ProposeClaims`` fallback
      in ``CreateClaims`` routes to the open-research path. Either way,
      the same downstream verification pipeline runs.
    * ``"verify"``: ``question`` is a single claim to verify (SciFact-
      style). The graph skips decomposition and seeds exactly one Claim
      from the user-provided text via ``SeedClaim``.

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
            # Fresh objective from the question — this is the only path
            # where the question text is used. K5 guard fires here so
            # resume / targeted runs (which ignore the question) aren't
            # blocked by the validator. Stage-runner callers passing
            # "(resumed)" never reach this branch when the DB has an
            # existing objective; they only reach it on a fresh DB,
            # where the placeholder string would correctly fail.
            _validate_research_question(question)
            objective_id = f"obj_{uuid.uuid4().hex[:12]}"
            starting_phase = "analyzed" if skip_preplanning else "new"
            objective = Objective(
                entity_id=objective_id,
                objective_id=objective_id,
                description=question,
                phase=starting_phase,
                # mode="verify" seeds claim_to_verify so the Decompose node
                # (graph/nodes.py) skips and CreateClaims routes to
                # SeedClaim. mode="research" leaves it None; Decompose
                # runs and CreateClaims routes to MultiSeedClaim, with
                # the empty-decomposition fallback to ProposeClaims.
                claim_to_verify=question if mode == "verify" else None,
            )
            await repo.save(objective)
            if verbose:
                logger.info(
                    f"Created objective: {objective_id} (mode={mode}, "
                    f"phase={starting_phase})"
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

    # Auto-load providers. "all" includes web_search (general-domain) alongside
    # the specialist biomedical providers; "web_search" loads only the web
    # provider (previously a dead flag — there was no web_search provider to
    # load). An explicit ``providers`` dict overrides both.
    if providers is None:
        from ..providers import get_all_providers, get_provider

        if provider == "all":
            providers = get_all_providers()
        elif provider == "web_search":
            providers = {"web_search": get_provider("web_search")}

    evidence_gatherer = (
        get_default_gatherer(
            model=model, providers=providers, embedding_model=embedding_model
        )
        if model
        else None
    )

    # Build graph state and deps
    from ..thresholds import IBE_AGREEMENT_K_DEFAULT

    resolved_k = (
        ibe_agreement_k if ibe_agreement_k is not None else IBE_AGREEMENT_K_DEFAULT
    )
    if resolved_k < 1:
        raise ValueError(
            f"ibe_agreement_k must be >= 1 (got {resolved_k}). "
            f"K=1 is single-run; K>=2 enables Reichenbach-style agreement check."
        )
    state = EpistemicGraphState(
        objective_id=objective_id,
        question=question,
        skip_preplanning=skip_preplanning,
        ibe_agreement_k=resolved_k,
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
        providers=providers,
    )

    # Run the graph
    if verbose:
        logger.info(f"Starting epistemic graph for objective {objective_id}")

    entry_node = (start_at or PrepareObjective)()
    if stop_after is None and output_dir is None:
        # Fast path: no stop, no instrumentation — let pydantic-graph
        # drive the run in one shot.
        graph_result = await epistemic_graph.run(
            entry_node,
            state=state,
            deps=deps,
        )
        result = graph_result.output
    else:
        # Stage-runner mode: drive the graph node-by-node so we can
        # break after a named node and/or record visit timings. The
        # DB is the checkpoint; callers resume by passing start_at
        # on a later invocation. See
        # docs/superpowers/plans/2026-05-03-stage-runners.md.
        visits: list[dict[str, Any]] = []
        async with epistemic_graph.iter(entry_node, state=state, deps=deps) as run:
            while run.result is None:
                next_node = run.next_node
                if next_node is None:
                    break
                node_class = type(next_node)
                t0 = time.monotonic()
                await run.next()
                elapsed_ms = (time.monotonic() - t0) * 1000
                if output_dir is not None:
                    visits.append(
                        {
                            "ts": time.time(),
                            "node": node_class.__name__,
                            "ms": elapsed_ms,
                        }
                    )
                if stop_after is not None and node_class is stop_after:
                    break
        result = run.result.output if run.result is not None else None
        if output_dir is not None:
            await _emit_artifacts(output_dir, visits, repo, objective_id)
        if stop_after is not None:
            await _check_stage_invariant(stop_after, state, repo)

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
