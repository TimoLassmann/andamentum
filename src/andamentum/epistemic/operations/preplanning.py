"""Preplanning operations (Phases 0-2 + Phase 4 reflection).

Clarify the research question, classify its epistemic type, perform
conceptual analysis, plan evidence collection strategy, and reflect on
decomposition gaps after children run. These operations transform a raw
Objective into a well-formed research plan with provider-specific
evidence stubs and (when needed) refined sub-investigations.

Depends on: base (BaseOperation, OperationResult)
Operates on: Objective, Evidence entities
"""

from typing import Any

from .base import BaseOperation, OperationInput, OperationResult

from ..entities import (
    Evidence,
    Objective,
)


class ClarifyQuestionOperation(BaseOperation):
    """Clarify and refine the research question.

    Transforms a raw objective into a well-formed research question
    with key terms and clarifications.
    """

    entity_type = "objective"

    async def execute(self, work: OperationInput) -> OperationResult:
        objective = await self.repo.get("objective", work.entity_id)

        if not isinstance(objective, Objective):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Objective",
            )

        if objective.phase != "new":
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message=f"Already in phase: {objective.phase}",
            )

        clarified_question = objective.description

        if self.agent_runner:
            import logging

            logger = logging.getLogger(__name__)

            from ..alignment import validate_alignment

            MAX_CLARIFY_ROUNDS = 3
            key_terms: list[str] = []
            prior_feedback = ""

            for round_num in range(1, MAX_CLARIFY_ROUNDS + 1):
                # Clarify
                clarify_kwargs: dict[str, str] = {"question": objective.description}
                if prior_feedback:
                    clarify_kwargs["feedback"] = (
                        f"Your previous clarification drifted from the original intent. "
                        f"Issue: {prior_feedback}. "
                        f"Please try again, staying closer to the original question."
                    )

                result = await self.run_agent(
                    "epistemic_clarify_question", **clarify_kwargs
                )
                clarified_question = result.clarified_question
                key_terms = result.key_terms or []

                # Validate — does the clarification preserve intent?
                validation = await validate_alignment(
                    check_type="clarification",
                    research_question=objective.description,
                    output_to_validate=clarified_question,
                    context=result.reasoning,
                    model=getattr(self.agent_runner, "model", None),
                )

                if validation.aligned:
                    logger.info(
                        "Clarification validated in round %d: %s",
                        round_num,
                        clarified_question[:60],
                    )
                    break

                # Not aligned — log and retry with feedback
                prior_feedback = (
                    validation.issue
                    or validation.suggestion
                    or "drifted from original intent"
                )
                logger.warning(
                    "Clarification round %d drifted: %s. Retrying.",
                    round_num,
                    prior_feedback[:80],
                )
            else:
                # All rounds failed — use the original question
                logger.warning(
                    "Clarification failed after %d rounds. Using original question.",
                    MAX_CLARIFY_ROUNDS,
                )
                clarified_question = objective.description

            objective.clarified_question = clarified_question
            if key_terms:
                objective.key_terms = key_terms

        objective.phase = "clarified"
        await self.repo.save(objective)

        return OperationResult(
            success=True,
            entity_id=objective.entity_id,
            message=f"Clarified: {clarified_question[:50]}...",
        )


class ClassifyQuestionOperation(BaseOperation):
    """Classify the research question into one of seven epistemic types.

    Runs after clarify_question, before conceptual_analysis. Sets
    objective.question_type which drives verification track routing.

    Narrow judgment: one enum output from one LLM call.
    """

    entity_type = "objective"

    async def execute(self, work: OperationInput) -> OperationResult:
        objective = await self.repo.get("objective", work.entity_id)

        # Idempotent: skip if already classified
        if objective.question_type is not None:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message=f"Already classified as {objective.question_type}",
            )

        # Use clarified question if available, otherwise raw description
        question = objective.clarified_question or objective.description

        if not self.agent_runner:
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="No agent runner available for classify_question",
            )

        result = await self.run_agent(
            "epistemic_classify_question",
            question=question,
        )

        # ClassifyQuestionOutput.question_type is typed as the QuestionType
        # enum, so pydantic-ai has already enforced the vocabulary at the
        # OpenAI strict-structured-outputs boundary. No defensive check or
        # case-normalization needed here.
        question_type = result.question_type
        objective.question_type = question_type
        await self.repo.save(objective)

        return OperationResult(
            success=True,
            entity_id=work.entity_id,
            message=f"Classified as {question_type}: {result.reasoning}",
        )


class ConceptualAnalysisOperation(BaseOperation):
    """Perform conceptual analysis of the research question.

    Analyzes key terms, assumptions, and scope to prepare for planning.
    """

    entity_type = "objective"

    async def execute(self, work: OperationInput) -> OperationResult:
        objective = await self.repo.get("objective", work.entity_id)

        if not isinstance(objective, Objective):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Objective",
            )

        if objective.phase != "clarified":
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message=f"Phase {objective.phase} is not 'clarified'",
            )

        if self.agent_runner:
            # Pass clarified question and any key terms from the clarify step
            clarified = objective.clarified_question or objective.description
            key_terms = objective.key_terms

            await self.run_agent(
                "epistemic_conceptual_analysis",
                clarified_question=clarified,
                key_terms=", ".join(key_terms) if key_terms else "",
                question=objective.description,
            )

        objective.phase = "analyzed"
        await self.repo.save(objective)

        return OperationResult(
            success=True,
            entity_id=objective.entity_id,
            message="Conceptual analysis complete",
        )


class PlanTaskOperation(BaseOperation):
    """Plan evidence collection strategy.

    Uses deterministic provider selection based on domain keywords,
    then calls a narrow formulate_query agent once per provider
    to generate optimized search queries.
    """

    entity_type = "objective"

    async def execute(self, work: OperationInput) -> OperationResult:
        objective = await self.repo.get("objective", work.entity_id)

        if not isinstance(objective, Objective):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Objective",
            )

        if objective.phase != "analyzed":
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message=f"Phase {objective.phase} is not 'analyzed'",
            )

        # Step 1: LLM-based provider selection.
        # For each registered provider, a focused agent decides whether it
        # is relevant to this question. This is a narrow binary judgment
        # (yes/no), run once per provider. The agent sees the provider's
        # full description so it can reason about domain coverage.
        from ..providers import PROVIDER_DESCRIPTIONS, PROVIDER_REGISTRY

        clarified = objective.clarified_question or objective.description

        providers: list[str] = []
        if self.agent_runner:
            for provider_name in sorted(PROVIDER_REGISTRY):
                description = PROVIDER_DESCRIPTIONS.get(provider_name, "")
                if not description:
                    continue
                result = await self.run_agent(
                    "epistemic_select_provider",
                    question=clarified,
                    provider=provider_name,
                    provider_description=description,
                )
                if result.relevant:
                    providers.append(provider_name)

        # Always include web_search as universal fallback.
        if "web_search" not in providers:
            providers.append("web_search")

        # Step 2: Formulate queries — one narrow agent call per provider.
        # Each call receives the provider's enriched description so the
        # agent can tailor the query to the provider's strengths.
        #
        # Multi-seed-claim branching: when objective.decomposition is set,
        # formulate queries per (sub_investigation, provider) using each
        # sub-investigation's seed_claim as the question. Tag every
        # Evidence stub with sub_investigation_id so MultiSeedClaim can
        # link evidence per-claim later. This is the Option-2 design from
        # the multi-seed-claim audit: each Claim ends up with its OWN
        # evidence subset, avoiding the support_judgment-collision
        # problem (single scalar field can't represent "supports A,
        # contradicts B" simultaneously).
        created_evidence: list[str] = []

        sub_investigations: list[dict[str, Any]] = []
        if objective.decomposition:
            sub_investigations = (
                objective.decomposition.get("sub_investigations") or []
            )

        if sub_investigations:
            # Per-(sub_investigation, provider) query formulation.
            for sub in sub_investigations:
                sub_id = sub.get("id")
                seed_claim_text = sub.get("seed_claim", "")
                question_text = seed_claim_text or clarified
                for provider in providers:
                    query = question_text  # fallback if no agent_runner
                    if self.agent_runner:
                        result = await self.run_agent(
                            "epistemic_formulate_query",
                            question=question_text,
                            provider=provider,
                            provider_description=PROVIDER_DESCRIPTIONS.get(
                                provider, ""
                            ),
                        )
                        query = result.query

                    evidence = Evidence(
                        objective_id=objective.entity_id,
                        source_ref=query,
                        source_type=provider,
                        extracted=False,
                        sub_investigation_id=sub_id,
                    )
                    await self.repo.save(evidence)
                    created_evidence.append(evidence.entity_id)

            plan_msg = (
                f"Plan created with {len(created_evidence)} evidence sources "
                f"across {len(sub_investigations)} sub-investigations × "
                f"{len(providers)} providers: {', '.join(providers)}"
            )
        else:
            # Original per-objective behavior (no decomposition).
            for provider in providers:
                query = clarified  # fallback if no agent_runner
                if self.agent_runner:
                    result = await self.run_agent(
                        "epistemic_formulate_query",
                        question=clarified,
                        provider=provider,
                        provider_description=PROVIDER_DESCRIPTIONS.get(provider, ""),
                    )
                    query = result.query

                evidence = Evidence(
                    objective_id=objective.entity_id,
                    source_ref=query,
                    source_type=provider,
                    extracted=False,
                )
                await self.repo.save(evidence)
                created_evidence.append(evidence.entity_id)

            plan_msg = (
                f"Plan created with {len(created_evidence)} evidence sources: "
                f"{', '.join(providers)}"
            )

        objective.phase = "planned"
        await self.repo.save(objective)

        return OperationResult(
            success=True,
            entity_id=objective.entity_id,
            message=plan_msg,
            created_entities=created_evidence,
        )


class DecomposeQuestionOperation(BaseOperation):
    """Top-down decomposition of the research question into sub-investigations.

    Calls ``epistemic_decompose_question`` to produce 2-5 sub-investigations
    whose outcomes together settle (or characterize) the question. The
    result is returned via OperationResult; downstream graph wiring (which
    spawns sub-objectives and runs each through the per-claim pipeline) is
    deferred to Phase 2-3.

    Phase 1 use: call this operation standalone to inspect what
    decompositions the agent produces. Useful for prompt iteration before
    committing to graph integration.

    Replaces the bottom-up ``ProposeClaimsOperation`` flow for verificatory,
    explanatory, exploratory, comparative, predictive, compositional, and
    normative questions. ``seed_claim`` mode (when ``claim_to_verify`` is
    set on the objective) bypasses decomposition entirely — there is one
    claim and no decomposition is needed.

    Idempotent at the agent-call level: re-running stores the latest
    decomposition on the objective's metadata via the same pattern other
    preplanning operations use.
    """

    entity_type = "objective"

    async def execute(self, work: OperationInput) -> OperationResult:
        objective = await self.repo.get("objective", work.entity_id)

        if not isinstance(objective, Objective):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Objective",
                did_work=False,
            )

        if objective.claim_to_verify:
            # seed_claim mode: no decomposition needed.
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message=(
                    "Skipped: objective has claim_to_verify (seed_claim mode "
                    "bypasses top-down decomposition)"
                ),
                did_work=False,
            )

        if not self.agent_runner:
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="No agent_runner configured; decomposition requires an LLM",
                did_work=False,
            )

        # Idempotence: skip if already decomposed.
        if objective.decomposition is not None:
            existing = objective.decomposition
            sub_count = len(existing.get("sub_investigations", []))
            return OperationResult(
                success=True,
                entity_id=objective.entity_id,
                message=(
                    f"Already decomposed into {sub_count} sub-investigations "
                    f"(combination={existing.get('combination_rule')})"
                ),
                did_work=False,
            )

        question = objective.clarified_question or objective.description
        question_type = objective.question_type or "verificatory"

        result = await self.run_agent(
            "epistemic_decompose_question",
            question=question,
            question_type=question_type,
        )

        # Persist on the parent objective so SpawnSubObjectivesOperation
        # can read it. Stored as a plain dict so entities/ stays free of
        # dependencies on agents/output_models.py — and so the same code
        # path works for pydantic models in production and for
        # SimpleNamespace-shaped mocks in tests.
        objective.decomposition = {
            "sub_investigations": [
                {
                    "id": s.id,
                    "seed_claim": s.seed_claim,
                    "rationale": s.rationale,
                    "weight": getattr(s, "weight", 1.0),
                }
                for s in result.sub_investigations
            ],
            "combination_rule": result.combination_rule,
            "rationale": result.rationale,
        }
        objective.combination_rule = result.combination_rule
        await self.repo.save(objective)

        sub_count = len(result.sub_investigations)
        sub_summary = ", ".join(
            f"{s.id}: {s.seed_claim[:60]}" for s in result.sub_investigations
        )

        return OperationResult(
            success=True,
            entity_id=objective.entity_id,
            message=(
                f"Decomposed into {sub_count} sub-investigations "
                f"(combination={result.combination_rule}). {sub_summary}"
            ),
        )


class SpawnSubObjectivesOperation(BaseOperation):
    """Spawn one child Objective per sub-investigation in the parent's decomposition.

    Reads ``parent.decomposition`` (set by DecomposeQuestionOperation) and
    creates a child Objective for each sub-investigation. Each child has:

    - ``parent_objective_id`` set to the parent's entity_id
    - ``sub_investigation_id`` set to the decomposition's "A"/"B"/"C" tag
    - ``claim_to_verify`` populated from the sub-investigation's seed_claim
      (so each child runs in seed_claim mode — no further decomposition)
    - ``question_type`` inherited from the parent
    - ``description`` set to the seed_claim for human-readability

    The parent's ``sub_objective_ids`` is updated with the children's ids.

    Idempotent: if ``parent.sub_objective_ids`` is already populated, the
    operation is a no-op.

    Phase 2 only — graph wiring (which then runs each child Objective
    through the existing per-claim pipeline) is deferred to Phase 3.
    """

    entity_type = "objective"

    async def execute(self, work: OperationInput) -> OperationResult:
        objective = await self.repo.get("objective", work.entity_id)

        if not isinstance(objective, Objective):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Objective",
                did_work=False,
            )

        if objective.decomposition is None:
            return OperationResult(
                success=False,
                entity_id=objective.entity_id,
                message=(
                    "No decomposition on this objective — run "
                    "DecomposeQuestionOperation first"
                ),
                did_work=False,
            )

        sub_investigations = objective.decomposition.get("sub_investigations", [])
        if not sub_investigations:
            return OperationResult(
                success=False,
                entity_id=objective.entity_id,
                message="Decomposition has no sub_investigations to spawn",
                did_work=False,
            )

        # Delta-spawn: figure out which sub-investigation IDs already
        # have a child, and only spawn the rest. This keeps Phase-2
        # idempotence (re-running on a fully-spawned parent is a no-op)
        # while letting Phase-4 reflection-driven additions land
        # incrementally.
        existing_sub_ids: set[str] = set()
        for child_id in objective.sub_objective_ids:
            try:
                child = await self.repo.get("objective", child_id)
            except Exception:
                continue
            if child.sub_investigation_id:
                existing_sub_ids.add(child.sub_investigation_id)

        to_spawn = [
            sub
            for sub in sub_investigations
            if sub.get("id", "?") not in existing_sub_ids
        ]
        if not to_spawn:
            return OperationResult(
                success=True,
                entity_id=objective.entity_id,
                message=(
                    f"Already spawned {len(objective.sub_objective_ids)} "
                    f"sub-objectives"
                ),
                did_work=False,
            )

        spawned_ids: list[str] = []
        for sub in to_spawn:
            sub_id = sub.get("id", "?")
            seed_claim = sub.get("seed_claim", "")
            # phase="analyzed" so the graph's PrepareObjective short-circuits
            # the preplanning ops on the child — clarify/classify/analyze
            # have nothing to add: the seed_claim is already a clarified
            # claim sentence and question_type is inherited from the parent.
            child = Objective(
                description=seed_claim,
                question_type=objective.question_type,
                claim_to_verify=seed_claim,
                parent_objective_id=objective.entity_id,
                sub_investigation_id=sub_id,
                phase="analyzed",
            )
            # Make objective_id self-referential so repo.get_objective works.
            child.objective_id = child.entity_id
            await self.repo.save(child)
            spawned_ids.append(child.entity_id)

        objective.sub_objective_ids = list(objective.sub_objective_ids) + spawned_ids
        await self.repo.save(objective)

        return OperationResult(
            success=True,
            entity_id=objective.entity_id,
            message=f"Spawned {len(spawned_ids)} sub-objectives from decomposition",
            created_entities=spawned_ids,
        )


class ReflectOnGapsOperation(BaseOperation):
    """Reflect on the current decomposition's verdicts and add sub-investigations
    when a load-bearing gap is found.

    Runs after children have been scored and combined. Reads each
    child's posterior, summarizes the children's state for the agent,
    and asks ``epistemic_reflect_on_gaps`` whether the current children
    are adequate. When the agent declares a gap, the operation appends
    new sub-investigations to ``parent.decomposition`` (re-keyed with
    deterministic IDs ``D``, ``E``, ...) and bumps ``reflection_rounds``.

    Reflection is corrective, not search-like: the orchestrator caps the
    number of rounds (default 1). The operation is idempotent at the
    agent-call level — re-running after a sufficient verdict is a
    did_work=False no-op.

    Phase 4 of the unified architecture. The orchestrator
    (``run_research_question_decomposed``) decides *when* to call this op
    (e.g. only when the combined verdict is insufficient or any child
    flagged retrieval_failed); the op decides *what* to add.
    """

    entity_type = "objective"

    async def execute(self, work: OperationInput) -> OperationResult:
        objective = await self.repo.get("objective", work.entity_id)

        if not isinstance(objective, Objective):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Objective",
                did_work=False,
            )

        if objective.decomposition is None:
            return OperationResult(
                success=False,
                entity_id=objective.entity_id,
                message=(
                    "No decomposition on this objective — "
                    "DecomposeQuestionOperation must run before reflection"
                ),
                did_work=False,
            )

        if not objective.sub_objective_ids:
            return OperationResult(
                success=False,
                entity_id=objective.entity_id,
                message=(
                    "No spawned children — SpawnSubObjectivesOperation must "
                    "run before reflection"
                ),
                did_work=False,
            )

        if not self.agent_runner:
            return OperationResult(
                success=False,
                entity_id=objective.entity_id,
                message="No agent_runner configured; reflection requires an LLM",
                did_work=False,
            )

        # Build the per-child summary the agent consumes. Each child's
        # posterior is recomputed from repo state (cheap; a pure function
        # of stored evidence + claims). When compute_posterior returns
        # None — e.g. the child's question_type is ineligible for
        # posterior reporting — we surface "n/a" rather than crashing.
        from ..confidence import compute_posterior

        question = objective.clarified_question or objective.description
        rule = objective.combination_rule or "AND"

        child_lines: list[str] = []
        for child_id in objective.sub_objective_ids:
            try:
                child = await self.repo.get("objective", child_id)
            except Exception:
                child_lines.append(f"- (child {child_id[:8]}: lookup failed)")
                continue
            sub_id = child.sub_investigation_id or "?"
            seed_claim = child.claim_to_verify or child.description or ""
            try:
                report = await compute_posterior(self.repo, child_id)
            except Exception:
                report = None
            if report is None:
                verdict_str = "n/a"
                p_str = "n/a"
                terminal = "n/a"
            else:
                verdict_str = report.integration_verdict or "n/a"
                p_str = f"{report.posterior:.2f}"
                terminal = report.terminal_state
            child_lines.append(
                f"- {sub_id}: {seed_claim} → {verdict_str} (p={p_str}, "
                f"terminal_state={terminal})"
            )
        current_decomposition = "\n".join(child_lines)

        # Combined view derived from the same posteriors. We import here
        # to avoid creating a top-level dep cycle between
        # operations/preplanning.py and decomposed_runner.py.
        from ..decomposed_runner import combine_sub_verdicts
        from ..operations_runner import PipelineResult

        # Build minimal PipelineResult stand-ins so we can reuse
        # combine_sub_verdicts here without a parallel implementation.
        proxy_results: list[PipelineResult] = []
        for child_id in objective.sub_objective_ids:
            try:
                report = await compute_posterior(self.repo, child_id)
            except Exception:
                report = None
            proxy_results.append(
                PipelineResult(
                    objective_id=child_id,
                    iterations=0,
                    successful=1 if report else 0,
                    failed=0,
                    status="ok",
                    posterior=report,
                )
            )
        weights_list = [
            float(s.get("weight", 1.0))
            for s in objective.decomposition.get("sub_investigations", [])
        ]
        # Length mismatch (e.g. mid-reflection state where children
        # haven't all been spawned yet) → fall back to no weights so the
        # combiner doesn't raise.
        weights_arg = (
            weights_list if len(weights_list) == len(proxy_results) else None
        )
        combined = combine_sub_verdicts(proxy_results, rule, weights=weights_arg)
        combined_p_str = (
            f"{combined.posterior:.2f}" if combined.posterior is not None else "n/a"
        )

        result = await self.run_agent(
            "epistemic_reflect_on_gaps",
            question=question,
            combination_rule=rule,
            current_decomposition=current_decomposition,
            combined_verdict=combined.verdict,
            combined_posterior=combined_p_str,
        )

        round_num = objective.reflection_rounds + 1

        if result.sufficient:
            objective.reflection_history.append(
                {
                    "round": round_num,
                    "sufficient": True,
                    "gap_description": "",
                    "added_count": 0,
                    "rationale": result.rationale,
                }
            )
            objective.reflection_rounds = round_num
            await self.repo.save(objective)
            return OperationResult(
                success=True,
                entity_id=objective.entity_id,
                message=f"Sufficient at round {round_num}: {result.rationale}",
                did_work=False,
            )

        # Sufficient=False — append new sub-investigations, re-keying
        # ids deterministically so they don't collide with existing ones.
        existing_ids: set[str] = {
            s.get("id", "")
            for s in objective.decomposition.get("sub_investigations", [])
        }
        next_ord = ord("A")
        if existing_ids:
            valid_ords = [
                ord(i) for i in existing_ids if len(i) == 1 and i.isalpha()
            ]
            if valid_ords:
                next_ord = max(valid_ords) + 1

        new_subs: list[dict[str, Any]] = []
        for s in result.additional_sub_investigations or []:
            new_id = chr(next_ord)
            next_ord += 1
            new_subs.append(
                {
                    "id": new_id,
                    "seed_claim": s.seed_claim,
                    "rationale": s.rationale,
                    "weight": getattr(s, "weight", 1.0),
                }
            )

        if not new_subs:
            # Agent said sufficient=False but added no sub-investigations.
            # Treat as a no-op decision and record it.
            objective.reflection_history.append(
                {
                    "round": round_num,
                    "sufficient": False,
                    "gap_description": result.gap_description,
                    "added_count": 0,
                    "rationale": result.rationale,
                }
            )
            objective.reflection_rounds = round_num
            await self.repo.save(objective)
            return OperationResult(
                success=True,
                entity_id=objective.entity_id,
                message=(
                    f"Round {round_num}: gap reported but no sub-investigations "
                    f"proposed: {result.gap_description}"
                ),
                did_work=False,
            )

        objective.decomposition["sub_investigations"] = (
            objective.decomposition.get("sub_investigations", []) + new_subs
        )
        objective.reflection_history.append(
            {
                "round": round_num,
                "sufficient": False,
                "gap_description": result.gap_description,
                "added_count": len(new_subs),
                "rationale": result.rationale,
            }
        )
        objective.reflection_rounds = round_num
        await self.repo.save(objective)

        return OperationResult(
            success=True,
            entity_id=objective.entity_id,
            message=(
                f"Round {round_num}: added {len(new_subs)} sub-investigations "
                f"to close gap: {result.gap_description}"
            ),
        )
