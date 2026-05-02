"""Preplanning operations (Phases 0-2 + Phase 4 reflection).

Clarify the research question, classify its epistemic type, perform
conceptual analysis, plan evidence collection strategy, and reflect on
decomposition gaps after children run. These operations transform a raw
Objective into a well-formed research plan with provider-specific
evidence stubs and (when needed) refined sub-investigations.

Depends on: base (BaseOperation, OperationResult)
Operates on: Objective, Evidence entities
"""

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

        # Phase 6 of the Move-3 plan: typed Decomposition access.
        sub_investigations = (
            objective.decomposition.sub_investigations
            if objective.decomposition
            else []
        )

        if sub_investigations:
            # Phase 2 of lazy-escalation: per-sub-claim, pick the
            # SINGLE BEST provider via LLM rank (instead of querying
            # all providers in parallel). Round 1 starts narrow; later
            # rounds (driven by demand from scrutiny) will pull other
            # providers from the candidate list via InvestigateClaim
            # operation.
            #
            # If the ranker can't pick (no agent_runner, or LLM
            # output doesn't match a candidate), fall back to the
            # first relevant provider — keeps the pipeline alive
            # but loses the lazy-escalation benefit for that sub.
            for sub in sub_investigations:
                sub_id = sub.id
                seed_claim_text = sub.seed_claim
                question_text = seed_claim_text or clarified

                chosen_provider = providers[0] if providers else "web_search"
                if self.agent_runner and len(providers) > 1:
                    candidates_text = "\n".join(
                        f"- {p}: {PROVIDER_DESCRIPTIONS.get(p, '')}"
                        for p in providers
                    )
                    rank_result = await self.run_agent(
                        "epistemic_rank_providers",
                        sub_claim=question_text,
                        candidates=candidates_text,
                    )
                    if rank_result.chosen_provider in providers:
                        chosen_provider = rank_result.chosen_provider
                    # else: fall back to providers[0] (already set)

                # Formulate query for the chosen provider only.
                query = question_text  # fallback if no agent_runner
                if self.agent_runner:
                    result = await self.run_agent(
                        "epistemic_formulate_query",
                        question=question_text,
                        provider=chosen_provider,
                        provider_description=PROVIDER_DESCRIPTIONS.get(
                            chosen_provider, ""
                        ),
                    )
                    query = result.query

                evidence = Evidence(
                    objective_id=objective.entity_id,
                    source_ref=query,
                    source_type=chosen_provider,
                    extracted=False,
                    sub_investigation_id=sub_id,
                )
                await self.repo.save(evidence)
                created_evidence.append(evidence.entity_id)

            plan_msg = (
                f"Plan created with {len(created_evidence)} evidence sources "
                f"across {len(sub_investigations)} sub-investigations "
                f"(round-1 lazy: one provider per sub-claim, ranked)"
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
            sub_count = len(existing.sub_investigations)
            return OperationResult(
                success=True,
                entity_id=objective.entity_id,
                message=(
                    f"Already decomposed into {sub_count} sub-investigations "
                    f"(combination={existing.combination_rule})"
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

        # Persist a typed Decomposition on the parent objective. Phase 6
        # of the Move-3 plan: this used to be a raw dict; now it's a
        # Decomposition pydantic model so consumers access fields by name
        # rather than via dict.get(...). The conversion mirrors the
        # agent-output schema (QuestionDecomposition) into the entity
        # data model (Decomposition) — same shape, separate concerns.
        from ..entities.decomposition import Decomposition, SubInvestigation

        objective.decomposition = Decomposition(
            sub_investigations=[
                SubInvestigation(
                    id=s.id,
                    seed_claim=s.seed_claim,
                    rationale=s.rationale,
                    weight=getattr(s, "weight", 1.0),
                )
                for s in result.sub_investigations
            ],
            combination_rule=result.combination_rule,
            rationale=result.rationale,
        )
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

