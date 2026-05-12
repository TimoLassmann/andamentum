"""Preplanning operations (Phases 0-2 + Phase 4 reflection).

Clarify the research question, classify its epistemic type, perform
conceptual analysis, and reflect on decomposition gaps after children
run. Evidence-gathering itself is no longer in this module — it lives
in :mod:`andamentum.epistemic.operations.dispatch_gather`, driven by
the description-driven dispatch path. See ``docs/superpowers/plans/
2026-05-12-description-driven-provider-dispatch.md``.

Depends on: base (BaseOperation, OperationResult)
Operates on: Objective entities
"""

from .base import BaseOperation, OperationInput, OperationResult

from ..entities import (
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
