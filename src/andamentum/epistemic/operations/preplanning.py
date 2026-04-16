"""Preplanning operations (Phases 0-2).

Clarify the research question, classify its epistemic type, perform
conceptual analysis, and plan evidence collection strategy. These
operations transform a raw Objective into a well-formed research plan
with provider-specific evidence stubs.

Depends on: base (BaseOperation, OperationResult)
Operates on: Objective, Evidence entities
"""

from .base import BaseOperation, OperationResult

from ..entities import (
    Evidence,
    Objective,
)
from ..patterns import WorkItem


class ClarifyQuestionOperation(BaseOperation):
    """Clarify and refine the research question.

    Transforms a raw objective into a well-formed research question
    with key terms and clarifications.
    """

    entity_type = "objective"

    async def execute(self, work: WorkItem) -> OperationResult:
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

    async def execute(self, work: WorkItem) -> OperationResult:
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

        # Adapter normalizes to lowercase
        question_type = result.question_type

        # Validate against known types
        from ..primitives import QuestionType

        valid_types = {qt.value for qt in QuestionType}
        if question_type not in valid_types:
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message=f"Invalid question type: {question_type}. Valid: {valid_types}",
            )

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

    async def execute(self, work: WorkItem) -> OperationResult:
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

    async def execute(self, work: WorkItem) -> OperationResult:
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

        # Step 1: Semantic provider selection via embedding similarity.
        from ..provider_routing import select_providers

        clarified = objective.clarified_question or objective.description

        if not self.embedding_model:
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message=(
                    "Semantic provider routing requires an embedding_model. "
                    "Pass embedding_model= to create_operations()."
                ),
            )

        providers = await select_providers(
            question=clarified,
            embedding_model=self.embedding_model,
        )

        # Step 2: Formulate queries — one narrow agent call per provider
        created_evidence: list[str] = []

        for provider in providers:
            query = clarified  # fallback if no agent_runner

            if self.agent_runner:
                try:
                    result = await self.run_agent(
                        "epistemic_formulate_query",
                        question=clarified,
                        provider=provider,
                    )
                    query = result.query
                except Exception:
                    # On agent failure, use clarified question as query
                    pass

            evidence = Evidence(
                objective_id=objective.entity_id,
                source_ref=query,
                source_type=provider,
                extracted=False,
            )
            await self.repo.save(evidence)
            created_evidence.append(evidence.entity_id)

        objective.phase = "planned"
        await self.repo.save(objective)

        return OperationResult(
            success=True,
            entity_id=objective.entity_id,
            message=f"Plan created with {len(created_evidence)} evidence sources: {', '.join(providers)}",
            created_entities=created_evidence,
        )
