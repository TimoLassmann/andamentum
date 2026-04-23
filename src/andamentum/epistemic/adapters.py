"""Agent Adapters - Strict field access for epistemic agent outputs.

Each adapter transforms a typed Pydantic model (returned by the SDK) into a
dataclass (consumed by operations). The SDK already validates agent output
against the manifest's output_model, so fields are guaranteed to exist.

If an adapter accesses a field that doesn't exist, Python raises AttributeError
— a visible contract violation instead of a silent wrong default.

Architecture: Layer 1 (framework-agnostic)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .primitives import (
    CausalRole,
    CriticismCategory,
    DataSourceType,
    MethodType,
    PredictionType,
    TemporalApproach,
)


# ══════════════════════════════════════════════════════════════════════════════
# RESULT DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class WriteAnswerResult:
    """Result from write_answer agent.

    Manifest fields: title, verdict, answer.
    """

    title: str = "Research Summary"
    verdict: str = ""
    answer: str = ""


@dataclass
class ExtractResult:
    """Result from evidence extraction.

    Manifest fields: source_type, source_ref, relevant_quotes,
    experimental_context, limitations.
    """

    content: str  # Joined from relevant_quotes
    limitations: list[str] = field(default_factory=list)
    experimental_context: Optional[str] = None


@dataclass
class DeductiveResult:
    """Result from deductive validation.

    Manifest fields: claim_id, deductive_soundness, confidence_estimate,
    passes_deductive_validation, issues_found, issue_types, recommendation.
    """

    passes_deductive_validation: bool = False
    issues_found: list[str] = field(default_factory=list)
    issue_types: list[str] = field(default_factory=list)


@dataclass
class ComputationalResult:
    """Result from computational verification code generation.

    Manifest fields: claim_id, verification_code, packages_required,
    expected_behavior, test_description.

    NOTE: This agent generates verification CODE, not execution results.
    The operation must handle the fact that code has not been run.
    """

    verification_code: str = ""
    packages_required: list[str] = field(default_factory=list)
    expected_behavior: str = ""
    test_description: str = ""


@dataclass
class ClarifyResult:
    """Result from question clarification.

    Manifest fields: ambiguity_level, clarified_question, key_terms, reasoning.
    """

    clarified_question: str = ""
    key_terms: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class ConceptualAnalysisResult:
    """Result from conceptual analysis.

    Manifest fields: terms, definitions, assumptions, context_summary.
    """

    terms: list[str] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    context_summary: str = ""


@dataclass
class ResolveUncertaintyResult:
    """Result from uncertainty resolution.

    Manifest fields: uncertainty_id, can_resolve, resolution, remaining_concerns.
    """

    resolution: str = ""
    can_resolve: bool = False
    remaining_concerns: list[str] = field(default_factory=list)


@dataclass
class InvestigateClaimResult:
    """Result from claim investigation.

    Manifest fields: evidence_queries (list with source_type, query), reasoning.
    """

    evidence_queries: list[Any] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class AnalyzeArgumentResult:
    """Result from argument analysis.

    Manifest fields: premises, conclusion, validity, soundness, fallacies.
    """

    premises: list[str] = field(default_factory=list)
    conclusion: str = ""
    validity: str = "indeterminate"  # valid, invalid, indeterminate
    soundness: str = "questionable"  # sound, unsound, questionable
    fallacies: list[str] = field(default_factory=list)


@dataclass
class RecordDecisionResult:
    """Result from decision recording.

    Manifest fields (epistemic_decide): statement, justification,
    claim_indices, reversible, reversal_conditions.
    """

    statement: str = ""
    justification: str = ""


@dataclass
class GenerateCounterqueryResult:
    """Result from adversarial query generation.

    Manifest fields: query, framing.
    """

    query: str
    framing: str


@dataclass
class EvaluateCounterargumentResult:
    """Result from counterargument evaluation.

    Manifest fields: relevance, specificity, evidence_backed,
    source_credibility, category, justification.
    """

    relevance: float = 0.0
    specificity: float = 0.0
    evidence_backed: float = 0.0
    source_credibility: float = 0.0
    category: CriticismCategory = CriticismCategory.METHODOLOGICAL
    justification: str = ""


@dataclass
class ClassifyEvidenceDomainResult:
    """Result from evidence domain classification.

    Manifest fields: method_type, data_source, temporal_approach,
    causal_role, confidence, justification. The four classification
    fields are enums — pydantic-ai enforces the enum constraint via
    the JSON schema so downstream code never has to coerce from str.
    """

    method_type: MethodType = MethodType.THEORETICAL
    data_source: DataSourceType = DataSourceType.SECONDARY
    temporal_approach: TemporalApproach = TemporalApproach.CROSS_SECTIONAL
    causal_role: CausalRole = CausalRole.PHENOMENOLOGICAL
    confidence: float = 0.0
    justification: str = ""


@dataclass
class CheckPairwiseIndependenceResult:
    """Result from pairwise independence check.

    Manifest fields: independent, rationale.
    """

    independent: bool = True
    rationale: str = ""


@dataclass
class ClassifyPredictionResult:
    """Result from prediction classification.

    Manifest fields: prediction_type, specificity, success_criteria,
    failure_criteria, time_horizon, justification.
    """

    prediction_type: PredictionType = PredictionType.QUALITATIVE
    specificity: float = 0.0
    success_criteria: str = ""
    failure_criteria: str = ""
    time_horizon: str = "indefinite"  # Literal[5 values] — see output_models.TimeHorizon
    justification: str = ""


@dataclass
class IdentifyTestableAspectResult:
    """Result from testable aspect identification.

    Manifest fields: testable_dimension, observation_type.
    """

    testable_dimension: str
    observation_type: str


@dataclass
class SpecifyPredictionResult:
    """Result from prediction specification.

    Manifest fields: expected_observation, conditions, timeframe, measurability.
    """

    expected_observation: str
    conditions: str
    timeframe: str
    measurability: str


@dataclass
class DefineFalsificationResult:
    """Result from falsification criterion definition.

    Manifest fields: falsification_criterion.
    """

    falsification_criterion: str


@dataclass
class AssessEvidenceQualityResult:
    """Result from evidence quality assessment.

    Manifest fields: source_credibility, relevance, specificity,
    recency_appropriate, justification.
    """

    source_credibility: float = 0.0
    relevance: float = 0.0
    specificity: float = 0.0
    recency_appropriate: float = 0.0
    justification: str = ""


@dataclass
class ClassifyQuestionResult:
    """Result from question type classification.

    Manifest fields: question_type, reasoning.
    """

    question_type: str
    reasoning: str


@dataclass
class ContrastiveEvaluationResult:
    """Result from contrastive evaluation.

    Manifest fields: better_claim, distinguishing_observation, confidence.
    """

    better_claim: str
    distinguishing_observation: str
    confidence: float


@dataclass
class CrossClaimConsistencyResult:
    """Result from cross-claim consistency check.

    Manifest fields: conflicts, tension_point.
    """

    conflicts: bool
    tension_point: str


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def adapt_write_answer(raw: Any) -> WriteAnswerResult:
    """Adapt epistemic_write_answer output."""
    return WriteAnswerResult(
        title=raw.title,
        verdict=getattr(raw, "verdict", ""),
        answer=raw.answer,
    )


def adapt_extract(raw: Any) -> ExtractResult:
    """Adapt epistemic_extract_evidence output.

    Joins relevant_quotes into a single content string.
    """
    quotes = raw.relevant_quotes
    content = "\n".join(quotes) if quotes else ""

    return ExtractResult(
        content=content,
        limitations=raw.limitations,
        experimental_context=raw.experimental_context,
    )


def adapt_deductive(raw: Any) -> DeductiveResult:
    """Adapt epistemic_deductive_validation output."""
    return DeductiveResult(
        passes_deductive_validation=raw.passes_deductive_validation,
        issues_found=raw.issues_found,
        issue_types=raw.issue_types,
    )


def adapt_computational(raw: Any) -> ComputationalResult:
    """Adapt epistemic_verify_computationally output.

    This agent generates verification CODE, not execution results.
    """
    return ComputationalResult(
        verification_code=raw.verification_code,
        packages_required=raw.packages_required,
        expected_behavior=raw.expected_behavior,
        test_description=raw.test_description,
    )


def adapt_clarify(raw: Any) -> ClarifyResult:
    """Adapt epistemic_clarify_question output."""
    return ClarifyResult(
        clarified_question=raw.clarified_question,
        key_terms=raw.key_terms,
        reasoning=raw.reasoning,
    )


def adapt_conceptual_analysis(raw: Any) -> ConceptualAnalysisResult:
    """Adapt epistemic_conceptual_analysis output."""
    return ConceptualAnalysisResult(
        terms=raw.terms,
        definitions=raw.definitions,
        assumptions=raw.assumptions,
        context_summary=raw.context_summary,
    )


def adapt_resolve_uncertainty(raw: Any) -> ResolveUncertaintyResult:
    """Adapt epistemic_resolve_uncertainty output."""
    return ResolveUncertaintyResult(
        resolution=raw.resolution,
        can_resolve=raw.can_resolve,
        remaining_concerns=raw.remaining_concerns,
    )


def adapt_investigate_claim(raw: Any) -> InvestigateClaimResult:
    """Adapt epistemic_investigate_claim output."""
    return InvestigateClaimResult(
        evidence_queries=raw.evidence_queries,
        reasoning=raw.reasoning,
    )


def adapt_analyze_argument(raw: Any) -> AnalyzeArgumentResult:
    """Adapt epistemic_analyze_argument output."""
    return AnalyzeArgumentResult(
        premises=raw.premises,
        conclusion=raw.conclusion,
        validity=raw.validity,
        soundness=raw.soundness,
        fallacies=raw.fallacies,
    )


def adapt_record_decision(raw: Any) -> RecordDecisionResult:
    """Adapt epistemic_decide output."""
    return RecordDecisionResult(
        statement=raw.statement,
        justification=raw.justification,
    )


def adapt_generate_counterquery(raw: Any) -> GenerateCounterqueryResult:
    """Adapt epistemic_generate_counterquery output."""
    return GenerateCounterqueryResult(
        query=raw.query.strip(),
        framing=raw.framing.strip(),
    )


def adapt_evaluate_counterargument(raw: Any) -> EvaluateCounterargumentResult:
    """Adapt epistemic_evaluate_counterargument output."""
    return EvaluateCounterargumentResult(
        relevance=raw.relevance,
        specificity=raw.specificity,
        evidence_backed=raw.evidence_backed,
        source_credibility=raw.source_credibility,
        category=raw.category,
        justification=raw.justification,
    )


def adapt_classify_evidence_domain(raw: Any) -> ClassifyEvidenceDomainResult:
    """Adapt epistemic_classify_evidence_domain output."""
    return ClassifyEvidenceDomainResult(
        method_type=raw.method_type,
        data_source=raw.data_source,
        temporal_approach=raw.temporal_approach,
        causal_role=raw.causal_role,
        confidence=raw.confidence,
        justification=raw.justification,
    )


def adapt_check_pairwise_independence(raw: Any) -> CheckPairwiseIndependenceResult:
    """Adapt epistemic_check_pairwise_independence output."""
    return CheckPairwiseIndependenceResult(
        independent=bool(raw.independent),
        rationale=raw.rationale,
    )


def adapt_classify_prediction(raw: Any) -> ClassifyPredictionResult:
    """Adapt epistemic_classify_prediction output."""
    return ClassifyPredictionResult(
        prediction_type=raw.prediction_type,
        specificity=raw.specificity,
        success_criteria=raw.success_criteria,
        failure_criteria=raw.failure_criteria,
        time_horizon=raw.time_horizon,
        justification=raw.justification,
    )


def adapt_identify_testable_aspect(raw: Any) -> IdentifyTestableAspectResult:
    """Adapt epistemic_identify_testable_aspect output.

    observation_type is a Literal on IdentifyTestableAspectOutput, so
    pydantic-ai enforces the value; no case/whitespace normalisation here.
    """
    return IdentifyTestableAspectResult(
        testable_dimension=raw.testable_dimension,
        observation_type=raw.observation_type,
    )


def adapt_specify_prediction(raw: Any) -> SpecifyPredictionResult:
    """Adapt epistemic_specify_prediction output."""
    return SpecifyPredictionResult(
        expected_observation=raw.expected_observation,
        conditions=raw.conditions,
        timeframe=raw.timeframe,
        measurability=raw.measurability.strip().lower(),
    )


def adapt_define_falsification(raw: Any) -> DefineFalsificationResult:
    """Adapt epistemic_define_falsification output."""
    return DefineFalsificationResult(
        falsification_criterion=raw.falsification_criterion,
    )


def adapt_assess_evidence_quality(raw: Any) -> AssessEvidenceQualityResult:
    """Adapt epistemic_assess_evidence_quality output."""
    return AssessEvidenceQualityResult(
        source_credibility=raw.source_credibility,
        relevance=raw.relevance,
        specificity=raw.specificity,
        recency_appropriate=raw.recency_appropriate,
        justification=raw.justification,
    )


def adapt_classify_question(raw: Any) -> ClassifyQuestionResult:
    """Adapt epistemic_classify_question output.

    ClassifyQuestionOutput.question_type is a QuestionType enum —
    pydantic-ai already constrains the value via the JSON schema's
    enum, so no case/whitespace normalization is required here.
    """
    return ClassifyQuestionResult(
        question_type=raw.question_type,
        reasoning=raw.reasoning,
    )


@dataclass
class IdentifySingleIssueResult:
    has_issue: bool
    description: str
    issue_type: str
    reversal_test: bool


def adapt_identify_single_issue(raw: Any) -> IdentifySingleIssueResult:
    issue_type = raw.issue_type.strip().lower() if raw.issue_type else ""
    return IdentifySingleIssueResult(
        has_issue=bool(raw.has_issue),
        description=raw.description.strip() if raw.description else "",
        issue_type=issue_type,
        reversal_test=issue_type in {"unknown", "contradiction"},
    )


def adapt_contrastive_evaluation(raw: Any) -> ContrastiveEvaluationResult:
    """Adapt epistemic_contrastive_evaluation output."""
    return ContrastiveEvaluationResult(
        better_claim=raw.better_claim.strip().upper(),
        distinguishing_observation=raw.distinguishing_observation,
        confidence=float(raw.confidence),
    )


def adapt_cross_claim_consistency(raw: Any) -> CrossClaimConsistencyResult:
    """Adapt epistemic_cross_claim_consistency output."""
    return CrossClaimConsistencyResult(
        conflicts=bool(raw.conflicts),
        tension_point=raw.tension_point,
    )


@dataclass
class FormulateQueryResult:
    """Result from search query formulation.

    Manifest fields: query, rationale.
    """

    query: str
    rationale: str


@dataclass
class SelectProviderResult:
    """Result from provider selection.

    Manifest fields: relevant, reasoning.
    """

    relevant: bool
    reasoning: str


def adapt_select_provider(raw: Any) -> SelectProviderResult:
    """Adapt epistemic_select_provider output."""
    return SelectProviderResult(
        relevant=bool(raw.relevant),
        reasoning=str(raw.reasoning).strip(),
    )


def adapt_formulate_query(raw: Any) -> FormulateQueryResult:
    """Adapt epistemic_formulate_query output."""
    return FormulateQueryResult(
        query=raw.query.strip(),
        rationale=raw.rationale.strip(),
    )


@dataclass
class ExtractAssertionResult:
    """Result from assertion extraction.

    Manifest fields: assertion.
    """

    assertion: str


@dataclass
class DraftClaimResult:
    """Result from claim drafting.

    Manifest fields: statement, scope, direction.
    """

    statement: str
    scope: str
    direction: str


def adapt_extract_assertion(raw: Any) -> ExtractAssertionResult:
    """Adapt epistemic_extract_assertion output."""
    return ExtractAssertionResult(assertion=raw.assertion.strip())


def adapt_draft_claim(raw: Any) -> DraftClaimResult:
    """Adapt epistemic_draft_claim output."""
    return DraftClaimResult(
        statement=raw.statement.strip(),
        scope=raw.scope.strip(),
        # direction is a Literal on DraftClaimOutput — pydantic-ai enforces
        # the value; no .strip().lower() needed.
        direction=raw.direction,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTER REGISTRY
# ══════════════════════════════════════════════════════════════════════════════


ADAPTERS: dict[str, Callable[..., Any]] = {
    "epistemic_write_answer": adapt_write_answer,
    "epistemic_extract_evidence": adapt_extract,
    "epistemic_deductive_validation": adapt_deductive,
    "epistemic_verify_computationally": adapt_computational,
    "epistemic_clarify_question": adapt_clarify,
    "epistemic_conceptual_analysis": adapt_conceptual_analysis,
    "epistemic_resolve_uncertainty": adapt_resolve_uncertainty,
    "epistemic_investigate_claim": adapt_investigate_claim,
    "epistemic_analyze_argument": adapt_analyze_argument,
    "epistemic_record_decision": adapt_record_decision,
    "epistemic_generate_counterquery": adapt_generate_counterquery,
    "epistemic_evaluate_counterargument": adapt_evaluate_counterargument,
    "epistemic_classify_evidence_domain": adapt_classify_evidence_domain,
    "epistemic_check_pairwise_independence": adapt_check_pairwise_independence,
    "epistemic_classify_prediction": adapt_classify_prediction,
    "epistemic_identify_testable_aspect": adapt_identify_testable_aspect,
    "epistemic_specify_prediction": adapt_specify_prediction,
    "epistemic_define_falsification": adapt_define_falsification,
    "epistemic_assess_evidence_quality": adapt_assess_evidence_quality,
    "epistemic_classify_question": adapt_classify_question,
    "epistemic_contrastive_evaluation": adapt_contrastive_evaluation,
    "epistemic_cross_claim_consistency": adapt_cross_claim_consistency,
    "epistemic_select_provider": adapt_select_provider,
    "epistemic_formulate_query": adapt_formulate_query,
    "epistemic_extract_assertion": adapt_extract_assertion,
    "epistemic_draft_claim": adapt_draft_claim,
    "epistemic_identify_single_issue": adapt_identify_single_issue,
}


def get_adapter(agent_name: str) -> Optional[Callable[..., Any]]:
    """Get the adapter function for an agent."""
    return ADAPTERS.get(agent_name)


def adapt_agent_output(agent_name: str, raw: Any) -> Any:
    """Adapt raw agent output using the appropriate adapter.

    Args:
        agent_name: Name of the agent that produced output
        raw: Raw output from agent

    Returns:
        Adapted result or raw output if no adapter found
    """
    adapter = get_adapter(agent_name)
    if adapter:
        return adapter(raw)
    return raw
