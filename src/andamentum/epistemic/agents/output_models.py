"""Output models for epistemic agents.

Each Pydantic BaseModel corresponds to the output_model from a .md agent manifest.
Field names MUST match what adapters.py accesses via attribute access.

Architecture: Layer 1 (framework-agnostic, pure Pydantic)
"""

from typing import Literal

from pydantic import BaseModel, Field

from ..primitives import (
    CausalRole,
    DataSourceType,
    MethodType,
    TemporalApproach,
)


# ── Preplanning ──────────────────────────────────────────────────────────


class ClarifyQuestionOutput(BaseModel):
    """Output from epistemic_clarify_question agent."""

    ambiguity_level: str = Field(
        description='Level of ambiguity - "clear", "moderate", or "high"'
    )
    clarified_question: str = Field(
        description="Rewritten question that is unambiguous and specific"
    )
    key_terms: list[str] = Field(
        description="Terms that need explicit definition for this investigation"
    )
    reasoning: str = Field(
        description="Brief explanation of interpretation choice and any alternatives considered"
    )


class ConceptualAnalysisOutput(BaseModel):
    """Output from epistemic_conceptual_analysis agent."""

    terms: list[str] = Field(
        description="Key terms being defined (parallel to definitions list)"
    )
    definitions: list[str] = Field(
        description="Working definition for each term (parallel to terms list)"
    )
    assumptions: list[str] = Field(description="Assumptions embedded in the question")
    context_summary: str = Field(
        description="2-3 sentence summary for downstream agents"
    )


class ClassifyQuestionOutput(BaseModel):
    """Output from epistemic_classify_question agent.

    Narrow agent: outputs a single enum + reasoning. This is the routing
    decision for the entire downstream pipeline.
    """

    question_type: str = Field(
        description=(
            'One of: "verificatory", "explanatory", "exploratory", '
            '"comparative", "predictive", "compositional", "normative"'
        )
    )
    reasoning: str = Field(
        description="One sentence explaining the classification choice"
    )


# ── Evidence ─────────────────────────────────────────────────────────────


class ExtractEvidenceOutput(BaseModel):
    """Output from epistemic_extract_evidence agent."""

    source_type: str = Field(
        description="Type of source - paper, dataset, note, conversation, webpage, book, report"
    )
    source_ref: str = Field(description="Reference to the source (URL, DOI, file path)")
    relevant_quotes: list[str] = Field(
        description="Key facts, findings, or quotes from the source content that are relevant to the objective."
    )
    experimental_context: str = Field(
        description="For empirical sources - describe the experimental setup, sample size, methodology. Empty string if not applicable."
    )
    limitations: list[str] = Field(
        description="Limitations, caveats, or scope restrictions mentioned in or inferred from the source."
    )


# ── Verification ─────────────────────────────────────────────────────────


class AssessEvidenceOutput(BaseModel):
    """Output from epistemic_assess_evidence agent (split scrutiny: evidence weight only)."""

    claim_id: str = Field(description="ID of the claim being assessed")
    evidence_weight: str = Field(description="strong, moderate, weak, or conflicting")
    confidence_estimate: float = Field(
        description="0.0-1.0 probability claim is true given evidence"
    )
    justification: str = Field(
        description="Brief explanation of why this weight was assigned"
    )


class IdentifySingleIssueOutput(BaseModel):
    """Output from epistemic_identify_single_issue agent.

    Flat output: one issue per call. Called in a loop with previously found
    issues passed as context to avoid duplicates.
    """

    has_issue: bool = Field(
        description="Whether there is an issue to report. Set to false if no more issues."
    )
    description: str = Field(
        description="What the issue is (empty string if has_issue is false)"
    )
    issue_type: str = Field(
        description="One of: unknown, contradiction, evidence_gap, risk, assumption, "
        "scope_difference, methodological_variation, definitional_variation, "
        "perspectival, evidence_corrupted. Empty string if has_issue is false."
    )


class DeductiveValidationOutput(BaseModel):
    """Output from epistemic_deductive_validation agent."""

    claim_id: str = Field(description="ID of the claim being validated")
    deductive_soundness: str = Field(
        description="Overall deductive assessment - sound, questionable, or unsound"
    )
    confidence_estimate: float = Field(
        description="Confidence in the deductive assessment (0.0-1.0)"
    )
    passes_deductive_validation: bool = Field(
        description="Whether the claim passes deductive validation. TRUE if deductive_soundness is sound or questionable without blocking issues."
    )
    issues_found: list[str] = Field(
        description="List of deductive issues found - logical inconsistencies, physical implausibilities, missing premises"
    )
    issue_types: list[str] = Field(
        description="Type for each issue. Use blocking types for genuine logical failures, assumption for acknowledged but non-fatal gaps."
    )
    recommendation: str = Field(
        description="One of - promote (deductively sound), hold (questionable, needs clarification), demote (deductively unsound)"
    )


class VerifyComputationallyOutput(BaseModel):
    """Output from epistemic_verify_computationally agent."""

    claim_id: str = Field(description="ID of the claim being verified")
    verification_code: str = Field(
        description="Complete Python code that tests the claim"
    )
    packages_required: list[str] = Field(
        description="Python packages needed to run the code (e.g., numpy, scipy)"
    )
    expected_behavior: str = Field(
        description="What the test should output if the claim is true"
    )
    test_description: str = Field(
        description="Human-readable explanation of what the test does"
    )


class AnalyzeArgumentOutput(BaseModel):
    """Output from epistemic_analyze_argument agent."""

    premises: list[str] = Field(description="Identified premises supporting the claim")
    conclusion: str = Field(description="The claim restated as a conclusion")
    validity: str = Field(
        description='Does conclusion follow from premises? "valid", "invalid", or "indeterminate"'
    )
    soundness: str = Field(
        description='Are premises true/supported? "sound", "unsound", or "questionable"'
    )
    fallacies: list[str] = Field(
        description='Logical fallacies detected (e.g., "correlation_causation", "hasty_generalization")'
    )


# ── Uncertainty ──────────────────────────────────────────────────────────


class ResolveUncertaintyOutput(BaseModel):
    """Output from epistemic_resolve_uncertainty agent."""

    uncertainty_id: str = Field(description="ID of the uncertainty being evaluated")
    can_resolve: bool = Field(
        description="Whether the uncertainty can now be resolved (true) or remains open (false)"
    )
    resolution: str = Field(
        description="How the uncertainty was resolved, if can_resolve is true"
    )
    remaining_concerns: list[str] = Field(
        description="Genuinely NEW concerns revealed by the evidence that are DIFFERENT from the "
        "original uncertainty. Do not restate the original limitation in different words. "
        "Empty list is the normal case."
    )


class InvestigateClaimQueryItem(BaseModel):
    """Single evidence query in investigate_claim output."""

    source_type: str = Field(
        description='Evidence provider to query (e.g., "openalex", "web_search", "all")'
    )
    query: str = Field(
        description="Natural language search query targeting the specific gap"
    )


class InvestigateClaimOutput(BaseModel):
    """Output from epistemic_investigate_claim agent."""

    evidence_queries: list[InvestigateClaimQueryItem] = Field(
        description="Targeted evidence searches to resolve scrutiny doubt"
    )
    reasoning: str = Field(
        description="Explanation of what evidence gaps were identified and why these queries target them"
    )


# ── Synthesis ────────────────────────────────────────────────────────────


class WriteAnswerOutput(BaseModel):
    """Output from epistemic_write_answer agent."""

    title: str = Field(description="A concise title for the research report")
    verdict: str = Field(
        default="",
        description="One sentence answering the research question — the bottom line",
    )
    answer: str = Field(description="A direct answer to the research question")


class AnswerValidation(BaseModel):
    """Output from epistemic_validate_answer agent."""

    approved: bool = Field(
        description="True if the answer faithfully represents the data, False if corrections needed"
    )
    feedback: list[str] = Field(
        description="Plain text corrections describing what needs fixing. Empty if approved."
    )


class IdentifyTestableAspectOutput(BaseModel):
    """Output from epistemic_identify_testable_aspect agent.

    Narrow agent: identifies one testable dimension of a claim.
    """

    testable_dimension: str = Field(
        description="What would be observably different if this claim is true vs false (one sentence)"
    )
    observation_type: str = Field(
        description='Type of observation: "quantitative", "qualitative", or "binary"'
    )


class SpecifyPredictionOutput(BaseModel):
    """Output from epistemic_specify_prediction agent.

    Narrow agent: specifies prediction details for one testable aspect.
    """

    expected_observation: str = Field(
        description="What should be observed if the claim is true"
    )
    conditions: str = Field(description="Under what conditions this prediction holds")
    timeframe: str = Field(description="When the observation should be possible")
    measurability: str = Field(description='"quantitative", "qualitative", or "binary"')


class DefineFalsificationOutput(BaseModel):
    """Output from epistemic_define_falsification agent.

    Narrow agent: defines what would disprove a prediction.
    """

    falsification_criterion: str = Field(
        description="What specific observation would disprove this prediction (one sentence)"
    )


class RecordDecisionOutput(BaseModel):
    """Output from epistemic_record_decision (epistemic_decide) agent."""

    statement: str = Field(description="What was decided - clear, actionable statement")
    justification: str = Field(
        description="Why this decision was made, referencing specific claims and evidence"
    )
    claim_indices: list[int] = Field(
        description="Indices of the claims this decision is based on"
    )
    reversible: bool = Field(
        description="Whether this decision can be reversed if new evidence emerges"
    )
    reversal_conditions: str = Field(
        description="Under what conditions would this decision be reconsidered?"
    )


# ── Focused Judgment Agents ─────────────────────────────────────────────


class GenerateCounterqueryOutput(BaseModel):
    """Output from epistemic_generate_counterquery agent.

    Narrow agent: generates one adversarial search query per call.
    """

    query: str = Field(
        description="Search query designed to find evidence AGAINST the claim"
    )
    framing: str = Field(
        description="What angle this query targets (e.g., 'replication failures', 'alternative explanations')"
    )


class EvaluateCounterargumentOutput(BaseModel):
    """Output from epistemic_evaluate_counterargument agent."""

    relevance: float = Field(
        description="0.0-1.0: Does this address the claim's specific assertions?"
    )
    specificity: float = Field(
        description="0.0-1.0: Is this targeted or a general objection?"
    )
    evidence_backed: float = Field(
        description="0.0-1.0: Does it cite evidence or is it speculative?"
    )
    source_credibility: float = Field(
        description="0.0-1.0: Is the source authoritative for this domain?"
    )
    category: str = Field(
        description="One of: methodological, empirical, logical, scope, statistical, theoretical, replication,"
        " alternative_explanation, ethical"
    )
    justification: str = Field(description="One sentence explaining the scores")


class ClassifyEvidenceDomainOutput(BaseModel):
    """Output from epistemic_classify_evidence_domain agent."""

    method_type: MethodType = Field(description="How was this knowledge generated?")
    data_source: DataSourceType = Field(description="What kind of data underlies this?")
    temporal_approach: TemporalApproach = Field(
        description="What is the time dimension of this evidence?"
    )
    causal_role: CausalRole = Field(
        description="What kind of causal claim does this evidence support?"
    )
    confidence: float = Field(
        description="0.0-1.0: Overall confidence in classification"
    )
    justification: str = Field(description="One sentence explaining the classification")


class ClassifyPredictionOutput(BaseModel):
    """Output from epistemic_classify_prediction agent."""

    prediction_type: str = Field(
        description="One of: quantitative, temporal, conditional, binary, qualitative"
    )
    specificity: float = Field(
        description="0.0-1.0: How specific and testable is this prediction?"
    )
    success_criteria: str = Field(description="What would confirm this prediction?")
    failure_criteria: str = Field(description="What would refute this prediction?")
    time_horizon: str = Field(
        description="Expected timeframe: immediate, short_term, medium_term, long_term, indefinite"
    )
    justification: str = Field(
        description="One sentence explaining type and specificity assessment"
    )


class AssessEvidenceQualityOutput(BaseModel):
    """Output from epistemic_assess_evidence_quality agent."""

    source_credibility: float = Field(
        description="0.0-1.0: Authority and reliability of the source"
        " (peer-reviewed journal > news > blog > anonymous)"
    )
    relevance: float = Field(
        description="0.0-1.0: How directly does this evidence address the claim?"
    )
    specificity: float = Field(
        description="0.0-1.0: How specific and detailed vs. vague and general?"
    )
    recency_appropriate: float = Field(
        description="0.0-1.0: Is this evidence current enough for the domain?"
    )
    justification: str = Field(
        description="One sentence explaining the quality assessment"
    )


class CheckPairwiseIndependenceOutput(BaseModel):
    """Output from epistemic_check_pairwise_independence agent.

    Narrow agent: checks if two evidence items are methodologically independent.
    """

    independent: bool = Field(
        description="Whether the two evidence items are from independent methods/groups/sources"
    )
    rationale: str = Field(
        description="One sentence explaining why they are or aren't independent"
    )


class ContrastiveEvaluationOutput(BaseModel):
    """Output from epistemic_contrastive_evaluation agent.

    Narrow agent: pairwise comparison of competing claims.
    """

    better_claim: str = Field(
        description='Which claim better explains the evidence: "A", "B", or "neither"'
    )
    distinguishing_observation: str = Field(
        description="One sentence: what single observation would distinguish between the claims"
    )
    confidence: float = Field(description="Confidence in the judgment (0.0-1.0)")


class CrossClaimConsistencyOutput(BaseModel):
    """Output from epistemic_cross_claim_consistency agent.

    Narrow agent: pairwise consistency check between claims.
    """

    conflicts: bool = Field(description="Whether the two claims contradict each other")
    tension_point: str = Field(
        description="One sentence identifying the specific premise in tension, or empty if no conflict"
    )


class SelectProviderOutput(BaseModel):
    """Output from epistemic_select_provider agent.

    Narrow binary judgment: is this provider relevant to this question?
    """

    relevant: bool = Field(
        description="True if this provider is likely to have relevant evidence for the question"
    )
    reasoning: str = Field(
        description="One sentence explaining why this provider is or is not relevant"
    )


class FormulateQueryOutput(BaseModel):
    """Output from epistemic_formulate_query agent.

    Narrow agent: produces one search query optimized for a specific provider.
    """

    query: str = Field(
        description="Search query optimized for this provider (5-15 words)"
    )
    rationale: str = Field(
        description="One sentence: why this query is appropriate for this provider"
    )


# ── Claim Proposal Decomposition ───────────────────────────────────────────


class ExtractAssertionOutput(BaseModel):
    """Output from epistemic_extract_assertion agent.

    Narrow agent: extracts one atomic factual assertion from one evidence item.
    """

    assertion: str = Field(
        description="One atomic factual assertion supported by this evidence (single sentence)"
    )


class ScreenRelevanceOutput(BaseModel):
    """Output from epistemic_screen_relevance agent.

    Simple yes/no relevance screening for one evidence item
    against a research question.
    """

    is_relevant: bool = Field(
        description="True if the evidence contains information that helps answer the research question"
    )
    reason: str = Field(
        description="One sentence explaining why this is or isn't relevant"
    )


class DraftClaimOutput(BaseModel):
    """Output from epistemic_draft_claim agent.

    Narrow agent: drafts one claim from a cluster of related assertions.
    """

    statement: str = Field(
        description="One claim statement capturing the shared content of the assertions"
    )
    scope: str = Field(description="Under what conditions this claim holds")
    direction: str = Field(
        description='"supports", "undermines", or "neutral" toward the research question'
    )


# ── Evidence Judgment ───────────────────────────────────────────────────


class EvidenceJudgmentOutput(BaseModel):
    """Output from epistemic_judge_evidence agent.

    Focused three-way classification: does this evidence support,
    contradict, or have no bearing on the claim?
    """

    verdict: Literal["supports", "contradicts", "no_bearing"] = Field(
        description='One of: "supports", "contradicts", "no_bearing"'
    )
    reasoning: str = Field(
        description="One sentence explaining why the evidence has this relationship to the claim"
    )


class IndependenceJudgmentOutput(BaseModel):
    """Output from epistemic_judge_independence agent.

    Binary judgment: are two evidence items methodologically independent?
    """

    independent: bool = Field(
        description="Whether the two evidence items could have arrived at their conclusions through different methods"
    )
    reasoning: str = Field(
        description="One sentence explaining why they are or aren't independent"
    )


# ── Similarity Validation ──────────────────────────────────────────────


class ValidateGroupOutput(BaseModel):
    """Output from epistemic_validate_group agent.

    Generic agent: validates whether grouped text items truly belong together.
    """

    subgroups: list[list[int]] = Field(
        description="List of subgroups after validation. Each subgroup is a list of item numbers (1-based). "
        "If all items belong together, return a single subgroup containing all item numbers. "
        "If items should be separated, return multiple subgroups."
    )


# ── Integration ─────────────────────────────────────────────────────────


class IntegrationAssessment(BaseModel):
    """Output of abductive integration: holistic evidence assessment."""

    verdict: Literal["supports", "contradicts", "insufficient"] = Field(
        description="'supports', 'contradicts', or 'insufficient'. "
        "Based on collective evidence weight, not individual counts."
    )
    confidence: float = Field(description="0.0-1.0 confidence in the verdict")
    reasoning: str = Field(description="The evidential chain explaining the verdict")
