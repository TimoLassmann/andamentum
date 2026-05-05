"""Output models for epistemic agents.

Each Pydantic BaseModel corresponds to the output_model from a .md agent manifest.
Field names MUST match what adapters.py accesses via attribute access.

Architecture: Layer 1 (framework-agnostic, pure Pydantic)
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from ..primitives import (
    CausalRole,
    CriticismCategory,
    DataSourceType,
    MethodType,
    PredictionType,
    QuestionType,
    TemporalApproach,
)

TimeHorizon = Literal[
    "immediate", "short_term", "medium_term", "long_term", "indefinite"
]


# ── Preplanning ──────────────────────────────────────────────────────────


class ClarifyQuestionOutput(BaseModel):
    """Output from epistemic_clarify_question agent."""

    ambiguity_level: Literal["clear", "moderate", "high"] = Field(
        description="How ambiguous is the original question?"
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

    question_type: QuestionType = Field(
        description="Epistemic question type — drives downstream verification routing"
    )
    reasoning: str = Field(
        description="One sentence explaining the classification choice"
    )


# ── Evidence ─────────────────────────────────────────────────────────────


class ExtractEvidenceOutput(BaseModel):
    """Output from epistemic_extract_evidence agent."""

    source_type: Literal[
        "paper", "dataset", "note", "conversation", "webpage", "book", "report"
    ] = Field(description="Type of source document")
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
    evidence_weight: Literal["strong", "moderate", "weak", "conflicting"] = Field(
        description="Overall evidential weight for the claim"
    )
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
    issue_type: Literal[
        "",
        "evidence_corrupted",
        "unknown",
        "contradiction",
        "evidence_gap",
        "risk",
        "assumption",
        "scope_difference",
        "methodological_variation",
        "definitional_variation",
        "perspectival",
    ] = Field(
        description=(
            "Issue classification. Empty string when has_issue is false. "
            '"evidence_corrupted" is a sentinel that triggers evidence '
            "invalidation rather than uncertainty creation; all other values "
            "map to UncertaintyType members."
        )
    )


class DeductiveValidationOutput(BaseModel):
    """Output from epistemic_deductive_validation agent."""

    claim_id: str = Field(description="ID of the claim being validated")
    deductive_soundness: Literal["sound", "questionable", "unsound"] = Field(
        description="Overall deductive assessment of the claim"
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
    recommendation: Literal["promote", "hold", "demote"] = Field(
        description=(
            "promote = deductively sound; "
            "hold = questionable, needs clarification; "
            "demote = deductively unsound"
        )
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
    validity: Literal["valid", "invalid", "indeterminate"] = Field(
        description="Does the conclusion follow from the premises?"
    )
    soundness: Literal["sound", "unsound", "questionable"] = Field(
        description="Are the premises true and well-supported?"
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
    observation_type: Literal["quantitative", "qualitative", "binary"] = Field(
        description="What kind of observation captures the testable difference?"
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
    measurability: Literal["quantitative", "qualitative", "binary"] = Field(
        description="How the observation will be measured"
    )


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
    category: CriticismCategory = Field(
        description=(
            "Category of criticism — must match a CriticismCategory enum "
            "member since downstream code keys weights off specific values "
            "(e.g. replication_failure, ad_hominem)."
        )
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

    prediction_type: PredictionType = Field(
        description="How is this prediction structured?"
    )
    specificity: float = Field(
        description="0.0-1.0: How specific and testable is this prediction?"
    )
    success_criteria: str = Field(description="What would confirm this prediction?")
    failure_criteria: str = Field(description="What would refute this prediction?")
    time_horizon: TimeHorizon = Field(
        description="Expected timeframe for observing the prediction"
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


class RankProvidersOutput(BaseModel):
    """Output from epistemic_rank_providers agent.

    Phase 2 of the lazy-escalation plan: pick the SINGLE best provider
    for a sub-claim from a list of candidates. Used in round 1 of the
    inquiry loop to narrow eager broad-search to lazy escalation —
    later rounds (driven by demand) can pull additional providers from
    the candidate list.

    Flat schema for small-LLM compatibility (single string + reasoning).
    """

    chosen_provider: str = Field(
        description=(
            "Name of the single best provider for this sub-claim, "
            "chosen from the provided candidate list. Must match one "
            "of the candidate names exactly (no paraphrasing)."
        )
    )
    reasoning: str = Field(
        description=(
            "One-sentence justification: why this provider is most "
            "likely to give a high-information-density answer for the "
            "specific sub-claim being investigated."
        )
    )


class FormulateQueryOutput(BaseModel):
    """Output from epistemic_formulate_query agent.

    Narrow agent: produces one search query optimized for a specific provider.
    """

    query: str = Field(description="Search query in the provider's native syntax")
    rationale: str = Field(
        description="One sentence: why this query style fits this question and provider"
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
    direction: Literal["supports", "undermines", "neutral"] = Field(
        description="The claim's stance toward the research question"
    )


# ── Evidence Judgment ───────────────────────────────────────────────────


class EvidenceJudgmentOutput(BaseModel):
    """Output from epistemic_judge_evidence agent.

    Structured scope-then-direction decomposition. The agent first decides
    whether the evidence falls within the claim's scope (population,
    condition, context, qualifier match) and only then judges direction.
    Surfacing the scope step makes failures debuggable: when the verdict is
    wrong, the JSONL trace shows whether the scope check or the direction
    check broke.

    The pydantic schema enforces the decomposition — small models cannot
    skip the scope reasoning by going straight to a verdict.
    """

    claim_scope_summary: str = Field(
        description=(
            "Short phrase describing what the claim covers, including any "
            "qualifiers or conditions. Examples: 'podocytes under injury', "
            "'new TB drugs, lesion penetration', 'stage III NSCLC patients'."
        )
    )
    evidence_scope_summary: str = Field(
        description=(
            "Short phrase describing what the evidence actually studies. "
            "Examples: 'healthy mouse podocytes at baseline', 'BTZ-043 in "
            "murine TB granulomas', 'all-comers cohort over 65'."
        )
    )
    in_scope: bool = Field(
        description=(
            "True if the evidence's scope falls within (or is a specific "
            "instance of) the claim's scope. False if topically related but "
            "pertaining to a different population, condition, or context. "
            "When False, the verdict MUST be 'no_bearing'."
        )
    )
    verdict: Literal["supports", "contradicts", "no_bearing"] = Field(
        description=(
            "If in_scope is False, MUST be 'no_bearing'. If in_scope is "
            "True, judge direction: 'supports' (evidence makes the claim "
            "more likely true) or 'contradicts' (evidence makes the claim "
            "less likely true)."
        )
    )
    reasoning: str = Field(
        description=(
            "One sentence justifying the verdict, referencing the scope "
            "analysis. For out-of-scope cases, name the scope mismatch."
        )
    )

    @model_validator(mode="after")
    def _verdict_consistent_with_scope(self) -> "EvidenceJudgmentOutput":
        """Cross-field invariant: ``in_scope`` and ``verdict`` must agree.

        The judge prompt states this rule (Step 4: "If in_scope is False,
        verdict MUST be 'no_bearing'"). The schema previously did not
        enforce it, so the LLM could return logically inconsistent
        outputs like ``in_scope=True, verdict='no_bearing'`` ("this is
        on-topic but I can't tell which way it leans") which the prompt
        explicitly forbids.

        Raising ValueError here triggers pydantic-ai's output-retry
        mechanism (the agent is registered with output_retries=3 in
        ``judge.py``) — the model gets the validation error message
        as feedback and retries.
        """
        if self.in_scope and self.verdict == "no_bearing":
            raise ValueError(
                "in_scope=True is incompatible with verdict='no_bearing'. "
                "If the evidence is in scope, set verdict to 'supports' "
                "or 'contradicts'. If the evidence really has no bearing, "
                "set in_scope=False."
            )
        if not self.in_scope and self.verdict != "no_bearing":
            raise ValueError(
                f"in_scope=False requires verdict='no_bearing' (got "
                f"{self.verdict!r}). Out-of-scope evidence cannot support "
                "or contradict a claim — set verdict='no_bearing' or "
                "reconsider whether the evidence is actually in scope."
            )
        return self


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
    """Output of abductive integration: holistic evidence assessment.

    Deprecated: kept for backwards-compat with snapshots from runs prior
    to the IBE-decomposed integration. New runs use the 4-stage IBE
    pipeline (NextCandidate, LovelinessScore, LikelinessScore,
    SelectedExplanation).
    """

    verdict: Literal["supports", "contradicts", "insufficient"] = Field(
        description="'supports', 'contradicts', or 'insufficient'. "
        "Based on collective evidence weight, not individual counts."
    )
    confidence: float = Field(description="0.0-1.0 confidence in the verdict")
    reasoning: str = Field(description="The evidential chain explaining the verdict")


# ── IBE Integration (4-stage decomposition) ──────────────────────────
#
# The integration step is decomposed into Peirce-style generative
# enumeration, two Lipton-style evaluative scoring agents (loveliness +
# likeliness, applied independently per candidate), and a comparative
# selection agent. Each agent's output schema is intentionally small
# so local models can produce structured outputs reliably; sub-virtue
# detail (mechanism / scope / parsimony) lives in prompt rubrics and
# in the reasoning text rather than as structured fields.


class NextCandidate(BaseModel):
    """One candidate verdict from iterative Peircean enumeration.

    Either ``done=True`` (no further meaningful candidate exists) or
    a candidate is provided. The orchestrator assigns IDs (A, B, ...)
    and runs this agent up to a hard cap; the agent never sees its own
    prior outputs except as ``already_proposed`` context.
    """

    done: bool = Field(
        description=(
            "True if no further meaningful candidate verdict can be added "
            "given the candidates already proposed. When True, verdict and "
            "description should be omitted."
        )
    )
    verdict: Optional[
        Literal[
            "supports",
            "contradicts",
            "insufficient",
            "supports_refined",
            "contradicts_refined",
        ]
    ] = Field(
        default=None,
        description=(
            "The candidate verdict. Use 'supports_refined' or "
            "'contradicts_refined' when the candidate is a directional "
            "verdict that holds only under a narrower scope than the "
            "claim's stated scope (e.g. 'true for one drug class only')."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        description=(
            "1-2 sentences explaining how this candidate would account for "
            "the evidence pattern. Must be distinct from already-proposed "
            "candidates."
        ),
    )


class LovelinessScore(BaseModel):
    """Lipton's loveliness: how good an explanation a candidate would be IF true.

    Evaluates explanatory virtue independently of the other candidates'
    scores (Kahneman independence). The agent sees one candidate at a
    time and never another candidate's score.
    """

    loveliness: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Loveliness in [0, 1]. High loveliness means the candidate, "
            "if true, would be a clean explanation: clear mechanism, "
            "matching scope, parsimony, and unifying power across the "
            "evidence pattern."
        ),
    )
    reasoning: str = Field(
        description=(
            "One paragraph touching on each loveliness virtue (mechanism "
            "clarity, scope match, parsimony, unifying power). The "
            "reasoning is the audit trail for the score."
        )
    )


class LikelinessScore(BaseModel):
    """Lipton's likeliness: how well a candidate fits the actual evidence.

    Evaluated independently of other candidates' scores (Kahneman
    independence). The agent sees one candidate at a time alongside the
    full evidence base and adversarial outcome; it does not see other
    candidates' likeliness scores.
    """

    likeliness: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Likeliness in [0, 1]. High likeliness means the candidate "
            "accounts for the supporting items, correctly handles or "
            "dismisses the contradicting items, and is consistent with "
            "the adversarial-search outcome."
        ),
    )
    reasoning: str = Field(
        description=(
            "One paragraph naming which evidence pieces the candidate "
            "explains and which it cannot account for. Specific is more "
            "useful than vague."
        )
    )


class SelectedExplanation(BaseModel):
    """Lipton's comparative selection: best candidate by loveliness × likeliness.

    The agent sees all scored candidates with their loveliness and
    likeliness values + reasonings, and picks the best. Confidence
    should reflect the *gap* to the runner-up: large gap → high
    confidence; small gap → moderate confidence (the literature is
    contested or the candidates are similarly defensible).
    """

    chosen_candidate_id: str = Field(
        description="The candidate_id (e.g. 'A', 'B') of the best explanation."
    )
    runner_up_candidate_id: str = Field(
        description=(
            "The candidate_id of the second-best explanation. Required "
            "even when the runner-up is much weaker — the gap between "
            "best and runner-up is the basis for confidence calibration."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the chosen verdict. Calibrate against the gap "
            "to the runner-up: large gap (chosen dominates on both "
            "loveliness and likeliness) → high confidence (>0.8); small "
            "gap (candidates similarly defensible) → moderate (0.4-0.6)."
        ),
    )
    reasoning: str = Field(
        description=(
            "One paragraph explaining why the chosen candidate beats the "
            "runner-up, and what the gap implies about confidence."
        )
    )


# ── Question Decomposition (top-down inquiry structure) ──────────────────
#
# Replaces the bottom-up assertion-clustering path of ProposeClaimsOperation
# for verificatory / exploratory / predictive questions. The decomposition
# step identifies the load-bearing structure of a research question
# *before* gathering evidence — what 2-5 sub-investigations would, together,
# settle or characterize the question. Each sub-investigation then runs
# through the per-claim pipeline.
#
# Schema deliberately slim (3 fields per sub-investigation, 3 fields at
# the top level) so local models can produce structured outputs reliably.
# Question-type-specific framing (testable claim vs facet vs predictive
# condition) lives in the prompt; the schema is uniform across types.


class SubInvestigation(BaseModel):
    """One sub-investigation in a top-down question decomposition.

    Each sub-investigation has the form of a checkable claim — testable
    for verificatory questions, characterizable for exploratory questions,
    or condition-like for predictive questions. The pipeline treats each
    one the same way (seed_claim flow), so the uniform schema is correct
    even when the question-type semantics differ.
    """

    id: str = Field(
        description="Stable identifier for this sub-investigation: 'A', 'B', 'C', ..."
    )
    seed_claim: str = Field(
        description=(
            "The sub-investigation expressed as a testable / characterizable "
            "claim. For verificatory questions, this is a falsifiable sub-claim "
            "whose truth would partially settle the original question. For "
            "exploratory questions, this is a facet-claim (e.g. 'X is well-"
            "characterized in dimension Y'). For predictive questions, this "
            "is a condition-claim. The pipeline runs the same machinery "
            "regardless."
        )
    )
    rationale: str = Field(
        description=(
            "One sentence: why this sub-investigation is load-bearing for "
            "the original question. What does its outcome tell us that "
            "another sub-investigation's outcome wouldn't?"
        )
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description=(
            "Relative importance of this sub-investigation to the question's "
            "answer, on a 0-10 scale. Used by the WEIGHTED_AND combination "
            "rule (weighted mean of child posteriors). For AND / OR / UNION "
            "the weight is ignored. Default 1.0 makes WEIGHTED_AND degenerate "
            "to a simple mean. Use values like 2.0 / 0.5 to express 'this "
            "sub-investigation is twice as critical / half as critical' as "
            "the others."
        ),
    )


class QuestionDecomposition(BaseModel):
    """Top-down decomposition of a research question into sub-investigations.

    A good decomposition has 2-5 sub-investigations that are:
    - Load-bearing (each one's outcome materially affects the answer)
    - Roughly orthogonal (investigating one doesn't trivialize another)
    - Cover the question's scope (a complete answer is reachable from
      their combined outcomes)

    The combination_rule tells downstream synthesis how to aggregate
    sub-investigation verdicts into the final answer for the question.
    """

    sub_investigations: list[SubInvestigation] = Field(
        description=(
            "2-5 sub-investigations. Fewer than 2 means the decomposition "
            "didn't actually decompose; more than 5 typically means "
            "over-fragmentation."
        ),
        min_length=2,
        max_length=5,
    )
    combination_rule: Literal["AND", "OR", "WEIGHTED_AND", "UNION"] = Field(
        description=(
            "How sub-investigation outcomes combine into the question's "
            "answer. AND: all must support for the question to support. "
            "OR: any one supports. WEIGHTED_AND: each contributes by "
            "importance. UNION: each contributes a piece of the answer "
            "(typical for exploratory questions where there is no single "
            "verdict, just a structured characterization)."
        )
    )
    rationale: str = Field(
        description=(
            "1-2 sentences explaining why this decomposition captures the "
            "question's load-bearing structure."
        )
    )
