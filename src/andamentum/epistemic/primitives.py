"""Epistemic primitives - Domain models and re-exports.

Core entities (Evidence, Claim, Uncertainty, Decision, Objective, Snapshot, Artefact)
are defined in entities/ and re-exported here for backward compatibility.

This module defines domain-specific models that are NOT entities:
- Scheduling: WorkItem, WorkItemStatus, WorkItemType
- Audit: EpistemicEvent
- Pre-planning: ClarifiedQuestion, ConceptualAnalysis, ArgumentAnalysis, PlanArguments
- Verification: ComputationalEvidence, AdversarialEvidence, ConvergentEvidence, etc.

Architecture: Layer 1 (framework-agnostic, no model calls)
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


# --- Entity re-exports (canonical definitions in entities/) ---
from .entities.claim import ClaimStage, Claim  # noqa: F401
from .entities.evidence import Evidence  # noqa: F401
from .entities.uncertainty import Uncertainty, UncertaintyType, UncertaintyScope  # noqa: F401
from .entities.decision import Decision  # noqa: F401
from .entities.objective import Objective  # noqa: F401
from .entities.snapshot import Snapshot  # noqa: F401
from .entities.artefact import Artefact  # noqa: F401


# --- Enums ---


class WorkItemStatus(str, Enum):
    """WorkItem lifecycle states."""

    QUEUED = "queued"  # Ready to execute when dependencies met
    RUNNING = "running"  # Currently executing
    DONE = "done"  # Completed successfully
    BLOCKED = "blocked"  # Waiting on dependencies
    FAILED = "failed"  # Execution failed


class WorkItemType(str, Enum):
    """Types of epistemic operations.

    Each type maps to an executor agent in the registry.
    """

    # Pre-planning steps (run before PLAN_TASK)
    CLARIFY_QUESTION = "clarify_question"
    CONCEPTUAL_ANALYSIS = "conceptual_analysis"

    # Core workflow
    PLAN_TASK = "plan_task"
    COLLECT_EVIDENCE = "collect_evidence"
    EXTRACT_EVIDENCE = "extract_evidence"
    PROPOSE_CLAIMS = "propose_claims"
    WORLD_KNOWLEDGE_CLAIMS = (
        "world_knowledge_claims"  # Generate claims from LLM world knowledge (fallback)
    )
    SCRUTINISE_CLAIM = "scrutinise_claim"
    ANALYZE_ARGUMENT = "analyze_argument"  # Formal argument structure analysis (separate from scrutiny)
    VERIFY_COMPUTATIONALLY = "verify_computationally"  # Test claims by running code
    PROMOTE_CLAIM = "promote_claim"
    FREEZE_SNAPSHOT = "freeze_snapshot"
    SYNTHESIZE_REPORT = "synthesize_report"
    DECIDE = "decide"  # Record decisions based on actionable claims
    DEMOTE_CLAIM = (
        "demote_claim"  # Demote claim to lower stage when evidence contradicted
    )
    RESOLVE_UNCERTAINTY = "resolve_uncertainty"  # Mark uncertainty as resolved
    REVERSE_DECISION = "reverse_decision"  # Reverse decision when circumstances change

    # Independence verification methods
    ADVERSARIAL_SEARCH = (
        "adversarial_search"  # Seek disconfirming evidence and counterarguments
    )
    ASSESS_CONVERGENCE = "assess_convergence"  # Assess cross-domain convergence
    GENERATE_PREDICTION = (
        "generate_prediction"  # Generate testable predictions from claims
    )
    RESOLVE_PREDICTION = "resolve_prediction"  # Resolve predictions against outcomes

    # Deductive validation track (parallel to inductive evidence)
    VALIDATE_DEDUCTIVELY = (
        "validate_deductively"  # First principles, consistency, plausibility checks
    )


class VerifiabilityType(str, Enum):
    """Types of claim verifiability for computational verification.

    Used by the claim classifier to determine how a claim can be verified.
    """

    COMPUTATIONALLY_VERIFIABLE = (
        "computationally_verifiable"  # Can be tested by running code
    )
    SIMULATION_VERIFIABLE = (
        "simulation_verifiable"  # Can be tested against a domain simulator
    )
    TEXTUALLY_VERIFIABLE = (
        "textually_verifiable"  # Can only be corroborated by other text
    )
    HUMAN_VERIFIABLE = "human_verifiable"  # Requires human judgment (normative claims)


class QuestionType(str, Enum):
    """Epistemic question type for verification track routing.

    Each type activates a different profile of verification tracks
    and applies different stage gate criteria. Domain-independent:
    applies regardless of whether the question is about genomics,
    economics, or philosophy.
    """

    VERIFICATORY = "verificatory"  # "Is P true?"
    EXPLANATORY = "explanatory"  # "Why P?" / "How does P work?"
    EXPLORATORY = "exploratory"  # "What might be involved in P?"
    COMPARATIVE = "comparative"  # "Is A better/more likely than B?"
    PREDICTIVE = "predictive"  # "What will happen if P?"
    COMPOSITIONAL = "compositional"  # "What are the parts/factors of X?"
    NORMATIVE = "normative"  # "Should we do X?"


# --- Pre-Planning Primitives ---


class ClarifiedQuestion(BaseModel):
    """Result of question clarification step.

    Runs before PLAN_TASK to disambiguate questions and identify key terms.
    Works autonomously with a detect-then-decide approach.
    """

    clarification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    objective_id: str = Field(
        description="Parent objective this clarification belongs to"
    )

    # Core output (matches agent's 4 fields)
    original_question: str = Field(
        description="The original question before clarification"
    )
    clarified_question: str = Field(description="Rewritten unambiguous question")
    ambiguity_level: str = Field(
        description="Level of ambiguity: 'clear', 'moderate', or 'high'"
    )
    key_terms: List[str] = Field(
        default_factory=list, description="Terms that need explicit definition"
    )
    reasoning: str = Field(
        default="", description="Explanation of interpretation choice"
    )

    created_at: datetime = Field(default_factory=datetime.now)

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to DocumentStore metadata format."""
        return {
            "epistemic_type": "clarified_question",
            "clarification_id": self.clarification_id,
            "objective_id": self.objective_id,
            "original_question": self.original_question,
            "clarified_question": self.clarified_question,
            "ambiguity_level": self.ambiguity_level,
            "key_terms": self.key_terms,
            "created_at": self.created_at.isoformat(),
        }


class ConceptualAnalysis(BaseModel):
    """Result of conceptual analysis step.

    Runs after CLARIFY_QUESTION to define terms and surface embedded assumptions.
    Provides context summary for all downstream agents.
    """

    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    objective_id: str = Field(description="Parent objective this analysis belongs to")
    clarification_id: Optional[str] = Field(
        None, description="Link to prior clarification if any"
    )

    # Core output (matches agent's 4 fields - parallel lists)
    terms: List[str] = Field(
        default_factory=list, description="Key terms being defined"
    )
    definitions: List[str] = Field(
        default_factory=list,
        description="Working definition for each term (parallel to terms)",
    )
    assumptions: List[str] = Field(
        default_factory=list, description="Assumptions embedded in the question"
    )
    context_summary: str = Field(
        default="", description="2-3 sentence summary for downstream agents"
    )

    created_at: datetime = Field(default_factory=datetime.now)

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to DocumentStore metadata format."""
        return {
            "epistemic_type": "conceptual_analysis",
            "analysis_id": self.analysis_id,
            "objective_id": self.objective_id,
            "clarification_id": self.clarification_id,
            "terms": self.terms,
            "definitions": self.definitions,
            "assumptions": self.assumptions,
            "context_summary": self.context_summary,
            "created_at": self.created_at.isoformat(),
        }


class ArgumentAnalysis(BaseModel):
    """Formal argument structure analysis.

    Runs alongside or before SCRUTINISE_CLAIM to analyze logical structure.
    Separate from scrutiny to keep both agents simple.
    """

    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str = Field(description="The claim being analyzed")
    objective_id: str = Field(description="Parent objective")

    # Core output (matches agent's 5 fields)
    premises: List[str] = Field(
        default_factory=list, description="Identified premises supporting the claim"
    )
    conclusion: str = Field(
        default="", description="The claim restated as a conclusion"
    )
    validity: str = Field(
        default="indeterminate",
        description="Does conclusion follow from premises? 'valid', 'invalid', or 'indeterminate'",
    )
    soundness: str = Field(
        default="questionable",
        description="Are premises true/supported? 'sound', 'unsound', or 'questionable'",
    )
    fallacies: List[str] = Field(
        default_factory=list,
        description="Logical fallacies detected (e.g., 'correlation_causation')",
    )

    created_at: datetime = Field(default_factory=datetime.now)

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to DocumentStore metadata format."""
        return {
            "epistemic_type": "argument_analysis",
            "analysis_id": self.analysis_id,
            "claim_id": self.claim_id,
            "objective_id": self.objective_id,
            "premises": self.premises,
            "conclusion": self.conclusion,
            "validity": self.validity,
            "soundness": self.soundness,
            "fallacies": self.fallacies,
            "created_at": self.created_at.isoformat(),
        }


class ComputationalEvidence(BaseModel):
    """Evidence from computational verification.

    Dual-execution architecture:
    1. Codex generates and tests verification code (LLM-assisted)
    2. Deterministic executor re-runs for reproducibility verification (no LLM)

    This creates a closed prediction-outcome loop that provides stronger
    evidence than textual sources alone.
    """

    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str = Field(description="The claim being verified")
    objective_id: str = Field(description="Parent objective")

    # Code and execution
    verification_code: str = Field(description="The Python code that tests the claim")
    packages_required: List[str] = Field(
        default_factory=list, description="Python packages needed"
    )

    # Phase 1: Codex execution results
    codex_verdict: str = Field(
        description="Result from Codex run: 'passed', 'failed', 'error'"
    )
    codex_output: str = Field(default="", description="Raw output from Codex execution")

    # Phase 2: Deterministic re-execution results
    deterministic_verdict: str = Field(
        default="pending", description="'matched', 'mismatched', 'error', 'pending'"
    )
    deterministic_output: str = Field(
        default="", description="Raw output from deterministic re-run"
    )

    # Final interpretation
    final_verdict: str = Field(
        description="'SUPPORTED', 'REFUTED', 'INCONCLUSIVE', 'INVALID_TEST'"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in the verdict"
    )
    measurements: Dict[str, Any] = Field(
        default_factory=dict, description="Extracted quantitative data"
    )
    reproducible: bool = Field(
        default=False, description="True if deterministic executor matched Codex"
    )
    explanation: str = Field(
        default="", description="Human-readable explanation of the verification"
    )

    created_at: datetime = Field(default_factory=datetime.now)

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to DocumentStore metadata format."""
        return {
            "epistemic_type": "computational_evidence",
            "evidence_id": self.evidence_id,
            "claim_id": self.claim_id,
            "objective_id": self.objective_id,
            "codex_verdict": self.codex_verdict,
            "deterministic_verdict": self.deterministic_verdict,
            "final_verdict": self.final_verdict,
            "confidence": self.confidence,
            "reproducible": self.reproducible,
            "created_at": self.created_at.isoformat(),
        }


# --- Adversarial Search Primitives ---


class CriticismCategory(str, Enum):
    """Categories of criticism discovered through adversarial search.

    Categories have different weights for evidence calculation.
    From spec Part 4.2.
    """

    METHODOLOGICAL = "methodological"  # Study design was flawed (High weight)
    STATISTICAL = "statistical"  # Analysis was incorrect (High weight)
    REPLICATION_FAILURE = (
        "replication_failure"  # Finding didn't replicate (Very High weight)
    )
    CONFOUNDING = "confounding"  # Effect explained by other variable (High weight)
    GENERALIZATION = "generalization"  # Finding doesn't generalize (Medium weight)
    INTERPRETATION = "interpretation"  # Data doesn't support conclusion (Medium weight)
    THEORETICAL = "theoretical"  # Theoretical framework is wrong (Medium weight)
    FRINGE = "fringe"  # Non-mainstream criticism (Low weight)
    AD_HOMINEM = (
        "ad_hominem"  # Attack on researcher, not research (Zero weight - filter out)
    )


# Category weights for evidence calculation
CRITICISM_CATEGORY_WEIGHTS: Dict[CriticismCategory, float] = {
    CriticismCategory.REPLICATION_FAILURE: 4.0,  # Very High
    CriticismCategory.METHODOLOGICAL: 3.0,  # High
    CriticismCategory.STATISTICAL: 3.0,  # High
    CriticismCategory.CONFOUNDING: 3.0,  # High
    CriticismCategory.GENERALIZATION: 2.0,  # Medium
    CriticismCategory.INTERPRETATION: 2.0,  # Medium
    CriticismCategory.THEORETICAL: 2.0,  # Medium
    CriticismCategory.FRINGE: 0.5,  # Low
    CriticismCategory.AD_HOMINEM: 0.0,  # Zero - should be filtered
}


class CounterargumentQuality(BaseModel):
    """Quality assessment of a counterargument.

    From spec Part 4.1: Assess counterarguments on 5 dimensions.
    Minimum threshold: combined score >= 2.5.
    """

    relevance: float = Field(ge=0.0, le=1.0, description="Does this address the claim?")
    specificity: float = Field(ge=0.0, le=1.0, description="Is this specific or vague?")
    evidence_backed: float = Field(ge=0.0, le=1.0, description="Does it cite evidence?")
    source_credibility: float = Field(
        ge=0.0, le=1.0, description="Is the critic qualified?"
    )
    novelty: float = Field(
        ge=0.0, le=1.0, description="Is this new vs already-addressed?"
    )

    @property
    def combined_score(self) -> float:
        """Combined quality score (sum of dimensions)."""
        return (
            self.relevance
            + self.specificity
            + self.evidence_backed
            + self.source_credibility
            + self.novelty
        )

    @property
    def passes_threshold(self) -> bool:
        """Minimum threshold: combined score >= 2.5."""
        return self.combined_score >= 2.5


class Counterargument(BaseModel):
    """A structured counterargument against a claim.

    Extracted from adversarial search results.
    """

    counterargument_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str = Field(description="The claim this counterargument addresses")

    # Content
    summary: str = Field(description="What is the criticism?")
    source_ref: str = Field(description="URL/DOI of the criticism source")
    source_author: Optional[str] = Field(
        None, description="Who is making this criticism?"
    )
    supporting_evidence: str = Field(
        default="", description="What evidence supports the criticism?"
    )

    # Classification
    category: CriticismCategory = Field(description="Type of criticism")
    quality: CounterargumentQuality = Field(description="Quality assessment")
    match_strength: str = Field(description="'strong', 'partial', 'weak', or 'none'")

    # Weighting
    weight: float = Field(
        default=0.0, description="Evidence weight based on quality and category"
    )

    created_at: datetime = Field(default_factory=datetime.now)

    def compute_weight(self) -> float:
        """Compute evidence weight based on quality and category."""
        if not self.quality.passes_threshold:
            return 0.0
        category_weight = CRITICISM_CATEGORY_WEIGHTS.get(self.category, 1.0)
        quality_factor = self.quality.combined_score / 5.0  # Normalize to 0-1
        match_factor = {"strong": 1.0, "partial": 0.6, "weak": 0.3, "none": 0.0}.get(
            self.match_strength, 0.5
        )
        return category_weight * quality_factor * match_factor

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to DocumentStore metadata format."""
        return {
            "epistemic_type": "counterargument",
            "counterargument_id": self.counterargument_id,
            "claim_id": self.claim_id,
            # Full content for reconstruction
            "summary": self.summary,
            "source_ref": self.source_ref,
            "source_author": self.source_author,
            "supporting_evidence": self.supporting_evidence,
            # Classification
            "category": self.category.value,
            "match_strength": self.match_strength,
            "weight": self.weight,
            # Full quality object for reconstruction
            "quality": {
                "relevance": self.quality.relevance,
                "specificity": self.quality.specificity,
                "evidence_backed": self.quality.evidence_backed,
                "source_credibility": self.quality.source_credibility,
                "novelty": self.quality.novelty,
            },
            "quality_score": self.quality.combined_score,
            "created_at": self.created_at.isoformat(),
        }


class AdversarialEvidence(BaseModel):
    """Evidence from adversarial search verification.

    Provides source independence by actively seeking disconfirming evidence.
    A claim that survives adversarial search is more credible.
    """

    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str = Field(description="The claim being adversarially tested")
    objective_id: str = Field(description="Parent objective")

    # Search metadata
    queries_used: List[str] = Field(
        default_factory=list, description="Adversarial queries executed"
    )
    sources_searched: int = Field(default=0, description="Number of sources examined")

    # Results
    counterarguments: List[Counterargument] = Field(
        default_factory=list, description="Discovered counterarguments"
    )
    supporting_weight: float = Field(
        default=0.0, description="Sum of supporting evidence weights"
    )
    adversarial_weight: float = Field(
        default=0.0, description="Sum of adversarial evidence weights"
    )

    # Balance calculation (from spec Part 6.3)
    adversarial_balance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="supporting / (supporting + adversarial)",
    )

    # Verdict
    verdict: str = Field(
        description="'SUPPORTED', 'CONTESTED', 'CHALLENGED', 'REFUTED'"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = Field(default="", description="Overall adversarial assessment")

    # Recommendation
    recommendation: str = Field(description="'maintain', 'weaken', 'refute', 'modify'")
    suggested_modifications: Optional[str] = Field(
        None, description="Claim modifications if 'modify'"
    )

    created_at: datetime = Field(default_factory=datetime.now)

    def calculate_balance(self) -> float:
        """Calculate adversarial balance score.

        From spec Part 6.3:
        balance = supporting / (supporting + adversarial)

        Interpretation:
        - > 0.8: Strongly supported
        - 0.6-0.8: Moderately supported
        - 0.4-0.6: Contested
        - 0.2-0.4: Weakly supported / likely false
        - < 0.2: Strongly challenged
        """
        total = self.supporting_weight + self.adversarial_weight
        if total == 0:
            return 0.5  # No evidence either way
        return self.supporting_weight / total

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to DocumentStore metadata format."""
        return {
            "epistemic_type": "adversarial_evidence",
            "evidence_id": self.evidence_id,
            "claim_id": self.claim_id,
            "objective_id": self.objective_id,
            # Full data for retrieval
            "queries_used": self.queries_used,
            "sources_searched": self.sources_searched,
            "counterarguments": [ca.to_metadata() for ca in self.counterarguments],
            "supporting_weight": self.supporting_weight,
            "adversarial_weight": self.adversarial_weight,
            "adversarial_balance": self.adversarial_balance,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "suggested_modifications": self.suggested_modifications,
            "created_at": self.created_at.isoformat(),
            # Summary fields for quick queries
            "queries_count": len(self.queries_used),
            "counterarguments_count": len(self.counterarguments),
        }

    @classmethod
    def from_metadata(cls, meta: Dict[str, Any]) -> "AdversarialEvidence":
        """Reconstruct from DocumentStore metadata."""
        from datetime import datetime

        # Reconstruct counterarguments
        counterarguments = []
        for ca_meta in meta.get("counterarguments", []):
            counterarguments.append(
                Counterargument(
                    counterargument_id=ca_meta.get("counterargument_id", ""),
                    claim_id=ca_meta.get("claim_id", ""),
                    summary=ca_meta.get("summary", ""),
                    source_ref=ca_meta.get("source_ref", ""),
                    source_author=ca_meta.get("source_author"),
                    supporting_evidence=ca_meta.get("supporting_evidence", ""),
                    category=CriticismCategory(
                        ca_meta.get("category", "factual_dispute")
                    ),
                    quality=CounterargumentQuality(
                        relevance=ca_meta.get("quality", {}).get("relevance", 0.0),
                        specificity=ca_meta.get("quality", {}).get("specificity", 0.0),
                        evidence_backed=ca_meta.get("quality", {}).get(
                            "evidence_backed", 0.0
                        ),
                        source_credibility=ca_meta.get("quality", {}).get(
                            "source_credibility", 0.0
                        ),
                        novelty=ca_meta.get("quality", {}).get("novelty", 0.0),
                    ),
                    match_strength=ca_meta.get("match_strength", "none"),
                    weight=ca_meta.get("weight", 0.0),
                )
            )

        return cls(
            evidence_id=meta.get("evidence_id", ""),
            claim_id=meta.get("claim_id", ""),
            objective_id=meta.get("objective_id", ""),
            queries_used=meta.get("queries_used", []),
            sources_searched=meta.get("sources_searched", 0),
            counterarguments=counterarguments,
            supporting_weight=meta.get("supporting_weight", 0.0),
            adversarial_weight=meta.get("adversarial_weight", 0.0),
            adversarial_balance=meta.get("adversarial_balance", 0.5),
            verdict=meta.get("verdict", "SUPPORTED"),
            confidence=meta.get("confidence", 0.0),
            explanation=meta.get("explanation", ""),
            recommendation=meta.get("recommendation", "maintain"),
            suggested_modifications=meta.get("suggested_modifications"),
            created_at=datetime.fromisoformat(
                meta.get("created_at", datetime.now().isoformat())
            ),
        )


# --- Cross-Domain Convergence Primitives ---


class MethodType(str, Enum):
    """How was this knowledge generated?

    Critical for independence: different methods have different error modes.
    """

    EXPERIMENTAL = "experimental"  # Active intervention or manipulation
    OBSERVATIONAL = "observational"  # Passive observation without intervention
    COMPUTATIONAL = "computational"  # Model-based inference or simulation
    THEORETICAL = "theoretical"  # Logical or mathematical derivation


class DataSourceType(str, Enum):
    """Where did the data come from?"""

    PRIMARY = "primary"  # New data collected specifically for this study
    SECONDARY = "secondary"  # Existing data reanalyzed
    SYNTHETIC = "synthetic"  # Generated or simulated data
    META = "meta"  # Aggregation of multiple existing studies


class TemporalApproach(str, Enum):
    """How does time factor in?"""

    CROSS_SECTIONAL = "cross_sectional"  # Single time point snapshot
    LONGITUDINAL = "longitudinal"  # Tracked over time
    RETROSPECTIVE = "retrospective"  # Looking back at past events
    PROSPECTIVE = "prospective"  # Looking forward to future events


class CausalRole(str, Enum):
    """What kind of claim is being made?"""

    MECHANISTIC = "mechanistic"  # How something works (the mechanism)
    PHENOMENOLOGICAL = "phenomenological"  # What is observed to happen
    INTERVENTIONAL = "interventional"  # What happens when we intervene
    PREDICTIVE = "predictive"  # What will happen in the future


# Dimension weights for domain distance calculation
# These 4 dimensions capture different error modes without being domain-specific
DOMAIN_DIMENSION_WEIGHTS: Dict[str, float] = {
    "method_type": 0.35,  # Most important for independence - different methods have different biases
    "data_source": 0.25,  # Data provenance affects reliability
    "temporal": 0.20,  # Time-based approaches have different selection biases
    "causal_role": 0.20,  # Different inference patterns
}


class DomainClassification(BaseModel):
    """Classification of evidence along 4 domain dimensions.

    Used to determine methodological independence for convergence analysis.
    Domains are defined by ERROR MODES, not topics - this is the key insight.

    Two sources from different topics but same method are NOT independent.
    Two sources from same topic but different methods ARE independent.
    """

    evidence_id: str = Field(description="ID of evidence being classified")
    claim_id: str = Field(description="ID of claim this evidence supports")

    # 4 dimension classifications (general, not domain-specific)
    method_type: MethodType = Field(description="How was this knowledge generated?")
    data_source: DataSourceType = Field(description="Where did the data come from?")
    temporal: TemporalApproach = Field(description="How does time factor in?")
    causal_role: CausalRole = Field(description="What kind of claim is being made?")

    # Confidence
    classification_confidence: float = Field(
        ge=0.0, le=1.0, description="How sure are we?"
    )
    classification_method: str = Field(
        default="automatic", description="'automatic', 'human', or 'hybrid'"
    )
    classification_notes: str = Field(
        default="", description="Explanation of classification"
    )

    @property
    def dimension_vector(self) -> Dict[str, str]:
        """Return classification as dict for distance calculation."""
        return {
            "method_type": self.method_type.value,
            "data_source": self.data_source.value,
            "temporal": self.temporal.value,
            "causal_role": self.causal_role.value,
        }

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to serializable dict for storage."""
        return {
            "evidence_id": self.evidence_id,
            "claim_id": self.claim_id,
            "method_type": self.method_type.value,
            "data_source": self.data_source.value,
            "temporal": self.temporal.value,
            "causal_role": self.causal_role.value,
            "classification_confidence": self.classification_confidence,
            "classification_method": self.classification_method,
            "classification_notes": self.classification_notes,
        }

    @classmethod
    def from_metadata(cls, meta: Dict[str, Any]) -> "DomainClassification":
        """Reconstruct from serialized dict."""
        return cls(
            evidence_id=meta.get("evidence_id", ""),
            claim_id=meta.get("claim_id", ""),
            method_type=MethodType(meta.get("method_type", "observational")),
            data_source=DataSourceType(meta.get("data_source", "primary")),
            temporal=TemporalApproach(meta.get("temporal", "cross_sectional")),
            causal_role=CausalRole(meta.get("causal_role", "phenomenological")),
            classification_confidence=meta.get("classification_confidence", 0.5),
            classification_method=meta.get("classification_method", "automatic"),
            classification_notes=meta.get("classification_notes", ""),
        )


class DomainCluster(BaseModel):
    """A cluster of evidence from the same domain.

    Evidence items with domain distance < threshold are grouped together.
    """

    cluster_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    evidence_ids: List[str] = Field(
        default_factory=list, description="Evidence in this cluster"
    )
    representative_classification: Optional[DomainClassification] = Field(
        None, description="Representative classification for this cluster"
    )
    cluster_size: int = Field(default=0, description="Number of evidence items")
    average_evidence_quality: float = Field(
        default=0.0, description="Average quality of evidence in cluster"
    )
    cluster_label: str = Field(
        default="", description="Human-readable label for this domain"
    )

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to serializable dict for storage."""
        return {
            "cluster_id": self.cluster_id,
            "evidence_ids": self.evidence_ids,
            "representative_classification": (
                self.representative_classification.to_metadata()
                if self.representative_classification
                else None
            ),
            "cluster_size": self.cluster_size,
            "average_evidence_quality": self.average_evidence_quality,
            "cluster_label": self.cluster_label,
        }

    @classmethod
    def from_metadata(cls, meta: Dict[str, Any]) -> "DomainCluster":
        """Reconstruct from serialized dict."""
        rep_class = None
        if meta.get("representative_classification"):
            rep_class = DomainClassification.from_metadata(
                meta["representative_classification"]
            )

        return cls(
            cluster_id=meta.get("cluster_id", ""),
            evidence_ids=meta.get("evidence_ids", []),
            representative_classification=rep_class,
            cluster_size=meta.get("cluster_size", 0),
            average_evidence_quality=meta.get("average_evidence_quality", 0.0),
            cluster_label=meta.get("cluster_label", ""),
        )


class ConvergentEvidence(BaseModel):
    """Evidence from cross-domain convergence assessment.

    Provides methodological independence by detecting when evidence from
    epistemically independent domains (different error modes) converges
    on the same conclusion.

    Key insight: Convergence is meaningful when evidence that COULD disagree
    (different error modes) actually agrees.
    """

    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str = Field(description="The claim being assessed for convergence")
    objective_id: str = Field(description="Parent objective")

    # Classification results
    evidence_classifications: List[DomainClassification] = Field(
        default_factory=list, description="Classifications for each evidence item"
    )
    total_evidence_count: int = Field(
        default=0, description="Total evidence items analyzed"
    )

    # Clustering results
    domain_clusters: List[DomainCluster] = Field(
        default_factory=list, description="Evidence grouped by domain"
    )
    num_independent_domains: int = Field(
        default=0, description="Number of independent domain clusters"
    )

    # Distance metrics
    average_inter_domain_distance: float = Field(
        default=0.0, description="Average distance between domains"
    )
    min_inter_domain_distance: float = Field(
        default=0.0, description="Minimum distance between any two domains"
    )

    # Independence verification
    independence_checks: Dict[str, bool] = Field(
        default_factory=dict,
        description="Results of independence checks (citation, author, data, temporal)",
    )
    independence_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Overall independence score"
    )

    # Convergence scoring
    convergence_detected: bool = Field(
        default=False, description="Was meaningful convergence detected?"
    )
    convergence_strength: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Strength of convergence"
    )
    convergence_justification: str = Field(
        default="", description="Why convergence is/isn't meaningful"
    )

    # Verdict
    verdict: str = Field(
        default="SINGLE_DOMAIN",
        description="'CONVERGENT', 'PARTIAL', 'SINGLE_DOMAIN', 'CONFLICTING'",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = Field(default="", description="Human-readable explanation")

    # Domain gap analysis
    missing_domains: List[str] = Field(
        default_factory=list, description="Suggested domains to expand coverage"
    )
    strongest_per_domain: Dict[str, str] = Field(
        default_factory=dict, description="Best evidence ID from each domain"
    )

    created_at: datetime = Field(default_factory=datetime.now)

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to DocumentStore metadata format."""
        return {
            "epistemic_type": "convergent_evidence",
            "evidence_id": self.evidence_id,
            "claim_id": self.claim_id,
            "objective_id": self.objective_id,
            # Full data for retrieval
            "evidence_classifications": [
                ec.to_metadata() for ec in self.evidence_classifications
            ],
            "total_evidence_count": self.total_evidence_count,
            "domain_clusters": [dc.to_metadata() for dc in self.domain_clusters],
            "num_independent_domains": self.num_independent_domains,
            "average_inter_domain_distance": self.average_inter_domain_distance,
            "min_inter_domain_distance": self.min_inter_domain_distance,
            "independence_checks": self.independence_checks,
            "independence_score": self.independence_score,
            "convergence_detected": self.convergence_detected,
            "convergence_strength": self.convergence_strength,
            "convergence_justification": self.convergence_justification,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "missing_domains": self.missing_domains,
            "strongest_per_domain": self.strongest_per_domain,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_metadata(cls, meta: Dict[str, Any]) -> "ConvergentEvidence":
        """Reconstruct from DocumentStore metadata."""
        from datetime import datetime

        # Reconstruct evidence classifications
        classifications = []
        for ec_meta in meta.get("evidence_classifications", []):
            classifications.append(DomainClassification.from_metadata(ec_meta))

        # Reconstruct domain clusters
        clusters = []
        for dc_meta in meta.get("domain_clusters", []):
            clusters.append(DomainCluster.from_metadata(dc_meta))

        return cls(
            evidence_id=meta.get("evidence_id", ""),
            claim_id=meta.get("claim_id", ""),
            objective_id=meta.get("objective_id", ""),
            evidence_classifications=classifications,
            total_evidence_count=meta.get("total_evidence_count", 0),
            domain_clusters=clusters,
            num_independent_domains=meta.get("num_independent_domains", 0),
            average_inter_domain_distance=meta.get(
                "average_inter_domain_distance", 0.0
            ),
            min_inter_domain_distance=meta.get("min_inter_domain_distance", 0.0),
            independence_checks=meta.get("independence_checks", {}),
            independence_score=meta.get("independence_score", 0.0),
            convergence_detected=meta.get("convergence_detected", False),
            convergence_strength=meta.get("convergence_strength", 0.0),
            convergence_justification=meta.get("convergence_justification", ""),
            verdict=meta.get("verdict", "SINGLE_DOMAIN"),
            confidence=meta.get("confidence", 0.0),
            explanation=meta.get("explanation", ""),
            missing_domains=meta.get("missing_domains", []),
            strongest_per_domain=meta.get("strongest_per_domain", {}),
            created_at=datetime.fromisoformat(
                meta.get("created_at", datetime.now().isoformat())
            ),
        )


class DeductiveEvidence(BaseModel):
    """Evidence from deductive validation of a claim.

    Provides logical soundness verification parallel to inductive evidence gathering.
    Checks whether a claim can be derived from first principles, is internally
    consistent, and is physically plausible.

    Key insight: A claim with strong empirical evidence could still be logically
    incoherent or physically impossible. This catches failures that evidence alone
    cannot reveal.
    """

    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str = Field(description="The claim being validated deductively")
    objective_id: str = Field(description="Parent objective")

    # Core deductive checks
    derived_from_first_principles: bool = Field(
        default=False, description="Can claim be derived from fundamental premises?"
    )
    is_internally_consistent: bool = Field(
        default=True, description="Does claim contradict itself?"
    )
    is_physically_plausible: bool = Field(
        default=True,
        description="Does claim violate conservation laws, causality, etc.?",
    )

    # Overall verdict
    deductive_soundness: str = Field(
        default="unknown", description="'sound', 'questionable', 'unsound'"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence in the assessment"
    )

    # Issues found during validation
    issues_found: List[str] = Field(
        default_factory=list, description="List of deductive issues found"
    )
    issue_types: List[str] = Field(
        default_factory=list,
        description="Types of issues: logical_inconsistency, physical_implausibility, missing_premise, assumption",
    )

    # Recommendation
    recommendation: str = Field(
        default="hold", description="'promote', 'hold', 'demote'"
    )
    explanation: str = Field(
        default="", description="Human-readable explanation of the assessment"
    )

    created_at: datetime = Field(default_factory=datetime.now)

    def passes_validation(self) -> bool:
        """Check if claim passes deductive validation (can proceed to ROBUST)."""
        return self.deductive_soundness == "sound" or (
            self.deductive_soundness == "questionable"
            and self.is_internally_consistent
            and self.is_physically_plausible
        )

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to DocumentStore metadata format."""
        return {
            "epistemic_type": "deductive_evidence",
            "evidence_id": self.evidence_id,
            "claim_id": self.claim_id,
            "objective_id": self.objective_id,
            "derived_from_first_principles": self.derived_from_first_principles,
            "is_internally_consistent": self.is_internally_consistent,
            "is_physically_plausible": self.is_physically_plausible,
            "deductive_soundness": self.deductive_soundness,
            "confidence": self.confidence,
            "issues_found": self.issues_found,
            "issue_types": self.issue_types,
            "recommendation": self.recommendation,
            "explanation": self.explanation,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_metadata(cls, meta: Dict[str, Any]) -> "DeductiveEvidence":
        """Reconstruct from DocumentStore metadata."""
        from datetime import datetime

        return cls(
            evidence_id=meta.get("evidence_id", ""),
            claim_id=meta.get("claim_id", ""),
            objective_id=meta.get("objective_id", ""),
            derived_from_first_principles=meta.get(
                "derived_from_first_principles", False
            ),
            is_internally_consistent=meta.get("is_internally_consistent", True),
            is_physically_plausible=meta.get("is_physically_plausible", True),
            deductive_soundness=meta.get("deductive_soundness", "unknown"),
            confidence=meta.get("confidence", 0.0),
            issues_found=meta.get("issues_found", []),
            issue_types=meta.get("issue_types", []),
            recommendation=meta.get("recommendation", "hold"),
            explanation=meta.get("explanation", ""),
            created_at=datetime.fromisoformat(
                meta.get("created_at", datetime.now().isoformat())
            ),
        )


# --- Temporal Prediction Registry ---


class PredictionType(str, Enum):
    """Types of testable predictions.

    Predictions vary in specificity and testability.
    """

    QUANTITATIVE = (
        "quantitative"  # Specific numeric prediction (e.g., "20-30% improvement")
    )
    QUALITATIVE = "qualitative"  # Directional prediction (e.g., "will improve")
    CONDITIONAL = "conditional"  # If-then prediction (e.g., "if X then Y")
    TEMPORAL = "temporal"  # Time-bound prediction (e.g., "within 6 months")
    BINARY = "binary"  # Yes/no prediction (e.g., "will succeed")


class PredictionStatus(str, Enum):
    """Status of a prediction in the registry.

    Tracks lifecycle from creation to resolution.
    """

    PENDING = "pending"  # Prediction made, awaiting outcome
    CONFIRMED = "confirmed"  # Outcome matched prediction
    PARTIALLY_CONFIRMED = "partially_confirmed"  # Outcome partially matched
    REFUTED = "refuted"  # Outcome contradicted prediction
    EXPIRED = "expired"  # Time window passed without resolution
    WITHDRAWN = "withdrawn"  # Prediction withdrawn (claim modified)


class Prediction(BaseModel):
    """A testable prediction derived from a claim.

    Predictions provide TEMPORAL INDEPENDENCE:
    - Made at time T1 based on claim
    - Resolved at time T2 against actual outcomes
    - Confirmation at T2 is strong evidence because prediction at T1
      could not have been influenced by the outcome

    Key insight: Predictions should be RISKY - specific enough that
    they could meaningfully fail. Vague predictions that are hard to
    refute provide weak evidence when confirmed.
    """

    prediction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str = Field(description="The claim this prediction tests")
    objective_id: str = Field(description="Parent objective")

    # Prediction content
    prediction_statement: str = Field(description="The specific testable prediction")
    prediction_type: PredictionType = Field(default=PredictionType.QUALITATIVE)
    specificity_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How specific/risky is this prediction? Higher = more testable",
    )

    # Testability criteria
    success_criteria: str = Field(description="What would confirm this prediction?")
    failure_criteria: str = Field(description="What would refute this prediction?")
    measurement_method: str = Field(
        default="", description="How to measure the outcome"
    )

    # Time bounds
    prediction_made_at: datetime = Field(default_factory=datetime.now)
    resolution_deadline: Optional[datetime] = Field(
        default=None, description="When prediction should be resolved by"
    )
    time_horizon: str = Field(
        default="unspecified", description="e.g., 'immediate', '1 week', '1 year'"
    )

    # Context
    assumptions: List[str] = Field(
        default_factory=list, description="Conditions assumed for prediction"
    )
    confounders: List[str] = Field(
        default_factory=list, description="Factors that could affect outcome"
    )

    # Status
    status: PredictionStatus = Field(default=PredictionStatus.PENDING)

    def is_expired(self) -> bool:
        """Check if prediction has passed its deadline."""
        if self.resolution_deadline is None:
            return False
        return datetime.now() > self.resolution_deadline

    def days_until_deadline(self) -> Optional[int]:
        """Days until resolution deadline, or None if no deadline."""
        if self.resolution_deadline is None:
            return None
        delta = self.resolution_deadline - datetime.now()
        return max(0, delta.days)

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to DocumentStore metadata format."""
        return {
            "epistemic_type": "prediction",
            "prediction_id": self.prediction_id,
            "claim_id": self.claim_id,
            "objective_id": self.objective_id,
            "statement": self.prediction_statement,
            "prediction_type": self.prediction_type.value,
            "specificity_score": self.specificity_score,
            "success_criteria": self.success_criteria,
            "failure_criteria": self.failure_criteria,
            "time_horizon": self.time_horizon,
            "status": self.status.value,
            "prediction_made_at": self.prediction_made_at.isoformat(),
            "resolution_deadline": self.resolution_deadline.isoformat()
            if self.resolution_deadline
            else None,
        }


class PredictionOutcome(BaseModel):
    """The observed outcome for resolving a prediction.

    Records what actually happened when the prediction was tested.
    """

    outcome_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prediction_id: str = Field(description="The prediction being resolved")

    # Outcome data
    observed_outcome: str = Field(description="What was actually observed")
    outcome_source: str = Field(description="Source/reference for the outcome data")
    outcome_date: datetime = Field(default_factory=datetime.now)

    # Match assessment
    matches_prediction: bool = Field(description="Did outcome match prediction?")
    match_degree: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How closely did outcome match? 1.0 = exact match",
    )
    match_explanation: str = Field(
        default="", description="Why outcome does/doesn't match"
    )

    # Confounders
    confounders_present: List[str] = Field(
        default_factory=list, description="Any confounding factors observed"
    )
    assumptions_held: bool = Field(
        default=True, description="Did the prediction's assumptions hold?"
    )


class TemporalEvidence(BaseModel):
    """Evidence from temporal prediction verification.

    Provides TEMPORAL INDEPENDENCE by tracking predictions made before
    outcomes were known and their subsequent resolution.

    Key insight: A prediction made at T1 that is confirmed at T2 provides
    strong evidence because the prediction could not have been retrofitted
    to match the outcome.
    """

    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim_id: str = Field(description="The claim being verified")
    objective_id: str = Field(description="Parent objective")

    # Prediction summary
    predictions: List[Prediction] = Field(
        default_factory=list, description="All predictions for this claim"
    )
    total_predictions: int = Field(default=0, description="Total predictions generated")
    resolved_predictions: int = Field(
        default=0, description="Predictions that have been resolved"
    )

    # Resolution summary
    confirmed_count: int = Field(default=0)
    partially_confirmed_count: int = Field(default=0)
    refuted_count: int = Field(default=0)
    pending_count: int = Field(default=0)
    expired_count: int = Field(default=0)

    # Scoring
    confirmation_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of resolved predictions confirmed",
    )
    weighted_confirmation_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confirmation weighted by prediction specificity",
    )
    temporal_independence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall temporal independence strength",
    )

    # Verdict
    verdict: str = Field(
        default="PENDING",
        description="'STRONGLY_CONFIRMED', 'CONFIRMED', 'MIXED', 'REFUTED', 'PENDING', 'INSUFFICIENT'",
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = Field(default="", description="Human-readable explanation")

    # Analysis
    strongest_confirmation: Optional[str] = Field(
        default=None, description="Most convincing confirmed prediction"
    )
    strongest_refutation: Optional[str] = Field(
        default=None, description="Most convincing refuted prediction"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Suggestions for improving predictions"
    )

    created_at: datetime = Field(default_factory=datetime.now)

    def compute_confirmation_rate(self) -> float:
        """Compute confirmation rate from resolved predictions."""
        resolved = (
            self.confirmed_count + self.partially_confirmed_count + self.refuted_count
        )
        if resolved == 0:
            return 0.0
        # Partial confirmations count as 0.5
        confirmed_equivalent = (
            self.confirmed_count + 0.5 * self.partially_confirmed_count
        )
        return confirmed_equivalent / resolved

    def compute_temporal_score(self) -> float:
        """Compute overall temporal independence score.

        Weights:
        - Confirmation rate (40%)
        - Prediction specificity (30%)
        - Number of predictions (20%)
        - Time span covered (10%)
        """
        if self.total_predictions == 0:
            return 0.0

        # Base confirmation score
        confirmation_score = self.compute_confirmation_rate() * 0.4

        # Specificity bonus (average specificity of predictions)
        avg_specificity = (
            sum(p.specificity_score for p in self.predictions) / len(self.predictions)
            if self.predictions
            else 0.5
        )
        specificity_score = avg_specificity * 0.3

        # Coverage bonus (diminishing returns after 3 predictions)
        coverage_score = min(1.0, self.total_predictions / 3) * 0.2

        # Resolution bonus (predictions that have been tested)
        resolution_rate = (
            self.resolved_predictions / self.total_predictions
            if self.total_predictions > 0
            else 0
        )
        resolution_score = resolution_rate * 0.1

        return (
            confirmation_score + specificity_score + coverage_score + resolution_score
        )

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to DocumentStore metadata format."""
        return {
            "epistemic_type": "temporal_evidence",
            "evidence_id": self.evidence_id,
            "claim_id": self.claim_id,
            "objective_id": self.objective_id,
            "total_predictions": self.total_predictions,
            "resolved_predictions": self.resolved_predictions,
            "confirmed_count": self.confirmed_count,
            "refuted_count": self.refuted_count,
            "confirmation_rate": self.confirmation_rate,
            "temporal_independence_score": self.temporal_independence_score,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }


class PlanArguments(BaseModel):
    """Structured arguments for the deterministic epistemic workflow.

    The PLAN_TASK agent outputs this structure to configure the workflow.
    The workflow sequence is FIXED, but these arguments control:
    - Which evidence providers to use
    - Which verification methods to apply
    - What output to generate

    This design separates "what capabilities to use" (LLM decides)
    from "how to orchestrate them" (deterministic workflow).
    """

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    objective_id: str = Field(description="The objective this plan is for")

    # Evidence gathering strategy
    evidence_strategy: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {"provider": "web_search", "config": {"depth": "standard"}}
        ],
        description="List of evidence providers to use with their configs",
    )

    # Verification strategy
    verification_strategy: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {"method": "scrutiny", "config": {}},
            {"method": "argument_analysis", "config": {}},
        ],
        description="List of verification methods to apply to each claim",
    )

    # Output configuration
    output_strategy: Dict[str, Any] = Field(
        default_factory=lambda: {"artefact_type": "summary"},
        description="Configuration for output generation",
    )

    # Optional focus areas for evidence gathering
    focus_areas: List[str] = Field(
        default_factory=list,
        description="Specific aspects to focus on during evidence gathering",
    )

    # Reasoning from the planner
    planning_rationale: str = Field(
        default="",
        description="Brief explanation of why these capabilities were chosen",
    )

    created_at: datetime = Field(default_factory=datetime.now)

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to DocumentStore metadata format."""
        return {
            "epistemic_type": "plan_arguments",
            "plan_id": self.plan_id,
            "objective_id": self.objective_id,
            "evidence_providers": [e.get("provider") for e in self.evidence_strategy],
            "verification_methods": [
                v.get("method") for v in self.verification_strategy
            ],
            "artefact_type": self.output_strategy.get("artefact_type", "summary"),
            "created_at": self.created_at.isoformat(),
        }

    def get_evidence_providers(self) -> List[str]:
        """Get list of evidence provider names."""
        return [
            e.get("provider", "") for e in self.evidence_strategy if e.get("provider")
        ]

    def get_verification_methods(self) -> List[str]:
        """Get list of verification method names."""
        return [
            v.get("method", "") for v in self.verification_strategy if v.get("method")
        ]


# --- Workflow Primitives ---


class WorkItem(BaseModel):
    """Schedulable unit of epistemic work.

    WorkItems are the scheduling backbone of the orchestrator.
    They define what operation to perform and track dependencies.

    The orchestrator executes WorkItems when:
    - status == QUEUED
    - all dependencies are DONE
    """

    workitem_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    objective_id: str
    operation_type: WorkItemType
    description: str = Field(default="", description="Human-readable description")
    inputs: Dict[str, Any] = Field(
        default_factory=dict, description="References to object IDs or data"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="WorkItem IDs that must complete first"
    )
    acceptance_criteria: Dict[str, Any] = Field(
        default_factory=dict, description="Structured criteria for success"
    )
    priority: int = Field(
        default=5, ge=1, le=100, description="1=highest priority, higher=lower priority"
    )
    status: WorkItemStatus = Field(default=WorkItemStatus.QUEUED)
    retry_count: int = Field(
        default=0, description="Number of times this workitem has been retried"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(default="system")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    output_refs: List[str] = Field(
        default_factory=list, description="IDs of objects created by this WorkItem"
    )

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to DocumentStore metadata format."""
        return {
            "epistemic_type": "workitem",
            "workitem_id": self.workitem_id,
            "objective_id": self.objective_id,
            "operation_type": self.operation_type.value,
            "description": self.description,  # CRITICAL: Needed for task-specific queries
            "inputs": self.inputs,
            "workitem_status": self.status.value,
            "dependencies": self.dependencies,
            "priority": self.priority,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat(),
        }


class EpistemicEvent(BaseModel):
    """Append-only audit log entry.

    All state changes in the epistemic system are logged as events.
    This enables:
    - Full audit trail
    - Resumption after interruption
    - Understanding why decisions were made
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: str = Field(
        description="workitem_started, claim_promoted, gate_failed, etc."
    )
    actor: str = Field(description="Executor name or human")
    target_id: Optional[str] = Field(None, description="Object ID affected")
    target_type: Optional[str] = Field(
        None, description="evidence, claim, workitem, etc."
    )
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Event-specific details"
    )

    def to_metadata(self) -> Dict[str, Any]:
        """Convert to DocumentStore metadata format."""
        return {
            "epistemic_type": "event",
            "event_id": self.event_id,
            "event_type": self.event_type,
            "actor": self.actor,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "timestamp": self.timestamp.isoformat(),
        }
