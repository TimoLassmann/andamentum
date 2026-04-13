"""Output validators for epistemic executor outputs.

Validates output schemas for each WorkItemType operation:
- Clarification, conceptual analysis
- Evidence collection, extraction
- Claims, scrutiny, promotion
- Computational verification, adversarial search
- Convergence, predictions
- Snapshots, artefacts, decisions

Architecture: Layer 1 (framework-agnostic, no model calls)
"""

from typing import Dict, Any, Callable

from ..primitives import WorkItemType, ClaimStage
from .types import ValidationResult


class OutputValidators:
    """Validates executor outputs against constitution rules.

    Each operation type has a dedicated validator that checks:
    - HARD requirements (errors that block operation)
    - SOFT requirements (warnings that allow operation but log issues)
    """

    def __init__(self) -> None:
        """Initialize output validators registry."""
        self._validators: Dict[
            WorkItemType, Callable[[Dict[str, Any]], ValidationResult]
        ] = {
            # Pre-planning validators
            WorkItemType.CLARIFY_QUESTION: self._validate_clarification_output,
            WorkItemType.CONCEPTUAL_ANALYSIS: self._validate_conceptual_analysis_output,
            # Core workflow validators
            WorkItemType.PLAN_TASK: self._validate_plan_output,
            WorkItemType.COLLECT_EVIDENCE: self._validate_evidence_collection_output,
            WorkItemType.EXTRACT_EVIDENCE: self._validate_evidence_output,
            WorkItemType.PROPOSE_CLAIMS: self._validate_claims_output,
            WorkItemType.WORLD_KNOWLEDGE_CLAIMS: self._validate_world_knowledge_claims_output,
            WorkItemType.SCRUTINISE_CLAIM: self._validate_scrutiny_output,
            WorkItemType.ANALYZE_ARGUMENT: self._validate_argument_analysis_output,
            WorkItemType.VERIFY_COMPUTATIONALLY: self._validate_computational_verification_output,
            WorkItemType.ADVERSARIAL_SEARCH: self._validate_adversarial_search_output,
            WorkItemType.VALIDATE_DEDUCTIVELY: self._validate_deductive_validation_output,
            WorkItemType.ASSESS_CONVERGENCE: self._validate_convergence_output,
            WorkItemType.GENERATE_PREDICTION: self._validate_prediction_generation_output,
            WorkItemType.RESOLVE_PREDICTION: self._validate_prediction_resolution_output,
            WorkItemType.PROMOTE_CLAIM: self._validate_promotion_output,
            WorkItemType.FREEZE_SNAPSHOT: self._validate_snapshot_output,
            WorkItemType.SYNTHESIZE_REPORT: self._validate_write_answer_output,
            WorkItemType.DECIDE: self._validate_decision_output,
        }

    def validate(
        self, operation_type: WorkItemType, output: Dict[str, Any]
    ) -> ValidationResult:
        """Validate executor output against constitution rules.

        Args:
            operation_type: Type of operation that produced this output
            output: The output dict from the executor

        Returns:
            ValidationResult with errors if invalid
        """
        result = ValidationResult(valid=True)

        # Get operation-specific validator
        validator = self._validators.get(operation_type)
        if validator:
            result = validator(output)
        else:
            result.add_warning(
                "UNKNOWN_OPERATION",
                f"No validator for operation type: {operation_type.value}",
            )

        return result

    def _validate_clarification_output(
        self, output: Dict[str, Any]
    ) -> ValidationResult:
        """Validate CLARIFY_QUESTION output.

        Format (from epistemic_clarify_question.md):
        - ambiguity_level: str ("clear", "moderate", "high")
        - clarified_question: str
        - key_terms: list[str]
        - reasoning: str
        """
        result = ValidationResult(valid=True)

        # HARD: Must have clarified_question
        if not output.get("clarified_question"):
            result.add_error("CLARIF_001", "clarified_question is required")

        # HARD: Must have ambiguity_level with valid value
        ambiguity_level = output.get("ambiguity_level", "")
        valid_levels = {"clear", "moderate", "high"}
        if not ambiguity_level:
            result.add_error("CLARIF_002", "ambiguity_level is required")
        elif ambiguity_level not in valid_levels:
            result.add_error(
                "CLARIF_003",
                f"ambiguity_level must be one of {valid_levels}, got: {ambiguity_level}",
            )

        # SOFT: High ambiguity should have reasoning
        if ambiguity_level == "high" and not output.get("reasoning"):
            result.add_warning(
                "CLARIF_010",
                "High ambiguity questions should include reasoning for interpretation choice",
            )

        return result

    def _validate_conceptual_analysis_output(
        self, output: Dict[str, Any]
    ) -> ValidationResult:
        """Validate CONCEPTUAL_ANALYSIS output.

        Format (from epistemic_conceptual_analysis.md):
        - terms: list[str]
        - definitions: list[str] (parallel to terms)
        - assumptions: list[str]
        - context_summary: str
        """
        result = ValidationResult(valid=True)

        terms = output.get("terms", [])
        definitions = output.get("definitions", [])

        # HARD: Must have context_summary
        if not output.get("context_summary"):
            result.add_error("CONCEPT_001", "context_summary is required")

        # HARD: terms and definitions must be parallel lists
        if terms and definitions and len(terms) != len(definitions):
            result.add_error(
                "CONCEPT_002",
                f"terms and definitions must be parallel lists of same length (got {len(terms)} terms, {len(definitions)} definitions)",
            )

        # SOFT: Should have at least one term defined
        if not terms:
            result.add_warning(
                "CONCEPT_010",
                "No terms defined - consider identifying key terms for the investigation",
            )

        # SOFT: Should have assumptions surfaced
        if not output.get("assumptions"):
            result.add_warning(
                "CONCEPT_011",
                "No assumptions surfaced - most questions have embedded assumptions",
            )

        return result

    def _validate_argument_analysis_output(
        self, output: Dict[str, Any]
    ) -> ValidationResult:
        """Validate ANALYZE_ARGUMENT output.

        Format (from epistemic_analyze_argument.md):
        - premises: list[str]
        - conclusion: str
        - validity: str ("valid", "invalid", "indeterminate")
        - soundness: str ("sound", "unsound", "questionable")
        - fallacies: list[str]
        """
        result = ValidationResult(valid=True)

        # HARD: Must have conclusion
        if not output.get("conclusion"):
            result.add_error("ARGMT_001", "conclusion is required")

        # HARD: validity must be valid value
        validity = output.get("validity", "")
        valid_validity = {"valid", "invalid", "indeterminate"}
        if validity and validity not in valid_validity:
            result.add_error(
                "ARGMT_002",
                f"validity must be one of {valid_validity}, got: {validity}",
            )

        # HARD: soundness must be valid value
        soundness = output.get("soundness", "")
        valid_soundness = {"sound", "unsound", "questionable"}
        if soundness and soundness not in valid_soundness:
            result.add_error(
                "ARGMT_003",
                f"soundness must be one of {valid_soundness}, got: {soundness}",
            )

        # SOFT: Should have at least one premise
        if not output.get("premises"):
            result.add_warning(
                "ARGMT_010",
                "No premises identified - most arguments have explicit or implicit premises",
            )

        # SOFT: Consistency check - sound requires valid
        if soundness == "sound" and validity == "invalid":
            result.add_warning(
                "ARGMT_011",
                "soundness='sound' but validity='invalid' - sound arguments must be valid",
            )

        return result

    def _validate_computational_verification_output(
        self, output: Dict[str, Any]
    ) -> ValidationResult:
        """Validate VERIFY_COMPUTATIONALLY output.

        Format (from epistemic_verify_computationally.md):
        - claim_id: str - ID of the claim being verified
        - verification_code: str - Complete Python code that tests the claim
        - packages_required: list[str] - Python packages needed (e.g., numpy, scipy)
        - expected_behavior: str - What the test should output if the claim is true
        - test_description: str - Human-readable explanation of what the test does

        HARD validation (errors):
        - Must have claim_id
        - Must have verification_code

        SOFT validation (warnings):
        - Should have test_description for explanation
        - Code should follow expected template structure
        """
        result = ValidationResult(valid=True)

        # HARD: Must have claim_id
        if not output.get("claim_id"):
            result.add_error(
                "COMPVER_001", "claim_id is required for computational verification"
            )

        # HARD: Must have verification_code
        verification_code = output.get("verification_code", "")
        if not verification_code:
            result.add_error("COMPVER_002", "verification_code is required")

        # HARD: Code must be non-trivial (at least 50 chars to avoid placeholder code)
        if verification_code and len(verification_code.strip()) < 50:
            result.add_error(
                "COMPVER_003", "verification_code appears to be placeholder (too short)"
            )

        # SOFT: Should have test_description
        if not output.get("test_description"):
            result.add_warning(
                "COMPVER_010",
                "test_description is recommended for explaining what the test does",
            )

        # SOFT: Code should have required template structure
        if verification_code:
            # Check for run_verification function (from template)
            if "def run_verification" not in verification_code:
                result.add_warning(
                    "COMPVER_011",
                    "verification_code should define run_verification() function per template",
                )

            # Check for JSON output (required for structured results)
            if (
                "json.dumps" not in verification_code
                and "import json" not in verification_code
            ):
                result.add_warning(
                    "COMPVER_012",
                    "verification_code should output JSON for structured results",
                )

            # Check for determinism hints (seeds for randomness)
            has_random = (
                "random" in verification_code.lower()
                or "np.random" in verification_code
            )
            has_seed = (
                "seed(" in verification_code or "random_state" in verification_code
            )
            if has_random and not has_seed:
                result.add_warning(
                    "COMPVER_013",
                    "Code uses randomness but may not set seed - could cause non-reproducible results",
                )

        return result

    def _validate_plan_output(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate plan_task executor output.

        NEW Format (deterministic workflow arguments):
        - evidence_strategy: [{provider, config}, ...]
        - verification_strategy: [{method, config}, ...]
        - output_strategy: {artefact_type, ...}
        - focus_areas: [str, ...]
        - planning_rationale: str

        LEGACY Format (deprecated):
        - tasks: [{description, task_type, priority}, ...]
        """
        result = ValidationResult(valid=True)

        # Check for NEW format (deterministic workflow arguments)
        # Use 'in' check to detect new format even if lists are empty
        has_new_format = (
            "evidence_strategy" in output or "verification_strategy" in output
        )
        evidence_strategy = output.get("evidence_strategy", [])
        verification_strategy = output.get("verification_strategy", [])

        if has_new_format:
            # NEW FORMAT: Validate evidence strategy
            if not evidence_strategy:
                result.add_warning(
                    "PLAN_001", "No evidence strategy specified - will use defaults"
                )

            for i, task in enumerate(evidence_strategy):
                if not task.get("provider"):
                    result.add_warning(
                        "PLAN_011", f"Evidence task {i} missing provider field"
                    )

            # NEW FORMAT: Validate verification strategy
            if not verification_strategy:
                result.add_warning(
                    "PLAN_012", "No verification strategy specified - will use defaults"
                )

            for i, task in enumerate(verification_strategy):
                if not task.get("method"):
                    result.add_warning(
                        "PLAN_013", f"Verification task {i} missing method field"
                    )

            return result

        # LEGACY FORMAT: tasks list (deprecated but still supported)
        tasks = output.get("tasks", [])
        if not tasks:
            result.add_error(
                "PLAN_001",
                "Plan must specify evidence_strategy and verification_strategy (or legacy tasks)",
            )
            return result

        # Validate each task has required fields
        valid_types = {t.value for t in WorkItemType}
        for i, task in enumerate(tasks):
            if not task.get("description"):
                result.add_warning("PLAN_011", f"Task {i} missing description field")
            task_type = task.get("task_type", "")
            if task_type and task_type not in valid_types:
                result.add_warning(
                    "PLAN_012",
                    f"Task {i} has unknown task_type: {task_type}",
                    valid_types=list(valid_types),
                )

        return result

    def _validate_evidence_collection_output(
        self, output: Dict[str, Any]
    ) -> ValidationResult:
        """Validate collect_evidence executor output.

        Format (from epistemic_collect_evidence.md manifest):
        - sources: [{url, source_type, summary}, ...]
        - search_successful: bool
        - no_results_reason: str

        SOFT VALIDATION: No sources is a WARNING, not an error.
        The pipeline can continue and report "no evidence found" honestly.
        """
        result = ValidationResult(valid=True)

        sources = output.get("sources", [])

        # SOFT VALIDATION: No sources is a warning, not blocking error
        # The pipeline can continue and be honest about finding nothing
        if not sources:
            result.add_warning(
                "EVID_010",
                "Evidence collection found no sources - will continue with partial results",
            )
            return result

        # Validate each source has required fields
        for i, source in enumerate(sources):
            if not source.get("url"):
                result.add_warning("EVID_011", f"Source {i} missing url field")

        return result

    def _validate_evidence_output(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate extract_evidence executor output.

        SOFT VALIDATION: Missing content is a warning, missing source_ref is an error.
        """
        result = ValidationResult(valid=True)

        # Must have source_ref (structural requirement - ERROR)
        if not output.get("source_ref"):
            result.add_error("EVID_020", "Evidence must have source_ref")

        # Should have some content (soft - WARNING if empty)
        quotes = output.get("relevant_quotes", [])
        summary = output.get("summary", "")
        if not quotes and not summary:
            result.add_warning(
                "EVID_021",
                "Evidence extraction found no relevant content - continuing with empty evidence",
            )

        return result

    def _validate_claims_output(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate propose_claims executor output.

        SOFT VALIDATION: No claims is a warning (pipeline continues to synthesis).
        Supports both new format (claims list) and legacy format (statements/scopes arrays).
        """
        result = ValidationResult(valid=True)

        # Support both new format (claims list) and legacy format (statements list)
        claims_list = output.get("claims", [])
        statements = output.get("statements", [])
        scopes = output.get("scopes", [])

        # New format: claims is a list of objects with statement, scope, etc.
        if claims_list:
            # Extract statements from claims list for validation
            for i, claim in enumerate(claims_list):
                stmt = claim.get("statement", "") if isinstance(claim, dict) else ""
                if len(stmt.strip()) < 10:
                    result.add_warning(
                        "CLAIM_003",
                        f"Claim statement {i} seems too short",
                        statement=stmt,
                    )
            return result

        # Legacy format: parallel arrays
        # SOFT VALIDATION: No claims is a warning, not an error
        # The pipeline can still synthesize results based on evidence alone
        if not statements:
            result.add_warning(
                "CLAIM_001",
                "No claims proposed - synthesis will proceed with evidence only",
            )
            return result

        # Scopes should match statements (structural - ERROR)
        if scopes and len(scopes) != len(statements):
            result.add_error("CLAIM_002", "scopes must match statements length")

        # Claims should have substance (quality - WARNING)
        for i, stmt in enumerate(statements):
            if len(stmt.strip()) < 10:
                result.add_warning(
                    "CLAIM_003", f"Claim statement {i} seems too short", statement=stmt
                )

        return result

    def _validate_scrutiny_output(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate scrutinise_claim executor output."""
        result = ValidationResult(valid=True)

        # Must reference a claim
        if not output.get("claim_id"):
            result.add_error("SCRUT_001", "Scrutiny must reference claim_id")

        # Must have recommendation
        recommendation = output.get("recommendation", "")
        valid_recs = {"promote", "hold", "demote"}
        if recommendation not in valid_recs:
            result.add_error(
                "SCRUT_002",
                f"recommendation must be one of {valid_recs}, got: {recommendation}",
            )

        # passes_scrutiny should be consistent with recommendation
        passes = output.get("passes_scrutiny", False)
        if passes and recommendation == "demote":
            result.add_warning(
                "SCRUT_003", "passes_scrutiny=True but recommendation is 'demote'"
            )
        if not passes and recommendation == "promote":
            result.add_warning(
                "SCRUT_004", "passes_scrutiny=False but recommendation is 'promote'"
            )

        return result

    def _validate_promotion_output(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate promote_claim executor output."""
        result = ValidationResult(valid=True)

        if not output.get("claim_id"):
            result.add_error("PROMO_001", "Promotion must reference claim_id")

        proposed_stage = output.get("proposed_stage", "")
        valid_stages = {s.value for s in ClaimStage}
        if proposed_stage not in valid_stages:
            result.add_error(
                "PROMO_002", f"proposed_stage must be one of {valid_stages}"
            )

        if not output.get("justification"):
            result.add_error("PROMO_003", "Promotion must include justification")

        return result

    def _validate_snapshot_output(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate freeze_snapshot executor output.

        Field names match agent manifest (epistemic_freeze_snapshot.md):
        - include_indices: list[int] - indices of claims to include
        - exclude_indices: list[int] - indices of claims to exclude
        - minimum_stage: str
        - snapshot_rationale: str
        """
        result = ValidationResult(valid=True)

        # Check for include_indices (manifest field name, not llm_models.py name)
        include_indices = output.get("include_indices", [])
        if not include_indices:
            result.add_error(
                "SNAP_001",
                "Snapshot must include at least one claim (include_indices is empty)",
            )

        return result

    def _validate_write_answer_output(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate epistemic_write_answer output.

        Field names match agent manifest (epistemic_write_answer.md):
        - title: str
        - answer: str

        HARD: Must have non-empty title and answer (min 50 chars)
        SOFT: Warning for short answer (<200 chars)
        """
        result = ValidationResult(valid=True)

        if not output.get("title"):
            result.add_error("SYNTH_001", "Report must have a title")

        answer = output.get("answer", "")
        if not answer or len(answer) < 50:
            result.add_error(
                "SYNTH_002", f"Answer too short ({len(answer)} chars, minimum 50)"
            )
        elif len(answer) < 200:
            result.add_warning(
                "SYNTH_003",
                f"Answer is short ({len(answer)} chars) - consider expanding",
            )

        return result

    def _validate_decision_output(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate decide executor output.

        HARD validation (errors):
        - Must have statement
        - Must have justification
        - Must reference at least one claim

        SOFT validation (warnings):
        - Should have reversal_conditions when reversible
        """
        result = ValidationResult(valid=True)

        # HARD: Must have statement
        if not output.get("statement"):
            result.add_error("DECIDE_001", "Decision must have a statement")

        # HARD: Must have justification
        if not output.get("justification"):
            result.add_error("DECIDE_002", "Decision must have a justification")

        # HARD: Must reference at least one claim (agent outputs claim_indices, transformed to claim_ids later)
        claim_indices = output.get("claim_indices", [])
        if not claim_indices:
            result.add_error(
                "DECIDE_003",
                "Decision must reference at least one claim (via claim_indices)",
            )

        # SOFT: Should have reversal conditions when reversible
        reversible = output.get("reversible", True)
        reversal_conditions = output.get("reversal_conditions", "")
        if reversible and not reversal_conditions:
            result.add_warning(
                "DECIDE_010",
                "Decision marked reversible but no reversal_conditions provided - consider specifying when to reconsider",
            )

        return result

    def _validate_world_knowledge_claims_output(
        self, output: Dict[str, Any]
    ) -> ValidationResult:
        """Validate WORLD_KNOWLEDGE_CLAIMS output.

        Expected fields (from epistemic_world_knowledge_claims.md):
        - statements: list[str] - Claims generated from world knowledge
        - scopes: list[str] - Scope for each statement
        - reasoning: list[str] - Reasoning for each statement
        - confidence_notes: list[str] - Confidence notes for each statement
        - knowledge_type: str - 'synthesis', 'factual', 'theoretical'
        """
        result = ValidationResult(valid=True)

        statements = output.get("statements", [])
        if not statements:
            result.add_warning("WK_001", "No statements generated from world knowledge")
            return result

        # Check parallel lists have same length
        scopes = output.get("scopes", [])
        reasoning = output.get("reasoning", [])
        if len(scopes) != len(statements):
            result.add_warning(
                "WK_002",
                f"Mismatched scopes count ({len(scopes)}) vs statements ({len(statements)})",
            )
        if len(reasoning) != len(statements):
            result.add_warning(
                "WK_003",
                f"Mismatched reasoning count ({len(reasoning)}) vs statements ({len(statements)})",
            )

        return result

    def _validate_adversarial_search_output(
        self, output: Dict[str, Any]
    ) -> ValidationResult:
        """Validate ADVERSARIAL_SEARCH output.

        Expected fields (from epistemic_adversarial_search.md):
        - verdict: str - 'SUPPORTED', 'CONTESTED', 'CHALLENGED', 'REFUTED'
        - counterarguments: list[dict] - Found counterarguments
        - adversarial_balance: float - Balance score (0=against, 1=for)
        - recommendation: str - 'maintain', 'weaken', 'refute', 'modify'
        """
        result = ValidationResult(valid=True)

        verdict = output.get("verdict", "")
        if verdict not in ("SUPPORTED", "CONTESTED", "CHALLENGED", "REFUTED", ""):
            result.add_warning("ADV_001", f"Unknown verdict: {verdict}")

        balance = output.get("adversarial_balance", 0.5)
        if not 0 <= balance <= 1:
            result.add_warning(
                "ADV_002", f"adversarial_balance should be 0-1, got {balance}"
            )

        return result

    def _validate_deductive_validation_output(
        self, output: Dict[str, Any]
    ) -> ValidationResult:
        """Validate VALIDATE_DEDUCTIVELY output.

        Expected fields (from epistemic_deductive_validation.md):
        - claim_id: str - ID of the claim being validated
        - deductive_soundness: str - 'sound', 'questionable', 'unsound'
        - confidence_estimate: float - 0.0-1.0
        - passes_deductive_validation: bool
        - issues_found: list[str] - Issues discovered
        - issue_types: list[str] - Types of issues
        - recommendation: str - 'promote', 'hold', 'demote'
        """
        result = ValidationResult(valid=True)

        # HARD: Must have claim_id
        if not output.get("claim_id"):
            result.add_error(
                "DEDUCT_001", "claim_id is required for deductive validation"
            )

        # HARD: deductive_soundness must be valid value
        soundness = output.get("deductive_soundness", "")
        valid_soundness = {"sound", "questionable", "unsound"}
        if not soundness:
            result.add_error("DEDUCT_002", "deductive_soundness is required")
        elif soundness not in valid_soundness:
            result.add_error(
                "DEDUCT_003",
                f"deductive_soundness must be one of {valid_soundness}, got: {soundness}",
            )

        # HARD: recommendation must be valid value
        recommendation = output.get("recommendation", "")
        valid_recs = {"promote", "hold", "demote"}
        if not recommendation:
            result.add_error("DEDUCT_004", "recommendation is required")
        elif recommendation not in valid_recs:
            result.add_error(
                "DEDUCT_005",
                f"recommendation must be one of {valid_recs}, got: {recommendation}",
            )

        # SOFT: confidence_estimate should be 0-1
        confidence = output.get("confidence_estimate", 0.5)
        if not 0 <= confidence <= 1:
            result.add_warning(
                "DEDUCT_010", f"confidence_estimate should be 0-1, got {confidence}"
            )

        # SOFT: Consistency checks
        passes = output.get("passes_deductive_validation", False)
        if soundness == "unsound" and passes:
            result.add_warning(
                "DEDUCT_011",
                "deductive_soundness='unsound' but passes_deductive_validation=True",
            )
        if soundness == "sound" and not passes:
            result.add_warning(
                "DEDUCT_012",
                "deductive_soundness='sound' but passes_deductive_validation=False",
            )

        return result

    def _validate_convergence_output(self, output: Dict[str, Any]) -> ValidationResult:
        """Validate ASSESS_CONVERGENCE output.

        Expected fields (from epistemic_assess_convergence.md):
        - verdict: str - 'CONVERGENT', 'PARTIAL', 'SINGLE_DOMAIN', 'CONFLICTING'
        - convergence_strength: float - 0-1
        - num_independent_domains: int
        """
        result = ValidationResult(valid=True)

        verdict = output.get("verdict", "")
        if verdict not in ("CONVERGENT", "PARTIAL", "SINGLE_DOMAIN", "CONFLICTING", ""):
            result.add_warning("CONV_001", f"Unknown verdict: {verdict}")

        strength = output.get("convergence_strength", 0.0)
        if not 0 <= strength <= 1:
            result.add_warning(
                "CONV_002", f"convergence_strength should be 0-1, got {strength}"
            )

        return result

    def _validate_prediction_generation_output(
        self, output: Dict[str, Any]
    ) -> ValidationResult:
        """Validate GENERATE_PREDICTION output.

        Expected fields (from epistemic_generate_prediction.md):
        - predictions: list[dict] - Testable predictions
        - prediction_rationale: str
        """
        result = ValidationResult(valid=True)

        predictions = output.get("predictions", [])
        if not predictions:
            result.add_warning("PRED_001", "No predictions generated")

        # Each prediction should have statement, success_criteria, failure_criteria
        for i, pred in enumerate(predictions):
            if not pred.get("prediction_statement"):
                result.add_warning(
                    "PRED_002", f"Prediction {i} missing prediction_statement"
                )
            if not pred.get("success_criteria"):
                result.add_warning(
                    "PRED_003", f"Prediction {i} missing success_criteria"
                )

        return result

    def _validate_prediction_resolution_output(
        self, output: Dict[str, Any]
    ) -> ValidationResult:
        """Validate RESOLVE_PREDICTION output.

        Expected fields (from epistemic_resolve_prediction.md):
        - resolution_status: str - 'confirmed', 'refuted', 'partially_confirmed', 'inconclusive'
        - actual_outcome: str
        - confidence: float
        """
        result = ValidationResult(valid=True)

        status = output.get("resolution_status", "")
        if status not in (
            "confirmed",
            "refuted",
            "partially_confirmed",
            "inconclusive",
            "",
        ):
            result.add_warning("RESOL_001", f"Unknown resolution_status: {status}")

        confidence = output.get("confidence", 0.5)
        if not 0 <= confidence <= 1:
            result.add_warning(
                "RESOL_002", f"confidence should be 0-1, got {confidence}"
            )

        return result
