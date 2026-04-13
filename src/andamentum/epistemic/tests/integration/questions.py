"""Integration test questions — one per epistemological tradition.

Each question is chosen to reliably trigger a specific philosophical pathway
in the epistemic system. The questions are real research questions with known
evidence landscapes, selected so the system's behavior is predictable.
"""

from dataclasses import dataclass


@dataclass
class PathwayTest:
    """A single integration test targeting one philosophical tradition."""

    tradition: str
    question: str
    rationale: str
    expected_operations: list[str]
    max_iterations: int = 50
    question_type_hint: str | None = None  # Expected classification


PATHWAY_TESTS: list[PathwayTest] = [
    # ── Doyle: Truth Maintenance System ─────────────────────────────────
    # Needs: adversarial search finds strong refutation → TMS demotes claim
    # Why this question works: homeopathy has overwhelming negative evidence
    # from Cochrane reviews and systematic analyses. Claims will be proposed
    # from initial evidence, then adversarial search will find devastating
    # counterevidence (balance << 0.3), triggering TMS demotion.
    PathwayTest(
        tradition="doyle",
        question="Does homeopathy have therapeutic effects beyond placebo?",
        rationale=(
            "Adversarial search will find Cochrane reviews and meta-analyses "
            "showing no effect beyond placebo. Claims initially proposed from "
            "positive-leaning sources will be refuted, triggering TMS demotion."
        ),
        expected_operations=["revalidate_claim"],
        question_type_hint="verificatory",
    ),

    # ── Peirce: Inquiry Cycling ─────────────────────────────────────────
    # Needs: scrutiny returns "needs_resolution" → investigation → re-scrutiny
    # Why this question works: the evidence base for this specific claim is
    # thin and indirect. Initial OpenAlex results will be tangentially related
    # (general gut-brain papers, not specifically this probiotic strain for
    # anxiety). Scrutiny should flag evidence gaps → investigation cycle.
    PathwayTest(
        tradition="peirce",
        question="Does Lactobacillus rhamnosus supplementation reduce anxiety symptoms in healthy adults?",
        rationale=(
            "Narrow claim about a specific probiotic strain. Initial evidence "
            "will be general gut-brain axis papers, not strain-specific RCTs. "
            "Scrutiny should flag the evidence gap, triggering investigation."
        ),
        expected_operations=["investigate_claim"],
        question_type_hint="verificatory",
    ),

    # ── Lipton: Contrastive Evaluation ──────────────────────────────────
    # Needs: explanatory question type → contrastive_evaluation fires
    # Why this question works: classified as "explanatory", which activates
    # the contrastive track. Multiple competing explanations exist (reverse
    # causation, cardioprotective fat, collider bias, lean body mass loss).
    # The system should propose multiple claims and compare them.
    PathwayTest(
        tradition="lipton",
        question="What explains the obesity paradox in heart failure — why do overweight patients have better survival?",
        rationale=(
            "Explanatory question type activates contrastive evaluation. "
            "Multiple competing mechanistic explanations exist in the literature. "
            "System should propose competing claims and evaluate which better "
            "explains the available evidence."
        ),
        expected_operations=["contrastive_evaluation"],
        question_type_hint="explanatory",
    ),

    # ── Kahneman: Independence of Judgment ──────────────────────────────
    # Needs: multiple evidence items → per-evidence scrutiny produces
    # diverse issues from different evidence items independently.
    # Why this question works: broad topic with diverse evidence quality.
    # OpenAlex will return high-quality RCTs alongside weaker observational
    # studies and reviews. Per-evidence scrutiny should find different
    # issues in different items.
    PathwayTest(
        tradition="kahneman",
        question="Is intermittent fasting more effective than continuous caloric restriction for long-term weight management?",
        rationale=(
            "Comparative question with diverse evidence base. High-quality "
            "RCTs coexist with weaker observational studies. Per-evidence "
            "scrutiny should independently identify different issues in each "
            "evidence item (methodology, scope, quality)."
        ),
        expected_operations=["scrutinise_claim"],
        question_type_hint="comparative",
    ),

    # ── Tetlock: Predictions and Falsification ──────────────────────────
    # Needs: claim reaches ROBUST → predictions generated
    # Why this question works: strong, well-established evidence base
    # across multiple independent domains (epidemiology, mechanistic,
    # clinical). If any question can push a claim to ROBUST, it's one
    # with deep multi-domain evidence.
    # NOTE: reaching ROBUST is the hardest stage — this test may not
    # trigger in every run. Consider it aspirational.
    PathwayTest(
        tradition="tetlock",
        question="Does regular aerobic exercise reduce the incidence of type 2 diabetes in adults with prediabetes?",
        rationale=(
            "Very strong evidence base from multiple RCTs (DPP study), "
            "epidemiological cohorts, and mechanistic studies. Best chance "
            "of reaching ROBUST stage where prediction generation fires. "
            "If it does, predictions should include falsification criteria."
        ),
        expected_operations=["generate_prediction"],
        question_type_hint="verificatory",
        max_iterations=60,  # More room to reach ROBUST
    ),

    # ── AGM: Minimal Change on Demotion ─────────────────────────────────
    # Verified implicitly by the Doyle test — when TMS demotes a claim,
    # evidence links must be preserved and verification flags reset
    # consistently. No separate question needed; we check the database
    # state after the Doyle test.
]
