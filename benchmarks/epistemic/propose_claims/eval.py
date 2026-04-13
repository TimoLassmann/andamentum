"""Eval: epistemic_propose_claims agent.

Tests claim generation on cases requiring genuine synthesis judgment:
conflicting evidence, evidence that doesn't answer the question, claims
that must be appropriately scoped, and multi-dimensional topics.

NOTE: The agent prompt uses "spaced repetition" as a worked example.
No cases here use that topic.

Run:
    cd packages/epistemic
    uv run python agent_evals/run_all.py --agent propose_claims
"""

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason, LLMJudge

from conftest import run_agent


# ── Custom Evaluators ────────────────────────────────────────────────────


class ProducesClaimsInRange(Evaluator[dict, object, dict]):
    """Check that the agent produces a reasonable number of claims."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        claims = getattr(ctx.output, "claims", [])
        expected = ctx.expected_output or {}
        min_claims = expected.get("min_claims", 1)  # type: ignore[union-attr]
        max_claims = expected.get("max_claims", 10)  # type: ignore[union-attr]

        if len(claims) < min_claims:
            return EvaluationReason(value=False, reason=f"Only {len(claims)} claims, expected ≥{min_claims}")
        if len(claims) > max_claims:
            return EvaluationReason(value=False, reason=f"{len(claims)} claims, expected ≤{max_claims}")
        return EvaluationReason(value=True, reason=f"{len(claims)} claims in expected range [{min_claims}, {max_claims}]")


class ClaimsHaveRequiredFields(Evaluator[dict, object, dict]):
    """Each claim must have statement, scope, and evidence_refs."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        claims = getattr(ctx.output, "claims", [])
        issues = []
        for i, claim in enumerate(claims):
            stmt = getattr(claim, "statement", "")
            scope = getattr(claim, "scope", "")
            refs = getattr(claim, "evidence_refs", "")
            if not stmt or len(stmt.strip()) < 10:
                issues.append(f"Claim {i}: empty or trivial statement")
            if not scope:
                issues.append(f"Claim {i}: missing scope")
            if not refs:
                issues.append(f"Claim {i}: missing evidence_refs")
        if issues:
            return EvaluationReason(value=False, reason="; ".join(issues))
        return EvaluationReason(value=True, reason=f"All {len(claims)} claims have required fields")


class ClaimsAnswerObjective(Evaluator[dict, object, dict]):
    """Claims should relate to the research question, not be tangential."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        claims = getattr(ctx.output, "claims", [])
        objective = ctx.inputs.get("objective", "").lower()
        key_words = [w for w in objective.split() if len(w) > 4]
        if not key_words or not claims:
            return EvaluationReason(value=True, reason="Skipped (no key words or no claims)")

        any_related = False
        for claim in claims:
            stmt = getattr(claim, "statement", "").lower()
            if any(kw in stmt for kw in key_words):
                any_related = True
                break
        if any_related:
            return EvaluationReason(value=True, reason="At least one claim references objective terms")
        return EvaluationReason(
            value=False,
            reason="No claims appear to reference the objective's key terms",
        )


# ── Task Function ────────────────────────────────────────────────────────


async def propose_claims_task(inputs: dict) -> object:
    """Run the propose_claims agent."""
    return await run_agent(
        "epistemic_propose_claims",
        objective=inputs["objective"],
        evidence_summaries=inputs["evidence_summaries"],
    )


# ── Dataset ──────────────────────────────────────────────────────────────

CASES = [
    Case(
        name="conflicting_evidence_requires_nuanced_claim",
        inputs={
            "objective": "Is intermittent fasting more effective than continuous caloric restriction for weight loss?",
            "evidence_summaries": [
                "[E001] Harvie et al. (International Journal of Obesity, 2011, N=107): 5:2 intermittent "
                "fasting produced similar weight loss to daily caloric restriction (-6.4 kg vs -5.6 kg, "
                "p=0.4) at 6 months. Intermittent fasting group showed greater insulin sensitivity improvement.",
                "[E002] Trepanowski et al. (JAMA Internal Medicine 2017, N=100 obese adults): Alternate-day "
                "fasting produced no significant difference in weight loss vs daily caloric restriction at "
                "6 or 12 months. Dropout rate was significantly higher in the fasting group (38% vs 29%).",
                "[E003] Wilkinson et al. (Cell Metabolism 2020, N=19): Time-restricted eating (10-hour window) "
                "reduced body weight, blood pressure, and cholesterol in metabolic syndrome patients. "
                "No control group — single arm study.",
            ],
        },
        expected_output={"min_claims": 1, "max_claims": 4},
        metadata={"tests": "Evidence says 'no significant difference'. A good claim must reflect this rather than claiming superiority of either approach. E003 is weak (N=19, no control) — should it be cited?"},
    ),
    Case(
        name="evidence_doesnt_answer_question",
        inputs={
            "objective": "Should healthy adults take a daily multivitamin to prevent chronic disease?",
            "evidence_summaries": [
                "[E001] COSMOS trial (Annals of Internal Medicine 2022, N=21,442): Daily multivitamin "
                "did not significantly reduce total cardiovascular events (HR 0.98, 95% CI 0.87-1.10) "
                "or cancer incidence (HR 0.97, 95% CI 0.86-1.09) over 3.6 years.",
                "[E002] US Preventive Services Task Force (JAMA 2022): Concluded 'evidence is insufficient "
                "to assess the balance of benefits and harms of multivitamin supplementation for the "
                "prevention of cardiovascular disease and cancer'. Grade: I (Insufficient).",
                "[E003] Jenkins et al. (JACC 2018): Meta-analysis found no benefit of multivitamins for "
                "cardiovascular disease, stroke, or all-cause mortality. Vitamin D alone showed a small "
                "trend toward cancer mortality reduction.",
            ],
        },
        expected_output={"min_claims": 1, "max_claims": 3},
        metadata={"tests": "Evidence consistently says 'no benefit found'. The correct claim is a negative finding. Agent should NOT invent positive claims when evidence doesn't support them."},
    ),
    Case(
        name="scope_limiting_is_critical",
        inputs={
            "objective": "Are electric vehicles better for the environment than internal combustion engine vehicles?",
            "evidence_summaries": [
                "[E001] IEA Global EV Outlook 2023: Over their lifetime, EVs produce 50-70% fewer "
                "greenhouse gas emissions than ICE vehicles in most electricity grids. In grids "
                "dominated by coal (e.g., parts of India, Poland), the benefit drops to 20-30%.",
                "[E002] Dai et al. (Nature Sustainability 2019): Lithium and cobalt mining for batteries "
                "causes significant local water contamination and ecosystem disruption. Cobalt mining "
                "in the DRC involves documented child labour and artisanal mining hazards.",
                "[E003] Circular Energy Storage (2022): Currently only 5% of lithium-ion batteries are "
                "recycled globally. Battery waste is projected to reach 8 million tonnes annually by 2040.",
                "[E004] Hoekstra (Nature Sustainability 2019): Meta-analysis of 11 LCA studies found "
                "EVs produce 66-69% lower lifecycle GHG emissions on average, but results vary by "
                "factor of 3x depending on methodological assumptions about grid mix and battery lifespan.",
            ],
        },
        expected_output={"min_claims": 2, "max_claims": 5},
        metadata={"tests": "Multi-dimensional question where 'better' spans GHG, mining impacts, recycling, grid dependency. Claims must be appropriately scoped — not a blanket 'yes' or 'no'."},
    ),
    Case(
        name="strong_evidence_narrow_question",
        inputs={
            "objective": "What is the heritability of human height?",
            "evidence_summaries": [
                "[E001] Silventoinen et al. (Twin Research and Human Genetics 2003): Meta-analysis of "
                "twin studies across 45 populations estimated heritability of height at 0.80 (95% CI "
                "0.75-0.85) in developed countries.",
                "[E002] Visscher et al. (American Journal of Human Genetics 2008): GWAS identified "
                "~700 common variants explaining approximately 20% of height variance. The gap between "
                "twin-based heritability (0.80) and GWAS-explained variance (0.20) is known as "
                "'missing heritability'.",
                "[E003] Yengo et al. (Nature 2022, N=5.4 million): Largest GWAS to date identified "
                "12,111 SNPs explaining 40% of height variance in European-ancestry populations. "
                "Heritability estimates varied substantially by ancestry: 0.69-0.80 across populations.",
            ],
        },
        expected_output={"min_claims": 1, "max_claims": 3},
        metadata={"tests": "Clear answer (~0.80) but with the interesting 'missing heritability' problem. Agent should formulate a direct answer and may note the GWAS gap."},
    ),
    Case(
        name="mixed_quality_evidence_same_direction",
        inputs={
            "objective": "Does sleep deprivation impair decision-making?",
            "evidence_summaries": [
                "[E001] Killgore et al. (Sleep 2006): 21 hours of sustained wakefulness impaired moral "
                "reasoning and risk assessment in military personnel (N=26, within-subject design).",
                "[E002] Harrison & Horne (Journal of Experimental Psychology: Applied, 2000): Sleep-deprived "
                "participants made riskier choices on the Iowa Gambling Task (N=12, crossover design).",
                "[E003] Reddit user u/neurosci_nerd: 'As a PhD student I can confirm that all-nighters "
                "make me terrible at making decisions. My advisor agrees based on his 30 years of experience.'",
                "[E004] Lim & Dinges (Annals of NY Academy of Sciences 2008): Comprehensive review of "
                "70+ studies found consistent deficits in attention, working memory, and executive "
                "function following sleep deprivation, with effect sizes of d=0.5-1.5.",
            ],
        },
        expected_output={"min_claims": 1, "max_claims": 3},
        metadata={"tests": "Three credible sources + one Reddit anecdote all pointing same direction. Should the agent cite E003? Good claims should reference E001/E002/E004 and ignore or deprioritise E003."},
    ),
]


def build_dataset() -> Dataset:
    return Dataset(
        name="propose_claims",
        cases=CASES,
        evaluators=[
            ProducesClaimsInRange(),
            ClaimsHaveRequiredFields(),
            ClaimsAnswerObjective(),
            LLMJudge(
                rubric=(
                    "The agent should propose claims that genuinely answer the research question "
                    "with appropriate scope. When evidence conflicts, claims should reflect the "
                    "nuance (e.g., 'no significant difference' rather than picking a winner). "
                    "When evidence consistently shows null results, the claim should be a negative "
                    "finding — not manufactured positivity. When evidence quality varies, claims "
                    "should preferentially cite stronger sources. Claims should not overreach "
                    "beyond what the evidence supports."
                ),
                include_input=True,
            ),
        ],
    )


async def run_eval(max_concurrency: int = 3):
    ds = build_dataset()
    return await ds.evaluate(propose_claims_task, max_concurrency=max_concurrency)
