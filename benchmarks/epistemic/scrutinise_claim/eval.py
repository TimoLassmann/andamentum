"""Eval: epistemic_scrutinise_claim agent.

Tests genuine epistemic judgment — not instruction-following. Cases are designed
so that the correct verdict requires reasoning the prompt doesn't prescribe:
ambiguous evidence quality, subtle logical flaws, scope mismatches between
claim and evidence, and cases where caveats exist but the claim is still sound.

Run:
    cd packages/epistemic
    uv run python agent_evals/run_all.py --agent scrutinise_claim
"""

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    Evaluator,
    EvaluatorContext,
    EvaluationReason,
    LLMJudge,
)

from conftest import run_agent


# ── Custom Evaluators ────────────────────────────────────────────────────


class ScrutinyVerdictCorrect(Evaluator[dict, object, dict]):
    """Check that passes_scrutiny matches expected verdict."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        expected = ctx.expected_output
        output = ctx.output
        expected_pass = expected.get("passes_scrutiny")  # type: ignore[union-attr]
        actual_pass = getattr(output, "passes_scrutiny", None)
        if actual_pass is None:
            return EvaluationReason(
                value=False, reason="Output missing passes_scrutiny field"
            )
        if actual_pass == expected_pass:
            return EvaluationReason(
                value=True,
                reason=f"Correctly {'accepted' if actual_pass else 'rejected'} claim",
            )
        return EvaluationReason(
            value=False,
            reason=f"Expected passes_scrutiny={expected_pass}, got {actual_pass}",
        )


class RecommendationValid(Evaluator[dict, object, dict]):
    """Check that recommendation is one of promote/hold/demote and consistent with verdict."""

    VALID = {"promote", "hold", "demote"}

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        output = ctx.output
        rec = getattr(output, "recommendation", "")
        passes = getattr(output, "passes_scrutiny", None)

        if rec not in self.VALID:
            return EvaluationReason(
                value=False,
                reason=f"Invalid recommendation '{rec}', expected one of {self.VALID}",
            )

        if passes and rec == "demote":
            return EvaluationReason(
                value=False,
                reason="Inconsistent: passes_scrutiny=True but recommends demote",
            )
        if not passes and rec == "promote":
            return EvaluationReason(
                value=False,
                reason="Inconsistent: passes_scrutiny=False but recommends promote",
            )

        return EvaluationReason(
            value=True, reason=f"Recommendation '{rec}' is valid and consistent"
        )


class ConfidenceInRange(Evaluator[dict, object, dict]):
    """Check that confidence_estimate is between 0 and 1."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        conf = getattr(ctx.output, "confidence_estimate", None)
        if conf is None:
            return EvaluationReason(value=False, reason="Missing confidence_estimate")
        if not (0.0 <= conf <= 1.0):
            return EvaluationReason(
                value=False, reason=f"confidence_estimate {conf} outside [0,1]"
            )
        return EvaluationReason(
            value=True, reason=f"Confidence {conf:.2f} in valid range"
        )


class IssuesDetected(Evaluator[dict, object, dict]):
    """For weak claims, check that issues were actually found."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        expected = ctx.expected_output
        should_find_issues = expected.get("should_find_issues", False)  # type: ignore[union-attr]
        issues = getattr(ctx.output, "issues_found", [])
        if should_find_issues and len(issues) == 0:
            return EvaluationReason(
                value=False, reason="Expected issues to be found but none were reported"
            )
        if should_find_issues and len(issues) > 0:
            return EvaluationReason(
                value=True, reason=f"Correctly found {len(issues)} issue(s)"
            )
        return EvaluationReason(value=True, reason="Issue detection check passed")


# ── Task Function ────────────────────────────────────────────────────────


async def scrutinise_task(inputs: dict) -> object:
    """Run the scrutinise_claim agent with given inputs."""
    return await run_agent(
        "epistemic_scrutinise_claim",
        claim_id=inputs["claim_id"],
        claim_statement=inputs["claim_statement"],
        claim_scope=inputs.get("claim_scope", "general"),
        evidence_summaries=inputs["evidence_summaries"],
    )


# ── Dataset ──────────────────────────────────────────────────────────────

CASES = [
    # ── SHOULD PASS despite complications ──
    Case(
        name="retracted_supporting_mixed_with_strong",
        inputs={
            "claim_id": "C001",
            "claim_statement": "Gut microbiome composition is associated with response to immune checkpoint inhibitors in melanoma patients.",
            "claim_scope": "Adult melanoma patients receiving anti-PD-1 therapy",
            "evidence_summaries": [
                "Gopalakrishnan et al. (Science 2018) found that melanoma patients responding to anti-PD-1 had higher gut microbiome diversity and enrichment of Faecalibacterium (N=112, p<0.01).",
                "Matson et al. (Science 2018) independently found distinct gut microbiome signatures in anti-PD-1 responders vs non-responders across multiple melanoma cohorts (N=42).",
                "Routy et al. (Science 2018) showed that antibiotic use (disrupting microbiome) was associated with worse outcomes on checkpoint inhibitors in a retrospective analysis of 249 patients across multiple cancer types.",
                "NOTE: An earlier study by Derosa et al. claiming microbiome transplant could restore checkpoint inhibitor response was retracted in 2023 due to data fabrication concerns.",
            ],
        },
        expected_output={"passes_scrutiny": True, "should_find_issues": True},
        metadata={
            "difficulty": "hard",
            "tests": "Can the agent promote despite a retracted study in the mix, recognising the other evidence is independent?",
        },
    ),
    Case(
        name="true_but_trivially_scoped",
        inputs={
            "claim_id": "C002",
            "claim_statement": "Lithium carbonate reduces suicide risk in patients with bipolar disorder.",
            "claim_scope": "Patients with bipolar I disorder on therapeutic lithium levels (0.6-1.2 mEq/L)",
            "evidence_summaries": [
                "Cipriani et al. (BMJ 2013) meta-analysis of 48 RCTs found lithium reduced suicide risk by 60% compared to placebo in mood disorders (OR 0.13, 95% CI 0.03-0.66).",
                "A Swedish national registry study (N=51,535, Lancet Psychiatry 2018) found 14% lower rate of suicide-related events during lithium treatment periods vs non-treatment periods within the same individuals.",
                "Baldessarini et al. (2006) systematic review found consistent anti-suicidal effects across 33 studies spanning 3 decades.",
            ],
        },
        expected_output={"passes_scrutiny": True, "should_find_issues": True},
        metadata={
            "difficulty": "medium",
            "tests": "Strong evidence but scope is narrow — agent should promote but note scope limitation",
        },
    ),
    # ── SHOULD FAIL: subtle problems ──
    Case(
        name="correlation_presented_as_causation",
        inputs={
            "claim_id": "C003",
            "claim_statement": "Eating organic food prevents cancer.",
            "claim_scope": "General adult population",
            "evidence_summaries": [
                "The NutriNet-Santé prospective cohort study (N=68,946, JAMA Internal Medicine 2018) found that participants reporting highest organic food consumption had 25% lower overall cancer risk (HR 0.75, 95% CI 0.63-0.88).",
                "A UK Million Women Study analysis found no significant association between organic food consumption and cancer risk after adjusting for lifestyle factors (N=623,080).",
                "The French study did not control for overall dietary quality, physical activity level, or socioeconomic status — all known confounders that correlate with organic food purchasing behaviour.",
            ],
        },
        expected_output={"passes_scrutiny": False, "should_find_issues": True},
        metadata={
            "difficulty": "hard",
            "tests": "Large prestigious study but uncontrolled confounders + contradicted by larger study. Causal claim from observational data.",
        },
    ),
    Case(
        name="survivorship_bias",
        inputs={
            "claim_id": "C004",
            "claim_statement": "Early-stage startups that receive venture capital funding have a 90% probability of eventual profitability.",
            "claim_scope": "Technology startups in the US receiving Series A funding",
            "evidence_summaries": [
                "A 2022 analysis by a prominent VC firm of their own portfolio (N=200 companies funded 2010-2015) found that 87% of companies that survived to year 5 achieved profitability.",
                "CB Insights reports that the overall failure rate for VC-backed startups is approximately 70-75%, but notes this includes companies at all stages.",
                "The VC firm's analysis excluded companies that shut down before year 5 from the denominator, and included only companies where the firm maintained board seats.",
            ],
        },
        expected_output={"passes_scrutiny": False, "should_find_issues": True},
        metadata={
            "difficulty": "hard",
            "tests": "Survivorship bias explicitly noted. Must recognise that excluding failures from the denominator invalidates the claim.",
        },
    ),
    Case(
        name="ecological_fallacy",
        inputs={
            "claim_id": "C005",
            "claim_statement": "Countries with higher chocolate consumption per capita have more Nobel Prize laureates, demonstrating that chocolate enhances cognitive function.",
            "claim_scope": "General population cognitive enhancement",
            "evidence_summaries": [
                "Messerli (NEJM 2012) found a strong linear correlation (r=0.791, p<0.0001) between per-capita chocolate consumption and the number of Nobel Prize laureates per 10 million population across 23 countries.",
                "The correlation was published in the New England Journal of Medicine, one of the highest-impact medical journals globally.",
                "Sweden was identified as an outlier with more laureates than predicted by chocolate consumption alone.",
            ],
        },
        expected_output={"passes_scrutiny": False, "should_find_issues": True},
        metadata={
            "difficulty": "hard",
            "tests": "Published in NEJM (high-prestige journal) but is a well-known example of ecological fallacy and spurious correlation. Must not be swayed by journal prestige.",
        },
    ),
    Case(
        name="scope_mismatch_between_evidence_and_claim",
        inputs={
            "claim_id": "C006",
            "claim_statement": "SSRIs are ineffective for treating depression.",
            "claim_scope": "All forms of depression across all populations",
            "evidence_summaries": [
                "Kirsch et al. (PLoS Medicine 2008) meta-analysis of FDA-submitted trials found that the effect of SSRIs vs placebo fell below clinical significance (d=0.32) for mild-to-moderate depression.",
                "Fournier et al. (JAMA 2010) found that SSRI-placebo differences were clinically meaningful only for patients with severe depression (Hamilton scores ≥25).",
                "Cipriani et al. (Lancet 2018) network meta-analysis of 522 trials (N=116,477) found all 21 antidepressants studied were more effective than placebo, with SSRIs showing consistent moderate effects.",
            ],
        },
        expected_output={"passes_scrutiny": False, "should_find_issues": True},
        metadata={
            "difficulty": "hard",
            "tests": "Claim says 'ineffective' but evidence shows they work for severe depression. Scope mismatch: evidence differentiates by severity, claim doesn't. Third source contradicts.",
        },
    ),
    Case(
        name="genuinely_uncertain_emerging_field",
        inputs={
            "claim_id": "C007",
            "claim_statement": "Senolytics — drugs that selectively eliminate senescent cells — extend healthy lifespan in humans.",
            "claim_scope": "Healthy ageing in adult humans",
            "evidence_summaries": [
                "Baker et al. (Nature 2016) showed that clearing senescent cells extended median lifespan by 17-35% in a transgenic mouse model (BubR1 progeroid mice).",
                "A Phase I trial of dasatinib+quercetin in 14 patients with idiopathic pulmonary fibrosis (Lancet EBioMedicine 2019) found reduced senescent cell markers and improved physical function over 3 weeks.",
                "Xu et al. (Nature Medicine 2018) demonstrated that transplanting senescent cells into young mice caused persistent physical dysfunction, reversed by senolytics.",
                "No Phase II or Phase III trials of senolytics for general healthy ageing have been completed as of 2025. The human evidence is limited to small disease-specific studies.",
            ],
        },
        expected_output={"passes_scrutiny": False, "should_find_issues": True},
        metadata={
            "difficulty": "hard",
            "tests": "Strong preclinical evidence from top journals, but claim is about humans and evidence is from mice + tiny Phase I. Must distinguish species gap.",
        },
    ),
]


def build_dataset() -> Dataset:
    return Dataset(
        name="scrutinise_claim",
        cases=CASES,
        evaluators=[
            ScrutinyVerdictCorrect(),
            RecommendationValid(),
            ConfidenceInRange(),
            IssuesDetected(),
            LLMJudge(
                rubric=(
                    "Assess whether the agent demonstrated genuine epistemic judgment — not just "
                    "pattern matching. Did it identify the specific methodological or logical flaw "
                    "in each case? Did it correctly weigh evidence quality against claim strength? "
                    "For cases that should pass, did it promote despite complications (e.g., a "
                    "retracted study in the mix)? For cases that should fail, did it identify the "
                    "specific reason (survivorship bias, ecological fallacy, scope mismatch, etc.) "
                    "rather than giving a generic rejection?"
                ),
                include_input=True,
                include_expected_output=True,
            ),
        ],
    )


async def run_eval(max_concurrency: int = 3):
    ds = build_dataset()
    return await ds.evaluate(scrutinise_task, max_concurrency=max_concurrency)
