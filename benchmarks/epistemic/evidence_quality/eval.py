"""Eval: epistemic_assess_evidence_quality agent.

Tests cases where source quality is ambiguous or where the obvious scoring
doesn't apply. Avoids cases where the prompt's scoring anchors directly
determine the answer (Lancet → 1.0, anonymous blog → 0.1).

Run:
    cd packages/epistemic
    uv run python agent_evals/run_all.py --agent evidence_quality
"""

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason, LLMJudge

from conftest import run_agent


# ── Custom Evaluators ────────────────────────────────────────────────────


class ScoresInRange(Evaluator[dict, object, dict]):
    """All four quality scores must be in [0, 1]."""

    FIELDS = ["source_credibility", "relevance", "specificity", "recency_appropriate"]

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, EvaluationReason]:
        results = {}
        for field in self.FIELDS:
            val = getattr(ctx.output, field, None)
            if val is None:
                results[f"{field}_range"] = EvaluationReason(value=False, reason=f"Missing {field}")
            elif not (0.0 <= val <= 1.0):
                results[f"{field}_range"] = EvaluationReason(value=False, reason=f"{field}={val} outside [0,1]")
            else:
                results[f"{field}_range"] = EvaluationReason(value=True, reason=f"{field}={val:.2f} OK")
        return results


class QualityOrdering(Evaluator[dict, object, dict]):
    """Check that scores respect expected bounds."""

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, EvaluationReason]:
        expected = ctx.expected_output or {}
        results = {}
        for dim in ["source_credibility", "relevance", "specificity", "recency_appropriate"]:
            val = getattr(ctx.output, dim, 0.5)
            lo = expected.get(f"min_{dim}")  # type: ignore[union-attr]
            hi = expected.get(f"max_{dim}")  # type: ignore[union-attr]
            if lo is not None and val < lo:
                results[f"{dim}_ordering"] = EvaluationReason(value=False, reason=f"{dim} {val:.2f} < expected min {lo}")
            elif hi is not None and val > hi:
                results[f"{dim}_ordering"] = EvaluationReason(value=False, reason=f"{dim} {val:.2f} > expected max {hi}")
            elif lo is not None or hi is not None:
                results[f"{dim}_ordering"] = EvaluationReason(value=True, reason=f"{dim} {val:.2f} in expected range")
        return results


class HasJustification(Evaluator[dict, object, dict]):
    """Justification field must be non-empty and substantive."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        justification = getattr(ctx.output, "justification", "")
        if not justification or len(justification.strip()) < 10:
            return EvaluationReason(value=False, reason="Justification is empty or too short")
        return EvaluationReason(value=True, reason=f"Justification present ({len(justification)} chars)")


# ── Task Function ────────────────────────────────────────────────────────


async def evidence_quality_task(inputs: dict) -> object:
    """Run the assess_evidence_quality agent."""
    return await run_agent(
        "epistemic_assess_evidence_quality",
        claim_statement=inputs["claim_statement"],
        evidence_content=inputs["evidence_content"],
        source_ref=inputs.get("source_ref", "unknown"),
    )


# ── Dataset ──────────────────────────────────────────────────────────────

CASES = [
    Case(
        name="retracted_nature_paper",
        inputs={
            "claim_statement": "LK-99 is a room-temperature ambient-pressure superconductor.",
            "evidence_content": (
                "Lee et al. (Nature, 2023 — RETRACTED) reported that a lead-apatite compound "
                "(Pb₁₀₋ₓCuₓ(PO₄)₆O, dubbed LK-99) exhibited zero resistivity and diamagnetic "
                "levitation at room temperature and ambient pressure. The paper was published in "
                "Nature but retracted 4 months later after multiple independent replication attempts "
                "failed to reproduce the results. The observed phenomena were attributed to Cu₂S "
                "impurities causing a structural phase transition, not superconductivity."
            ),
            "source_ref": "doi:10.1038/nature-retracted-example",
        },
        expected_output={"max_source_credibility": 0.3},
        metadata={"tests": "Retracted paper from top journal. Prestige should NOT override retraction status."},
    ),
    Case(
        name="arxiv_from_known_group",
        inputs={
            "claim_statement": "Large language models can perform chain-of-thought reasoning to solve multi-step math problems.",
            "evidence_content": (
                "Wei et al. (arXiv:2201.11903, 2022, Google Brain) demonstrated that prompting large "
                "language models with a chain of thought — a series of intermediate reasoning steps — "
                "significantly improved performance on arithmetic, commonsense, and symbolic reasoning "
                "benchmarks. On the GSM8K math benchmark, chain-of-thought prompting with PaLM 540B "
                "achieved 58.1% solve rate vs 17.9% for standard prompting. The paper was not "
                "peer-reviewed at publication time but has since been cited over 5,000 times and the "
                "findings have been independently replicated by multiple groups."
            ),
            "source_ref": "arxiv:2201.11903",
        },
        expected_output={"min_source_credibility": 0.6},
        metadata={"tests": "Not peer-reviewed (arXiv) but from major lab, massively cited, independently replicated. Credibility is ambiguous."},
    ),
    Case(
        name="industry_funded_with_positive_results",
        inputs={
            "claim_statement": "Drug X reduces LDL cholesterol by 50% in patients with familial hypercholesterolemia.",
            "evidence_content": (
                "The ODYSSEY FH I trial (NEJM 2015) randomised 486 patients with heterozygous familial "
                "hypercholesterolemia to alirocumab or placebo. At 24 weeks, alirocumab reduced LDL "
                "cholesterol by 57.9% vs 0.7% for placebo (p<0.001). The trial was funded by "
                "Sanofi and Regeneron Pharmaceuticals, who manufacture alirocumab. 8 of 12 authors "
                "disclosed financial relationships with the sponsors. The primary endpoint was a "
                "surrogate marker (LDL reduction), not clinical outcomes like cardiovascular events."
            ),
            "source_ref": "doi:10.1056/NEJMoa1501031",
        },
        expected_output={"min_source_credibility": 0.5, "max_source_credibility": 0.85},
        metadata={"tests": "NEJM publication but industry-funded with surrogate endpoint. Should not get maximum credibility despite top journal."},
    ),
    Case(
        name="who_guideline_as_evidence",
        inputs={
            "claim_statement": "Exclusive breastfeeding for the first 6 months reduces infant mortality in low-income settings.",
            "evidence_content": (
                "WHO Guideline: Protecting, promoting and supporting breastfeeding in facilities "
                "providing maternity and newborn services (2017). The WHO strongly recommends exclusive "
                "breastfeeding for the first 6 months of life based on systematic reviews of evidence "
                "from 47 countries. The recommendation is graded as 'strong' with 'moderate quality "
                "evidence'. The guideline notes that the underlying evidence is predominantly "
                "observational, as RCTs of breastfeeding vs non-breastfeeding are ethically infeasible."
            ),
            "source_ref": "WHO/NMH/NHD/17.1",
        },
        expected_output={"min_source_credibility": 0.7, "max_relevance": 0.9},
        metadata={"tests": "WHO guideline is authoritative but is a secondary synthesis, not primary evidence. Explicitly notes moderate quality."},
    ),
    Case(
        name="high_quality_but_wrong_domain",
        inputs={
            "claim_statement": "Cognitive behavioural therapy is effective for treating chronic lower back pain.",
            "evidence_content": (
                "A Cochrane systematic review (2020) of 75 RCTs (N=9,000+) found strong evidence "
                "that CBT produces moderate improvements in anxiety disorder symptoms (SMD=-0.56, "
                "95% CI -0.69 to -0.43) compared to waitlist controls. CBT was found effective "
                "across generalised anxiety disorder, social anxiety disorder, and panic disorder. "
                "The review did not examine pain conditions."
            ),
            "source_ref": "doi:10.1002/14651858.CD013574.pub2",
        },
        expected_output={"min_source_credibility": 0.8, "max_relevance": 0.3},
        metadata={"tests": "Excellent methodology (Cochrane SR) but wrong domain — anxiety, not pain. Must score high credibility but low relevance."},
    ),
    Case(
        name="preregistered_null_result",
        inputs={
            "claim_statement": "Power posing increases testosterone levels and risk-taking behaviour.",
            "evidence_content": (
                "Ranehill et al. (Psychological Science 2015) conducted a pre-registered direct "
                "replication (N=200, double the original sample) of Carney, Cuddy, & Yap's 2010 "
                "power posing study. The replication found no significant effects of power posing "
                "on testosterone (p=0.30), cortisol (p=0.88), or risk-taking behaviour (p=0.14). "
                "The original lead author later acknowledged the effects were likely not real. "
                "The replication was pre-registered on the Open Science Framework and used identical "
                "materials and procedures to the original study."
            ),
            "source_ref": "doi:10.1177/0956797614553946",
        },
        expected_output={"min_source_credibility": 0.8, "min_relevance": 0.8},
        metadata={"tests": "Pre-registered replication failure directly contradicting the claim. High quality AND high relevance — as evidence AGAINST the claim."},
    ),
    Case(
        name="animal_model_for_human_claim",
        inputs={
            "claim_statement": "Resveratrol supplementation extends human lifespan.",
            "evidence_content": (
                "Baur et al. (Nature 2006) found that resveratrol (22 mg/kg/day) improved survival "
                "and motor function in mice fed a high-calorie diet. The resveratrol-treated mice "
                "showed a 31% reduction in mortality risk (p<0.005). Effects were associated with "
                "increased SIRT1 activation and improved mitochondrial function. The dose used in "
                "mice is equivalent to approximately 1,500 mg/day in a 70 kg human — far exceeding "
                "amounts obtainable from dietary sources (red wine contains ~1.5 mg/L)."
            ),
            "source_ref": "doi:10.1038/nature05354",
        },
        expected_output={"min_source_credibility": 0.7, "max_relevance": 0.4},
        metadata={"tests": "Top journal, rigorous methodology, but mouse model for a human lifespan claim. Relevance should be low due to species gap and dose scaling."},
    ),
]


def build_dataset() -> Dataset:
    return Dataset(
        name="evidence_quality",
        cases=CASES,
        evaluators=[
            ScoresInRange(),
            QualityOrdering(),
            HasJustification(),
            LLMJudge(
                rubric=(
                    "The agent should demonstrate nuanced quality assessment: retracted papers "
                    "should have low credibility regardless of journal prestige; industry funding "
                    "and surrogate endpoints should temper credibility; evidence from wrong domains "
                    "should have low relevance despite high methodology; animal studies should have "
                    "limited relevance for human claims. The justification should identify the "
                    "specific quality issue, not just describe the source."
                ),
                include_input=True,
                include_expected_output=True,
            ),
        ],
    )


async def run_eval(max_concurrency: int = 3):
    ds = build_dataset()
    return await ds.evaluate(evidence_quality_task, max_concurrency=max_concurrency)
