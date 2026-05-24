"""Eval: epistemic_adversarial_search agent.

Tests adversarial evaluation on claims where the right recommendation requires
genuine judgment: recently-shifted consensus, claims with both strong and weak
criticism available, and claims where the obvious adversarial strategy is wrong.

Run:
    cd packages/epistemic
    uv run python agent_evals/run_all.py --agent adversarial_search
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


class RecommendationValid(Evaluator[dict, object, dict]):
    """Check recommendation is one of the valid values."""

    VALID = {"maintain", "weaken", "modify", "refute"}

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        rec = getattr(ctx.output, "recommendation", "")
        if rec in self.VALID:
            return EvaluationReason(value=True, reason=f"Valid recommendation: {rec}")
        return EvaluationReason(
            value=False,
            reason=f"Invalid recommendation '{rec}', expected one of {self.VALID}",
        )


class RecommendationMatchesExpected(Evaluator[dict, object, dict]):
    """Check that the recommendation direction matches expected."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        expected = ctx.expected_output or {}
        expected_recs = expected.get("expected_recommendations", [])  # type: ignore[union-attr]
        if not expected_recs:
            return EvaluationReason(
                value=True, reason="No expected recommendation specified"
            )

        rec = getattr(ctx.output, "recommendation", "")
        if rec in expected_recs:
            return EvaluationReason(
                value=True,
                reason=f"Recommendation '{rec}' matches expected {expected_recs}",
            )
        return EvaluationReason(
            value=False,
            reason=f"Recommendation '{rec}' not in expected {expected_recs}",
        )


class CounterargumentsPresent(Evaluator[dict, object, dict]):
    """For claims with known weaknesses, counterarguments should be found."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        expected = ctx.expected_output or {}
        should_find = expected.get("should_find_counterarguments", True)  # type: ignore[union-attr]
        counterargs = getattr(ctx.output, "counterarguments", [])

        if should_find and not counterargs:
            return EvaluationReason(
                value=False, reason="Expected counterarguments but none found"
            )
        if should_find and counterargs:
            return EvaluationReason(
                value=True, reason=f"Found {len(counterargs)} counterargument(s)"
            )
        return EvaluationReason(value=True, reason="Counterargument check passed")


class QueriesGenerated(Evaluator[dict, object, dict]):
    """Check that adversarial queries were generated."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        queries = getattr(ctx.output, "queries_executed", [])
        if not queries:
            return EvaluationReason(
                value=False, reason="No adversarial queries generated"
            )
        if len(queries) < 2:
            return EvaluationReason(
                value=False,
                reason=f"Only {len(queries)} query, expected ≥2 for thorough search",
            )
        return EvaluationReason(
            value=True, reason=f"{len(queries)} adversarial queries generated"
        )


class StrongestCriticismPresent(Evaluator[dict, object, dict]):
    """When counterarguments exist, strongest_criticism should be substantive."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        counterargs = getattr(ctx.output, "counterarguments", [])
        strongest = getattr(ctx.output, "strongest_criticism", "")
        if counterargs and (not strongest or len(strongest.strip()) < 10):
            return EvaluationReason(
                value=False,
                reason="Has counterarguments but strongest_criticism is empty",
            )
        return EvaluationReason(value=True, reason="Strongest criticism check passed")


# ── Task Function ────────────────────────────────────────────────────────


async def adversarial_search_task(inputs: dict) -> object:
    """Run the adversarial_search agent."""
    return await run_agent(
        "epistemic_adversarial_search",
        claim_statement=inputs["claim_statement"],
        claim_scope=inputs.get("claim_scope", "general"),
        existing_evidence=inputs.get("existing_evidence", ""),
    )


# ── Dataset ──────────────────────────────────────────────────────────────

CASES = [
    Case(
        name="recently_reversed_consensus",
        inputs={
            "claim_statement": "Daily low-dose aspirin (81mg) should be recommended for primary prevention of cardiovascular disease in adults aged 40-70 without prior cardiovascular events.",
            "claim_scope": "Adults aged 40-70 without history of cardiovascular events",
            "existing_evidence": (
                "Multiple older RCTs (2000-2015) supported aspirin for primary prevention. However, "
                "the ASPREE (2018), ARRIVE (2018), and ASCEND (2018) trials found no net benefit for "
                "most adults. The USPSTF changed its recommendation in 2022 from 'recommended for "
                "adults 50-59' to 'should not be initiated in adults ≥60' and weakened the recommendation "
                "for younger adults."
            ),
        },
        expected_output={
            "expected_recommendations": ["weaken", "modify"],
            "should_find_counterarguments": True,
        },
        metadata={
            "tests": "Consensus has genuinely shifted. Agent must recognise this isn't 'fringe criticism' but mainstream evidence reversal."
        },
    ),
    Case(
        name="valid_claim_with_noisy_criticism",
        inputs={
            "claim_statement": "Anthropogenic greenhouse gas emissions are the dominant cause of observed global warming since the mid-20th century.",
            "claim_scope": "Global average temperature change since 1950",
            "existing_evidence": (
                "IPCC AR6 (2021): 'It is unequivocal that human influence has warmed the atmosphere, "
                "ocean and land.' Attribution studies estimate human contribution at 1.0-1.2°C of the "
                "observed 1.1°C warming. Natural forcing alone cannot explain observed trends."
            ),
        },
        expected_output={
            "expected_recommendations": ["maintain"],
            "should_find_counterarguments": False,
        },
        metadata={
            "tests": "Overwhelming scientific consensus. Adversarial search will find climate denial — agent must classify this as fringe/non-credible rather than legitimate criticism."
        },
    ),
    Case(
        name="correct_but_scope_needs_modification",
        inputs={
            "claim_statement": "Mask mandates in schools significantly reduce COVID-19 transmission.",
            "claim_scope": "K-12 schools in the United States",
            "existing_evidence": (
                "CDC MMWR (2021): Arizona counties with school mask requirements had 3.5x lower "
                "case rates. However, the Cochrane review (Jefferson et al., 2023) of 78 RCTs "
                "found no clear evidence that masks in community settings reduce respiratory virus "
                "transmission. A Bangladesh cluster-RCT (Abaluck et al., Science 2022) found "
                "surgical masks reduced symptomatic seroprevalence by 11% but cloth masks showed "
                "no significant effect."
            ),
        },
        expected_output={
            "expected_recommendations": ["modify", "weaken"],
            "should_find_counterarguments": True,
        },
        metadata={
            "tests": "Claim uses 'significantly' but evidence is mixed. Cochrane found no clear effect. The CDC study is observational. Must distinguish RCT evidence from observational."
        },
    ),
    Case(
        name="technically_true_but_misleading",
        inputs={
            "claim_statement": "Nuclear energy has the lowest death rate per unit of energy produced of any major energy source.",
            "claim_scope": "Deaths per TWh across all energy sources including renewables",
            "existing_evidence": (
                "Our World in Data (based on Markandya & Wilkinson, Lancet 2007; updated with "
                "Sovacool et al. 2016): Nuclear causes ~0.03 deaths/TWh vs coal 24.6, oil 18.4, "
                "gas 2.8, wind 0.04, solar 0.05. These figures include Chernobyl and Fukushima. "
                "However, estimates of Chernobyl deaths range from 31 (immediate) to 4,000 (WHO) "
                "to 93,000 (Greenpeace), creating large uncertainty in the nuclear figure."
            ),
        },
        expected_output={
            "expected_recommendations": ["maintain", "modify"],
            "should_find_counterarguments": True,
        },
        metadata={
            "tests": "Claim is likely correct by most analyses, but the uncertainty in Chernobyl death estimates is a legitimate methodological criticism. Agent should find this nuance without over-weighting it."
        },
    ),
    Case(
        name="replication_crisis_claim",
        inputs={
            "claim_statement": "Ego depletion — the idea that self-control draws from a limited mental resource that can be exhausted — is a well-established psychological phenomenon.",
            "claim_scope": "Human self-control and decision-making",
            "existing_evidence": (
                "Baumeister et al. (1998) introduced the ego depletion model with a now-famous "
                "experiment (N=67). The original paper has been cited over 6,000 times. "
                "A pre-registered multi-lab replication (Hagger et al., 2016, 23 labs, N=2,141) "
                "found an effect size of d=0.04 (95% CI: -0.07 to 0.15), effectively null. "
                "Baumeister disputed the replication methodology."
            ),
        },
        expected_output={
            "expected_recommendations": ["weaken", "refute"],
            "should_find_counterarguments": True,
        },
        metadata={
            "tests": "Highly cited original finding that failed large-scale replication. The 6,000 citations should NOT be evidence for the claim — they reflect influence, not truth."
        },
    ),
]


def build_dataset() -> Dataset:
    return Dataset(
        name="adversarial_search",
        cases=CASES,
        evaluators=[
            RecommendationValid(),
            RecommendationMatchesExpected(),
            CounterargumentsPresent(),
            QueriesGenerated(),
            StrongestCriticismPresent(),
            LLMJudge(
                rubric=(
                    "The agent should demonstrate calibrated adversarial judgment. For claims where "
                    "consensus has genuinely shifted (aspirin), it should find the new evidence and "
                    "recommend modification. For claims backed by overwhelming consensus (climate), it "
                    "should classify contrarian criticism appropriately and recommend maintain. For "
                    "claims affected by the replication crisis, it should weigh large-scale replications "
                    "over citation counts. The strongest_criticism should be specific and steelmanned, "
                    "not a strawman."
                ),
                include_input=True,
                include_expected_output=True,
            ),
        ],
    )


async def run_eval(max_concurrency: int = 3):
    ds = build_dataset()
    return await ds.evaluate(adversarial_search_task, max_concurrency=max_concurrency)
