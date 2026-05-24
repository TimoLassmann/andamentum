"""Eval: epistemic_write_answer agent.

Tests synthesis judgment on cases where faithful representation is hard:
contradictory claims at different confidence levels, blocking uncertainties
that limit conclusions, and questions where the honest answer is "we don't know".

Run:
    cd packages/epistemic
    uv run python agent_evals/run_all.py --agent write_answer
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


class HasTitle(Evaluator[dict, object, dict]):
    """Check that title is present and reasonable length."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        title = getattr(ctx.output, "title", "")
        if not title or len(title.strip()) < 5:
            return EvaluationReason(value=False, reason="Title is empty or too short")
        if len(title) > 200:
            return EvaluationReason(
                value=False, reason=f"Title too long ({len(title)} chars)"
            )
        return EvaluationReason(value=True, reason=f"Title present: '{title[:60]}...'")


class AnswerSubstantive(Evaluator[dict, object, dict]):
    """Check that the answer is substantive and not a stub."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        answer = getattr(ctx.output, "answer", "")
        if not answer:
            return EvaluationReason(value=False, reason="Answer is empty")
        word_count = len(answer.split())
        if word_count < 50:
            return EvaluationReason(
                value=False, reason=f"Answer too short ({word_count} words)"
            )
        return EvaluationReason(value=True, reason=f"Answer is {word_count} words")


class AnswerMentionsKeyTerms(Evaluator[dict, object, dict]):
    """Check that the answer references key terms from the claims."""

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, EvaluationReason]:
        expected = ctx.expected_output or {}
        must_mention = expected.get("must_mention", [])  # type: ignore[union-attr]
        if not must_mention:
            return {
                "key_terms": EvaluationReason(
                    value=True, reason="No must_mention specified"
                )
            }

        answer = getattr(ctx.output, "answer", "").lower()
        results = {}
        for term in must_mention:
            found = term.lower() in answer
            results[f"mentions_{term}"] = EvaluationReason(
                value=found,
                reason=f"{'Found' if found else 'Missing'} expected term '{term}'",
            )
        return results


class AnswerReflectsUncertainty(Evaluator[dict, object, dict]):
    """When uncertainties are provided, the answer should reflect them."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        uncertainties = ctx.inputs.get("uncertainties", "")
        if not uncertainties:
            return EvaluationReason(value=True, reason="No uncertainties to check")

        answer = getattr(ctx.output, "answer", "").lower()
        hedge_words = [
            "however",
            "caveat",
            "limitation",
            "uncertain",
            "unclear",
            "debate",
            "caution",
            "note that",
            "although",
            "while",
            "insufficient",
            "cannot",
            "premature",
            "limited",
        ]
        has_hedging = any(w in answer for w in hedge_words)
        if has_hedging:
            return EvaluationReason(
                value=True,
                reason="Answer reflects uncertainty with appropriate hedging",
            )
        return EvaluationReason(
            value=False,
            reason="Uncertainties were provided but answer lacks hedging language",
        )


# ── Task Function ────────────────────────────────────────────────────────


async def write_answer_task(inputs: dict) -> object:
    """Run the write_answer agent."""
    return await run_agent(
        "epistemic_write_answer",
        objective=inputs["objective"],
        claims_summary=inputs["claims_summary"],
        evidence_summary=inputs.get("evidence_summary", ""),
        uncertainties=inputs.get("uncertainties", ""),
    )


# ── Dataset ──────────────────────────────────────────────────────────────

CASES = [
    Case(
        name="honest_null_result",
        inputs={
            "objective": "Do brain training games (e.g., Lumosity) improve general cognitive function?",
            "claims_summary": (
                "Claim 1 [SUPPORTED]: Brain training games improve performance on the specific tasks "
                "trained (near transfer), with moderate effect sizes (d=0.4-0.6). "
                "Claim 2 [HYPOTHESIS]: Brain training does NOT transfer to general cognitive ability, "
                "academic performance, or real-world functioning (far transfer). The largest RCT "
                "(ACTIVE trial, N=2,832) found no significant far transfer effects at 5-year follow-up."
            ),
            "evidence_summary": (
                "A 2016 FTC settlement found Lumosity's advertising claims about preventing cognitive "
                "decline were deceptive. A consensus statement signed by 73 cognitive scientists "
                "concluded there is no evidence that brain games improve general cognition."
            ),
            "uncertainties": (
                "U1 [BLOCKING]: The definition of 'cognitive improvement' varies across studies, "
                "making meta-analysis difficult. "
                "U2 [NON-BLOCKING]: Some positive findings exist for older adults with mild cognitive "
                "impairment, but these may not generalise to healthy populations."
            ),
        },
        expected_output={
            "must_mention": ["transfer", "FTC"],
        },
        metadata={
            "tests": "The honest answer is 'no, not for general cognition'. Agent must not soften this to be diplomatic — the evidence is clear."
        },
    ),
    Case(
        name="contradictory_claims_different_stages",
        inputs={
            "objective": "Is screen time harmful to children's development?",
            "claims_summary": (
                "Claim 1 [ROBUST]: Excessive screen time (>2 hours/day) in children under 5 is "
                "associated with delayed language development and reduced sleep quality. Multiple "
                "longitudinal studies with consistent findings. Adversarial balance: 0.75. "
                "Claim 2 [SUPPORTED]: Educational screen content (e.g., Sesame Street) has measurable "
                "positive effects on school readiness in disadvantaged children. Adversarial balance: 0.80. "
                "Claim 3 [HYPOTHESIS]: Social media use causes depression in teenagers. Evidence is "
                "primarily correlational; the causal direction is contested. Adversarial balance: 0.35."
            ),
            "evidence_summary": (
                "WHO guidelines recommend no screen time for children under 1, <1 hour for 1-4. "
                "Longitudinal data supports dose-response relationship for early childhood. "
                "For adolescents, a 2024 US Surgeon General's advisory cited 'growing body of "
                "evidence' but acknowledged methodological limitations."
            ),
            "uncertainties": (
                "U1 [BLOCKING]: Screen time research cannot establish causation — families that "
                "limit screens differ from those that don't on many confounders. "
                "U2 [NON-BLOCKING]: The type of content matters more than total hours — a distinction "
                "often lost in broad 'screen time' measures."
            ),
        },
        expected_output={
            "must_mention": ["language", "Sesame Street"],
        },
        metadata={
            "tests": "Three claims at different epistemic stages. Answer must calibrate confidence per-claim, not give a single verdict. Claim 3 (social media → depression) is only HYPOTHESIS with low adversarial balance — must be hedged heavily."
        },
    ),
    Case(
        name="answer_must_acknowledge_limits",
        inputs={
            "objective": "Will large language models achieve artificial general intelligence?",
            "claims_summary": (
                "Claim 1 [HYPOTHESIS]: Current LLM architectures exhibit emergent capabilities that "
                "suggest a path toward general intelligence. No consensus on whether this constitutes "
                "progress toward AGI. Adversarial balance: 0.30. "
                "Claim 2 [HYPOTHESIS]: LLMs fundamentally lack grounded understanding, causal reasoning, "
                "and persistent memory — capabilities likely required for AGI. Adversarial balance: 0.40."
            ),
            "evidence_summary": (
                "No empirical evidence directly addresses this question. Expert surveys show deep "
                "disagreement: median estimate for AGI ranges from 2040 to 'never' depending on "
                "which researchers are surveyed and how AGI is defined."
            ),
            "uncertainties": (
                "U1 [BLOCKING]: No agreed-upon definition of AGI exists. "
                "U2 [BLOCKING]: The question is inherently predictive — no amount of current "
                "evidence can definitively answer it. "
                "U3 [BLOCKING]: Expert opinion is deeply divided with no convergence."
            ),
        },
        expected_output={
            "must_mention": ["definition", "uncertain"],
        },
        metadata={
            "tests": "Three blocking uncertainties, two HYPOTHESIS claims, no robust evidence. The honest answer is 'we cannot answer this reliably'. Agent must not fabricate confidence."
        },
    ),
    Case(
        name="synthesis_with_adversarial_results",
        inputs={
            "objective": "Does the Mediterranean diet reduce cardiovascular disease risk?",
            "claims_summary": (
                "Claim 1 [ROBUST]: Mediterranean diet adherence is associated with 25-30% reduced "
                "risk of major cardiovascular events. Based on PREDIMED trial (N=7,447) and multiple "
                "cohort studies. Adversarial balance: 0.82. "
                "Claim 2 [PROVISIONAL]: The PREDIMED trial, the largest RCT, was retracted and "
                "republished in 2018 after randomisation irregularities were discovered at some "
                "sites. Re-analysis with corrected methods showed similar but slightly attenuated "
                "results. Adversarial balance: 0.65."
            ),
            "evidence_summary": (
                "Strong observational evidence from multiple Mediterranean countries (Lyon Diet Heart "
                "Study, HALE project). The PREDIMED retraction/republication is the most significant "
                "methodological concern. Effect sizes in observational studies may be inflated by "
                "healthy-user bias."
            ),
            "uncertainties": (
                "U1 [NON-BLOCKING]: Healthy-user bias — people who follow Mediterranean diets may "
                "differ in other health behaviours. "
                "U2 [NON-BLOCKING]: The 'Mediterranean diet' is not precisely defined and varies "
                "across studies."
            ),
        },
        expected_output={
            "must_mention": ["PREDIMED", "retract"],
        },
        metadata={
            "tests": "Claim 1 is ROBUST but the key trial was retracted/republished. Agent must mention this — it's the most important nuance. A faithful answer includes the retraction context."
        },
    ),
]


def build_dataset() -> Dataset:
    return Dataset(
        name="write_answer",
        cases=CASES,
        evaluators=[
            HasTitle(),
            AnswerSubstantive(),
            AnswerMentionsKeyTerms(),
            AnswerReflectsUncertainty(),
            LLMJudge(
                rubric=(
                    "The agent should produce answers that are faithfully calibrated to the evidence: "
                    "null results should be stated clearly (not softened); claims at different epistemic "
                    "stages must receive different levels of confidence in the prose; blocking "
                    "uncertainties must be acknowledged and should visibly limit the conclusions drawn; "
                    "when the honest answer is 'we don't know', the answer should say so rather than "
                    "manufacturing false confidence. Retracted or methodologically challenged evidence "
                    "must be mentioned even when the overall conclusion is positive."
                ),
                include_input=True,
            ),
        ],
    )


async def run_eval(max_concurrency: int = 3):
    ds = build_dataset()
    return await ds.evaluate(write_answer_task, max_concurrency=max_concurrency)
