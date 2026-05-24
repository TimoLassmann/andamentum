"""Eval: epistemic_extract_evidence agent.

Tests extraction judgment on sources where the right extraction is non-obvious:
mixed-quality content, implicit limitations, contradictions within a source,
and sources where the relevant content is buried in noise.

Run:
    cd packages/epistemic
    uv run python agent_evals/run_all.py --agent extract_evidence
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


class ExtractsRelevantQuotes(Evaluator[dict, object, dict]):
    """Check that relevant_quotes is non-empty and contains substantive content."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        quotes = getattr(ctx.output, "relevant_quotes", [])
        if not quotes:
            return EvaluationReason(value=False, reason="No relevant quotes extracted")
        if all(len(q.strip()) < 20 for q in quotes):
            return EvaluationReason(
                value=False, reason="All quotes are trivially short"
            )
        return EvaluationReason(value=True, reason=f"Extracted {len(quotes)} quote(s)")


class IdentifiesKeyFindings(Evaluator[dict, object, dict]):
    """Check that expected key terms appear somewhere in the extracted content."""

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, EvaluationReason]:
        expected = ctx.expected_output or {}
        must_mention = expected.get("must_mention", [])  # type: ignore[union-attr]
        if not must_mention:
            return {
                "key_findings": EvaluationReason(
                    value=True, reason="No must_mention specified"
                )
            }

        quotes = getattr(ctx.output, "relevant_quotes", [])
        context = getattr(ctx.output, "experimental_context", "")
        all_text = " ".join(quotes).lower() + " " + context.lower()

        results = {}
        for term in must_mention:
            found = term.lower() in all_text
            results[f"mentions_{term}"] = EvaluationReason(
                value=found,
                reason=f"{'Found' if found else 'Missing'} expected term '{term}'",
            )
        return results


class IdentifiesLimitations(Evaluator[dict, object, dict]):
    """For sources with clear limitations, check they're captured."""

    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        expected = ctx.expected_output or {}
        should_find_limitations = expected.get("has_limitations", False)  # type: ignore[union-attr]
        limitations = getattr(ctx.output, "limitations", [])

        if should_find_limitations and not limitations:
            return EvaluationReason(
                value=False, reason="Expected limitations but none found"
            )
        if should_find_limitations and limitations:
            return EvaluationReason(
                value=True, reason=f"Found {len(limitations)} limitation(s)"
            )
        return EvaluationReason(value=True, reason="Limitation check passed")


# ── Task Function ────────────────────────────────────────────────────────


async def extract_evidence_task(inputs: dict) -> object:
    """Run the extract_evidence agent."""
    return await run_agent(
        "epistemic_extract_evidence",
        objective=inputs["objective"],
        source_content=inputs["source_content"],
        source_ref=inputs.get("source_ref", "unknown"),
    )


# ── Dataset ──────────────────────────────────────────────────────────────

CASES = [
    Case(
        name="contradictory_findings_within_source",
        inputs={
            "objective": "Does moderate alcohol consumption have cardiovascular benefits?",
            "source_content": (
                "Wood et al. (Lancet 2018): A pooled analysis of 83 prospective studies in 19 countries "
                "(N=599,912) found a complex dose-response relationship between alcohol and cardiovascular "
                "outcomes. For all-cause mortality, the lowest risk was at approximately zero consumption. "
                "However, for non-fatal myocardial infarction, moderate consumption (100-200 g/week) was "
                "associated with lower risk (HR 0.94, 95% CI 0.91-0.97). For stroke, heart failure, and "
                "fatal hypertensive disease, risk increased monotonically with consumption. The authors "
                "concluded that the threshold for lowest overall risk was approximately 100 g/week "
                "(roughly 7 standard drinks), substantially lower than most national guidelines. "
                "Limitations: observational design cannot establish causation; 'sick quitter' bias may "
                "inflate apparent benefits of moderate drinking; self-reported alcohol consumption is "
                "known to underestimate actual intake by 40-60%."
            ),
            "source_ref": "doi:10.1016/S0140-6736(18)30134-X",
        },
        expected_output={
            "must_mention": ["599,912", "100 g"],
            "has_limitations": True,
        },
        metadata={
            "tests": "Source has contradictory findings for different endpoints. Must extract the nuance, not just one direction."
        },
    ),
    Case(
        name="signal_buried_in_noise",
        inputs={
            "objective": "What is the efficacy of psilocybin for treatment-resistant depression?",
            "source_content": (
                "Press release from Compass Pathways (2022): COMP360 psilocybin therapy Phase IIb results. "
                "CEO statement: 'We are encouraged by these groundbreaking results that could transform "
                "mental health treatment.' The company's stock rose 15% following the announcement. "
                "Analyst Jim Stevens of Goldman Sachs projects $3.2 billion peak revenue. "
                "ACTUAL TRIAL DATA: 233 patients randomised to 25mg, 10mg, or 1mg (control) psilocybin. "
                "Primary endpoint (MADRS change at 3 weeks): 25mg group showed -6.6 point improvement "
                "vs control (p<0.001). Response rate: 37% (25mg) vs 18% (1mg control). Duration of "
                "benefit unclear — efficacy waned by week 12 in most patients. Serious adverse events "
                "occurred in 14% of the 25mg group including suicidal behaviour in 3 patients. "
                "The 10mg dose did not separate from control on the primary endpoint."
            ),
            "source_ref": "compasspathways.com/press-release-2022",
        },
        expected_output={
            "must_mention": ["-6.6", "37%", "suicidal"],
            "has_limitations": True,
        },
        metadata={
            "tests": "Commercial press release with actual data buried among hype. Must extract the trial data and ignore the marketing."
        },
    ),
    Case(
        name="methodological_detail_matters",
        inputs={
            "objective": "Does early childhood bilingualism delay cognitive decline in ageing?",
            "source_content": (
                "Bialystok et al. (Neuropsychologia 2007): Retrospective analysis of clinical records "
                "from a memory clinic in Toronto. 184 patients diagnosed with dementia: 91 bilingual, "
                "93 monolingual. Bilingual patients reported symptom onset 4.1 years later than "
                "monolingual patients (mean age 75.5 vs 71.4, p=0.003). Groups were matched on "
                "cognitive test performance at diagnosis, education level, occupational status, and "
                "immigration status. However, the study relied on patient/family report for age of "
                "symptom onset (recall bias). Selection into a specialist memory clinic may differ "
                "between bilingual and monolingual populations. The study could not control for "
                "physical activity, social engagement, or diet — all known modifiers of cognitive decline."
            ),
            "source_ref": "doi:10.1016/j.neuropsychologia.2006.10.009",
        },
        expected_output={
            "must_mention": ["4.1 years", "recall bias"],
            "has_limitations": True,
        },
        metadata={
            "tests": "Must extract both the finding AND recognise the embedded methodological weaknesses as limitations."
        },
    ),
    Case(
        name="database_with_confidence_scores",
        inputs={
            "objective": "What genetic variants are associated with early-onset Alzheimer's disease?",
            "source_content": (
                "ClinVar database query results for Alzheimer disease, early-onset:\n"
                "1. APP gene, NM_000484.4:c.2149G>T (p.Val717Leu) — Pathogenic (reviewed by expert panel, "
                "   4 submitters concordant). Associated with familial Alzheimer disease type 1.\n"
                "2. PSEN1 gene, NM_000021.4:c.1175C>T (p.Ala392Val) — Pathogenic (criteria provided, "
                "   2 submitters concordant). Most common cause of early-onset familial AD.\n"
                "3. PSEN2 gene, NM_000447.3:c.422A>G (p.Asn141Ser) — Pathogenic/Likely pathogenic "
                "   (1 submitter, no assertion criteria provided). Rare variant, Volga German kindred.\n"
                "4. APOE e4 allele — Risk factor (genome-wide association). Present in ~14% of general "
                "   population, ~40% of late-onset AD patients. NOT considered causative for early-onset AD.\n"
                "Data last updated: 2024-12-01."
            ),
            "source_ref": "clinvar.ncbi.nlm.nih.gov",
        },
        expected_output={
            "must_mention": ["APP", "PSEN1", "PSEN2"],
            "has_limitations": False,
        },
        metadata={
            "tests": "Structured database output with varying confidence levels. Must extract variants AND note the confidence differences (expert panel vs 1 submitter). APOE is a distractor — risk factor for late-onset, not early-onset."
        },
    ),
    Case(
        name="preprint_with_extraordinary_claim",
        inputs={
            "objective": "Can AI systems currently pass the Turing test?",
            "source_content": (
                "Jones et al. (arXiv preprint, 2024): 'GPT-4 Passes the Turing Test'. The authors "
                "conducted an online experiment where 500 participants conversed with either GPT-4 "
                "or a human confederate for 5 minutes. 54% of participants judged GPT-4 to be human, "
                "vs 67% for actual humans. The authors claim this constitutes 'passing' the Turing test. "
                "Notable limitations: conversations were limited to 5 minutes; the original Turing test "
                "proposal specified no time limit; participants were recruited via Mechanical Turk and "
                "may not represent expert judges; GPT-4 was given a persona prompt ('act like a young "
                "adult'); the human confederates were not given specific instructions and some "
                "provided low-effort responses. The preprint has not been peer-reviewed. A response "
                "paper by Marcus & Davis argued the experimental design was insufficiently rigorous "
                "to support the claim."
            ),
            "source_ref": "arxiv:2401.xxxxx",
        },
        expected_output={
            "must_mention": ["54%", "5 minutes"],
            "has_limitations": True,
        },
        metadata={
            "tests": "Extraordinary claim from a preprint with serious methodological issues. Must extract the data but also capture the significant limitations and the response paper."
        },
    ),
]


def build_dataset() -> Dataset:
    return Dataset(
        name="extract_evidence",
        cases=CASES,
        evaluators=[
            ExtractsRelevantQuotes(),
            IdentifiesKeyFindings(),
            IdentifiesLimitations(),
            LLMJudge(
                rubric=(
                    "The agent should extract the most important factual content while handling "
                    "complexity: separating signal from noise in press releases, capturing "
                    "contradictory findings within a single source, noting embedded methodological "
                    "weaknesses, and distinguishing confidence levels in database entries. "
                    "Limitations should be specific to the source, not generic caveats."
                ),
                include_input=True,
            ),
        ],
    )


async def run_eval(max_concurrency: int = 3):
    ds = build_dataset()
    return await ds.evaluate(extract_evidence_task, max_concurrency=max_concurrency)
