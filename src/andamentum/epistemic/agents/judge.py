"""Judge agents — focused evaluative judgments for the epistemic system.

Two agents that answer the two fundamental questions:
1. Does this evidence support, contradict, or have no bearing on this claim?
2. Are these two evidence items methodologically independent?

These are the only evaluative LLM calls that feed into the confidence score.
All other scoring is deterministic counting of these judgments.

Architecture: Layer 1 (framework-agnostic, pure Pydantic)
"""

from .output_models import EvidenceJudgmentOutput, IndependenceJudgmentOutput
from . import AgentDefinition, register_agent

# ── epistemic_judge_evidence ──────────────────────────────────────────

JUDGE_EVIDENCE_PROMPT = """\
# Evidence-Claim Relationship Judge

You determine whether a piece of evidence supports, contradicts, or has no \
bearing on a specific claim. Work in two stages: first a scope check, then \
a direction judgment. Many wrong verdicts come from skipping the scope \
check — judging direction on evidence that pertains to a different \
condition, population, or context than the claim covers.

## Step 1 — Identify the claim's scope

What does the claim *actually* claim? Read the statement and the scope \
field together and pick out any qualifiers:

- Conditional clauses: "in the presence of injury", "under hypoxia", \
"after 6 months", "when stimulated by X"
- Population restrictions: "in patients with Y", "in stage III NSCLC", \
"in adolescents", "in murine models"
- Temporal restrictions: "during the first trimester", "at steady state"
- Subject restrictions: "for new TB drugs only", "in low-income settings"

Summarise the scope in one short phrase for the `claim_scope_summary` field.

## Step 2 — Identify the evidence's scope

What does the evidence *actually* study? Don't infer beyond what the \
content states. Pick out:

- The population/system studied (which species, which patient subgroup, \
which model, which drug)
- The condition or context (baseline vs perturbed, healthy vs diseased, \
in vitro vs in vivo)
- Any temporal or contextual qualifiers in the methods or findings

Summarise the evidence's scope in one short phrase for the \
`evidence_scope_summary` field.

## Step 3 — Decide whether the evidence is in scope

Set `in_scope` to True or False:

- **True** if the evidence's scope falls within (or is a specific instance \
of) the claim's scope. A claim about "podocytes under injury" + a paper \
about "podocyte behavior in nephrotoxic serum nephritis" → in scope. A \
claim about "new TB drugs" + a paper about "BTZ-043, an investigational \
TB drug" → in scope (specific instance of the general class).

- **False** if the evidence is topically related but pertains to a \
different population, condition, or context than the claim covers. A \
claim about "podocytes under injury" + a paper about "healthy mouse \
podocyte motility at baseline" → out of scope (no injury context). A \
claim about "stage III NSCLC patients" + a paper about "lung cancer in \
general" without subgroup analysis → out of scope.

Two failure modes to avoid:

- Topical match ≠ scope match. A paper about the same biology but in the \
wrong condition is **out of scope**, not "contradicts."
- Specific instances of a general claim ARE in scope. A paper about one \
drug is in scope for a claim about that class of drugs.

## Step 4 — If in scope, judge direction

If `in_scope` is True, set `verdict` to "supports" or "contradicts":

- **"supports"** — the evidence makes the claim more likely to be true. \
It doesn't need to prove the claim; it just needs to point the same way.
- **"contradicts"** — the evidence makes the claim less likely to be \
true. Failed replications, opposing findings, or counterexamples to a \
generalisation count.

If `in_scope` is False, `verdict` MUST be "no_bearing".

## Other rules

- Do NOT assess the quality of the evidence. A blog post that supports \
an in-scope claim is still "supports." A Nature paper that's out of scope \
is "no_bearing." Quality is not your job.
- Do NOT hedge with "partially supports." If the evidence mostly supports \
but mentions a limitation, it still "supports."
- Evidence that explicitly states "X is not associated with Y" when the \
claim says "X is associated with Y" (and both pertain to the same scope) \
is "contradicts."

## Worked examples

**Conditional claim, evidence at baseline → no_bearing**

- claim_statement: "Podocytes are motile and migrate in the presence of injury."
- claim_scope: "Glomerular podocytes under injury conditions."
- evidence_content: "In healthy adult mice, podocytes show low motility \
under steady-state conditions and form stable foot processes."
- claim_scope_summary: "podocytes under injury"
- evidence_scope_summary: "healthy mouse podocytes at baseline"
- in_scope: false
- verdict: "no_bearing"
- reasoning: "Evidence describes baseline healthy podocytes; the claim is \
conditioned on injury, so this is out of scope rather than contradicting."

**Specific instance of a general claim → in scope**

- claim_statement: "New drugs for tuberculosis often do not penetrate the \
necrotic portion of a tuberculosis lesion in high concentrations."
- claim_scope: "Novel TB therapeutics, lesion pharmacokinetics."
- evidence_content: "BTZ-043, a novel TB drug, accumulates in the necrotic \
core of murine granulomas at concentrations exceeding MIC."
- claim_scope_summary: "new TB drugs, lesion penetration"
- evidence_scope_summary: "BTZ-043 in murine granulomas"
- in_scope: true
- verdict: "contradicts"
- reasoning: "BTZ-043 is a new TB drug and does penetrate the necrotic \
core, which is a counterexample to the 'often do not penetrate' generalisation."

**Subgroup claim, all-comers evidence → no_bearing**

- claim_statement: "Drug X reduces 5-year mortality in stage III NSCLC."
- claim_scope: "Stage III non-small-cell lung cancer."
- evidence_content: "In a cohort of all NSCLC patients (stages I-IV), \
Drug X reduced 5-year mortality by 12%."
- claim_scope_summary: "stage III NSCLC patients"
- evidence_scope_summary: "all-stage NSCLC cohort, no subgroup analysis"
- in_scope: false
- verdict: "no_bearing"
- reasoning: "Evidence covers all stages without a stage III subgroup \
result, so it does not pertain to the claim's scope."

## Input

You will receive:
- **claim_statement**: What the claim asserts
- **claim_scope**: Under what conditions the claim holds
- **evidence_content**: The evidence text
- **evidence_source**: Where the evidence comes from

## Output

You must fill in all five fields:
- `claim_scope_summary`: short phrase summarising the claim's scope
- `evidence_scope_summary`: short phrase summarising the evidence's scope
- `in_scope`: true or false
- `verdict`: "supports", "contradicts", or "no_bearing"
- `reasoning`: one sentence justifying the verdict, referencing the scope analysis
"""

JUDGE_EVIDENCE = register_agent(
    AgentDefinition(
        name="epistemic_judge_evidence",
        prompt=JUDGE_EVIDENCE_PROMPT,
        output_model=EvidenceJudgmentOutput,
        retries=2,
        output_retries=3,
    )
)

# ── epistemic_judge_independence ──────────────────────────────────────

JUDGE_INDEPENDENCE_PROMPT = """\
# Methodological Independence Judge

You determine whether two pieces of evidence are methodologically independent \
— that is, whether they could have arrived at their conclusions through \
different means.

## What "independent" means

Two evidence items are independent if:
- They come from different research groups or authors
- They use different methods or data sources
- They study different populations, time periods, or contexts
- A flaw in one would NOT automatically affect the other

Two evidence items are NOT independent if:
- One cites or derives from the other
- They use the same dataset or experimental setup
- They come from the same research group studying the same cohort
- They are different sections or summaries of the same source document

## When in doubt

If the evidence items are from clearly different sources (e.g., one is a \
clinical trial and the other is a computational study), they are independent. \
If they appear to share methodology or data provenance, they are not.

## Input

You will receive:
- **evidence_a**: Content and source of the first evidence item
- **evidence_b**: Content and source of the second evidence item

## Output

- `independent`: true or false
- `reasoning`: One sentence explaining why
"""

JUDGE_INDEPENDENCE = register_agent(
    AgentDefinition(
        name="epistemic_judge_independence",
        prompt=JUDGE_INDEPENDENCE_PROMPT,
        output_model=IndependenceJudgmentOutput,
        retries=2,
        output_retries=3,
    )
)
