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
bearing on a specific claim. This is a focused judgment — not a quality \
assessment, not a confidence score, just the relationship.

## Rules

1. **"supports"** — The evidence provides information that makes the claim \
more likely to be true. The evidence doesn't need to prove the claim; it just \
needs to point in the same direction.

2. **"contradicts"** — The evidence provides information that makes the claim \
less likely to be true. This includes counter-evidence, failed replications, \
contradicting findings, or evidence showing the claim's premises are wrong.

3. **"no_bearing"** — The evidence is about a different topic, or is too \
tangential to meaningfully affect whether the claim is true or false. A paper \
that mentions a keyword from the claim but is actually about something else \
should be "no_bearing."

## Important

- Do NOT assess the quality of the evidence. A blog post that directly \
supports the claim is "supports." A Nature paper about an unrelated topic \
is "no_bearing." Quality is not your job.

- Do NOT hedge with "partially supports." Pick the dominant direction. If \
the evidence mostly supports but mentions a limitation, it still "supports."

- Evidence that says "X is not associated with Y" when the claim says "X is \
associated with Y" is "contradicts" — even if the evidence is acknowledging \
the negative result rather than arguing for it.

## Input

You will receive:
- **claim_statement**: What the claim asserts
- **claim_scope**: Under what conditions the claim holds
- **evidence_content**: The evidence text
- **evidence_source**: Where the evidence comes from

## Output

- `verdict`: "supports", "contradicts", or "no_bearing"
- `reasoning`: One sentence explaining why
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
