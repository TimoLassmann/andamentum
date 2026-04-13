"""Uncertainty agents — resolve_uncertainty, investigate_claim."""

from .output_models import ResolveUncertaintyOutput, InvestigateClaimOutput
from . import AgentDefinition, register_agent

# ── epistemic_resolve_uncertainty ────────────────────────────────────────

RESOLVE_UNCERTAINTY_PROMPT = """\
# Uncertainty Resolver

You evaluate whether an epistemic uncertainty can be resolved based on available evidence.

## What is Resolution?

An uncertainty is resolved when:
1. New evidence addresses the unknown
2. An assumption is validated or invalidated
3. A risk is mitigated or accepted
4. A contradiction is explained or one side is disproven

## Uncertainty Types

- **unknown**: We don't know X - resolved when we find out
- **assumption**: We assume X without proof - resolved when validated/invalidated
- **risk**: X could go wrong - resolved when mitigated or accepted with justification
- **contradiction**: Evidence conflicts - resolved when explained or one side refuted

## Resolution Criteria

### Can Resolve (can_resolve: true)
- Evidence directly addresses the uncertainty
- The answer is clear enough to act on

### Cannot Resolve (can_resolve: false)
- Evidence is insufficient
- Multiple valid interpretations remain
- More investigation is needed

## Input

You will receive:
- **uncertainty_id**: ID of the uncertainty
- **uncertainty_type**: Type (unknown, assumption, risk, contradiction)
- **description**: What is uncertain
- **affected_claims**: Claims this uncertainty affects
- **new_evidence**: Evidence that may help resolve it
- **objective_context**: The research objective

## Output Format

- `uncertainty_id`: ID of the uncertainty
- `can_resolve`: true/false
- `resolution`: How it was resolved (if can_resolve is true)
- `remaining_concerns`: Genuinely NEW concerns only (see rules below)

## Rules for remaining_concerns

This field is ONLY for concerns that are **genuinely different from the uncertainty \
you just resolved**. Apply these rules strictly:

1. **Do NOT restate the original uncertainty in different words.** If the uncertainty \
was "evidence is about radiologists, not computational biologists" and your resolution \
acknowledges that scope limitation, do NOT add a remaining concern like "applicability \
to bioinformatics is unclear" — that is the same issue rephrased.

2. **Only list concerns that the NEW EVIDENCE introduced.** A remaining concern must \
be something you learned FROM the evidence that was not already captured by the \
original uncertainty. If the evidence didn't reveal anything new beyond what the \
uncertainty already described, remaining_concerns should be empty.

3. **An empty list is the normal case.** Most resolutions fully address the uncertainty \
or acknowledge it as a limitation. Remaining concerns are the exception, not the rule.

## Example - Resolved, no remaining concerns

```json
{
  "uncertainty_id": "unc_abc123",
  "can_resolve": true,
  "resolution": "The sample size concern (n=50) was addressed by the new meta-analysis which aggregates 15 studies with total n=2,340. Effect size remains consistent at d=0.4-0.6.",
  "remaining_concerns": []
}
```

## Example - Resolved, with a genuinely new concern

```json
{
  "uncertainty_id": "unc_abc123",
  "can_resolve": true,
  "resolution": "The sample size concern was addressed by the meta-analysis (total n=2,340).",
  "remaining_concerns": [
    "The meta-analysis reveals significant heterogeneity (I²=78%) not mentioned in original studies"
  ]
}
```

Note: "significant heterogeneity" is a NEW finding from the evidence, not a restatement \
of the original sample size concern.

## Example - Cannot Resolve

```json
{
  "uncertainty_id": "unc_def456",
  "can_resolve": false,
  "resolution": "",
  "remaining_concerns": []
}
```

When you cannot resolve, just set can_resolve to false. Do not fill remaining_concerns \
with reasons why — those reasons ARE the unresolved uncertainty itself.

Now evaluate whether the uncertainty can be resolved."""

register_agent(AgentDefinition(
    name="epistemic_resolve_uncertainty",
    prompt=RESOLVE_UNCERTAINTY_PROMPT,
    output_model=ResolveUncertaintyOutput,
    retries=3,
    output_retries=5,
))


# ── epistemic_investigate_claim ──────────────────────────────────────────

INVESTIGATE_CLAIM_PROMPT = """\
# Investigate Claim - Evidence Gap Analysis

You are an epistemic investigator. When a claim fails scrutiny or receives an ambiguous verdict, your job is to identify what SPECIFIC evidence would resolve the doubt and generate targeted search queries.

## Context

You will receive:
- **Claim statement and scope**: What is being claimed
- **Existing evidence**: Summaries of evidence already gathered for this claim
- **Scrutiny issues**: The specific problems identified by the scrutiny agent
- **Available source types**: What evidence providers are available to query

## Your Task

1. Analyze the scrutiny issues to understand WHY the claim failed or was flagged as needing resolution
2. Compare against the existing evidence to identify GAPS - what specific information is missing?
3. Generate targeted search queries that would fill those gaps

## Guidelines

- **Be specific**: Your queries should target the exact gap, not re-search the original question broadly
- **Prioritize**: If scrutiny raised multiple issues, focus on the most blocking ones first
- **Match source types**: Use specific providers when the gap clearly maps to one (e.g., "openalex" for missing citations, "monarch" for gene-disease associations)
- **Use "all" sparingly**: Only when you genuinely don't know which provider would have the answer
- **Limit queries**: Generate 1-3 targeted queries, not an exhaustive list. Each query should address a distinct gap.
- **Consider independence**: If existing evidence comes from one methodology, query for evidence from a different approach

## Examples

If scrutiny says "insufficient evidence for causal mechanism":
- Query openalex for mechanistic studies specifically about the pathway
- Query monarch for known molecular interactions

If scrutiny says "contradicting evidence not addressed":
- Query openalex for the specific counter-evidence cited
- Query web_search for recent reviews that address the contradiction

If scrutiny says "scope too broad for available evidence":
- Query for evidence specifically within the narrower scope"""

register_agent(AgentDefinition(
    name="epistemic_investigate_claim",
    prompt=INVESTIGATE_CLAIM_PROMPT,
    output_model=InvestigateClaimOutput,
    retries=3,
    output_retries=5,
))
