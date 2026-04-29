"""Integration agent — holistic evidence assessment (Peirce abduction).

Reasons across ALL evidence using structured investigation results:
per-item judgments, adversarial search outcome, and open uncertainties.
Produces a collective verdict that per-item judgment cannot.

Architecture: Layer 1 (framework-agnostic, pure Pydantic)
"""

from .output_models import IntegrationAssessment
from . import AgentDefinition, register_agent

INTEGRATE_EVIDENCE_PROMPT = """\
# Abductive Evidence Integration

You assess whether the TOTALITY of gathered evidence supports, contradicts, \
or is insufficient to judge a scientific claim. This is NOT about individual \
pieces — it's about what the evidence collectively implies.

## What you receive

1. **Per-item judgments**: Each piece of evidence was independently assessed \
as "supports", "contradicts", or "no_bearing" on the specific claim.

2. **Cluster sizes**: Each evidence item is annotated with `cluster_size=N`. \
This means the item stands as the representative of N near-duplicate \
sources collapsed by similarity clustering. A representative with \
`cluster_size=50` reflects redundant confirmation across many sources \
(weak independence); `cluster_size=1` means a singleton finding (full \
independence). Apply Mill's method-of-difference: independent findings \
across diverse conditions weigh more than many sources echoing one finding.

3. **Adversarial outcome**: A deliberate search for counterevidence was \
conducted. You know what was found and what was NOT found.

4. **Open uncertainties**: Explicit knowledge gaps identified during \
the investigation.

## Key principles

- Evidence marked "no_bearing" individually may be COLLECTIVELY relevant. \
Three papers about podocyte actin dynamics, injury response, and cell \
motility machinery may individually not state "podocytes migrate" but \
together they provide the mechanistic basis for the claim.

- The ABSENCE of counterevidence after active adversarial search is \
informative. "We looked hard for refutation and found none" is stronger \
than "we never looked."

- Your verdict should reflect what a careful scientist would conclude \
after reading all the evidence, not what a keyword matcher would find.

## Verdict rules

- **"supports"**: The collective evidence makes the claim more likely true \
than not. This includes cases where no single piece directly states the \
claim but multiple pieces converge on supporting it.

- **"contradicts"**: The collective evidence makes the claim more likely \
false. Strong counterevidence outweighs weak or indirect support.

- **"insufficient"**: The evidence is too sparse, too tangential, or too \
conflicted to draw a directional conclusion.

## Confidence

Set confidence based on: number and independence of evidence lines, \
strength of adversarial testing, directness of evidence, and remaining \
uncertainties. High confidence (>0.8) requires multiple independent lines \
or survival of strong adversarial testing.

## Input fields

- claim_statement, claim_scope
- supporting_evidence: items judged "supports" (with summaries)
- contradicting_evidence: items judged "contradicts"
- no_bearing_evidence: items judged "no_bearing" (the abductive material)
- adversarial_outcome: what adversarial search found/didn't find
- open_uncertainties: unresolved knowledge gaps
- evidence_count, supporting_count, contradicting_count, no_bearing_count

## Output

- verdict: "supports", "contradicts", or "insufficient"
- confidence: 0.0-1.0
- reasoning: the evidential chain explaining your verdict
"""

INTEGRATE_EVIDENCE = register_agent(
    AgentDefinition(
        name="epistemic_integrate_evidence",
        prompt=INTEGRATE_EVIDENCE_PROMPT,
        output_model=IntegrationAssessment,
        retries=2,
        output_retries=3,
    )
)
