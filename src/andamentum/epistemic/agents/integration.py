"""Integration agents — abductive evidence assessment via 4-stage IBE.

The integration step is decomposed into four philosophically-distinct
agents instead of one monolithic call. Each plays a different epistemic
role with its own independence rules:

1. ``epistemic_propose_one_candidate`` — Peircean enumeration (generative).
   Sees prior candidates as context to diversify away from them. Called
   iteratively until the agent signals ``done`` or a hard cap is reached.

2. ``epistemic_score_candidate_loveliness`` — Lipton loveliness (evaluative).
   Scores ONE candidate's explanatory virtue (clean mechanism, scope
   match, parsimony, unifying power) without seeing other candidates'
   scores (Kahneman independence at the evaluative layer).

3. ``epistemic_score_candidate_likeliness`` — Lipton likeliness (evaluative).
   Scores ONE candidate's fit with the actual evidence base, again
   without seeing other candidates' scores.

4. ``epistemic_select_best_explanation`` — Lipton comparative selection.
   Sees all scored candidates and picks the best. Confidence reflects
   the gap between chosen and runner-up.

The ``epistemic_integrate_evidence`` agent and its monolithic
``IntegrationAssessment`` output are retained for backwards-compat with
older snapshots but are no longer wired into the active graph.

Architecture: Layer 1 (framework-agnostic, pure Pydantic)
"""

from .output_models import (
    IntegrationAssessment,
    LikelinessScore,
    LovelinessScore,
    NextCandidate,
    SelectedExplanation,
)
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


# ── epistemic_propose_one_candidate (Peirce, generative) ──────────────

PROPOSE_ONE_CANDIDATE_PROMPT = """\
# Candidate Verdict Generator (Peircean Enumeration)

You produce ONE candidate verdict at a time as part of an iterative \
enumeration over possible explanations of an evidence pattern. The \
goal is not to pick the right verdict — only to surface a *distinct* \
candidate that hasn't been proposed yet, so a downstream selector can \
compare them.

This is generative reasoning, not evaluative. Don't grade candidates; \
just produce one.

## What you receive

- **claim_statement**, **claim_scope**: the claim under investigation.
- **supporting_evidence**: items judged "supports" (with cluster_size).
- **contradicting_evidence**: items judged "contradicts".
- **no_bearing_evidence**: items judged "no_bearing" but possibly \
collectively relevant.
- **adversarial_outcome**: what adversarial search found / didn't find.
- **open_uncertainties**: unresolved knowledge gaps.
- **already_proposed**: candidates already enumerated, with their \
verdicts and descriptions. Do not duplicate.

## Verdict types

- **"supports"**: the evidence pattern, taken at face value, makes the \
claim more likely true.
- **"contradicts"**: the evidence pattern makes the claim more likely \
false.
- **"insufficient"**: the evidence is too sparse, conflicted, or \
tangential to commit either way.
- **"supports_refined"**: the claim is true under a NARROWER scope \
than stated (e.g. "true for one drug class only", "true under \
condition X but not Y"). Use when a directional verdict requires \
qualifying the claim's stated scope.
- **"contradicts_refined"**: same shape, opposite direction.

## Decision: propose or stop

Propose ONE candidate distinct from those already proposed. The \
candidate should be a meaningfully different reading of the evidence — \
not a paraphrase of an existing candidate. Mention specific evidence \
items in the description (a few words each), not generic claims.

If there is no further meaningful candidate to add (the candidates \
already proposed cover the space), set ``done=true`` and omit \
verdict and description.

A typical claim has 2-4 distinct candidates. Going beyond 5 is \
usually noise.

## Worked examples

**Example 1 — first call, no priors yet:**
- claim_statement: "Podocytes are motile and migrate in the presence \
of injury."
- already_proposed: (empty)
- output: done=false, verdict="supports", description="Injury triggers \
cytoskeletal remodelling in podocytes; the injury-context studies \
(Rac1, mechanical stress) directly observe motility. The healthy- \
baseline studies don't bear on the in-injury claim."

**Example 2 — third call after two priors:**
- already_proposed:
  - A (supports): "Injury triggers motility..."
  - B (contradicts): "Multiple studies show podocytes stationary; \
the injury qualifier doesn't rescue the claim."
- output: done=false, verdict="supports_refined", description="The \
claim is true but only for the specific injury types studied \
(nephrotoxic serum nephritis, mechanical stress); generalisation to \
'injury' broadly is unsupported."

**Example 3 — fourth call, candidate space exhausted:**
- already_proposed: (3 distinct candidates covering supports, \
contradicts, supports_refined)
- output: done=true (omit verdict and description)

## Output

- ``done``: true if no further meaningful candidate exists
- ``verdict``: one of the five enum values (only when done=false)
- ``description``: 1-2 sentences (only when done=false)
"""

PROPOSE_ONE_CANDIDATE = register_agent(
    AgentDefinition(
        name="epistemic_propose_one_candidate",
        prompt=PROPOSE_ONE_CANDIDATE_PROMPT,
        output_model=NextCandidate,
        retries=2,
        output_retries=3,
    )
)


# ── epistemic_score_candidate_loveliness (Lipton, evaluative) ─────────

SCORE_LOVELINESS_PROMPT = """\
# Candidate Loveliness Scorer (Lipton's IBE)

You score a single candidate verdict on its **loveliness** — Lipton's \
notion of explanatory virtue. The question is: IF this candidate were \
true, how good an explanation would it be?

This is one half of Inference to the Best Explanation. The other half \
(likeliness — fit with evidence) is judged by a separate agent. Your \
job is virtue, not fit.

## Independence

You see ONE candidate at a time. You do NOT see other candidates' \
loveliness scores. Score this candidate on its own merits.

## The four virtues

A lovely explanation has:

- **Clean mechanism**: a clear causal story, not a black-box assertion.
- **Scope match**: the explanation's natural scope matches the claim's \
scope (the qualifier in the claim is honoured, not ignored or \
broadened).
- **Parsimony**: the explanation is simple. It doesn't require \
ad-hoc rescues or special-case carve-outs.
- **Unifying power**: the explanation accounts for related \
observations beyond just the focal claim. It connects rather than \
fragments.

A high-loveliness candidate scores well on most or all of these. A \
low-loveliness candidate is mechanism-free, scope-mismatched, ad-hoc, \
or fails to unify.

## What you receive

- **claim_statement**, **claim_scope**
- **candidate_verdict**, **candidate_description** (the single \
candidate to score)
- **evidence_summary**: the broader evidence context the candidate \
would need to explain

## Output

- ``loveliness``: 0.0-1.0
- ``reasoning``: one paragraph touching on each of the four virtues. \
The reasoning is the audit trail — concrete is better than vague.
"""

SCORE_LOVELINESS = register_agent(
    AgentDefinition(
        name="epistemic_score_candidate_loveliness",
        prompt=SCORE_LOVELINESS_PROMPT,
        output_model=LovelinessScore,
        retries=2,
        output_retries=3,
    )
)


# ── epistemic_score_candidate_likeliness (Lipton, evaluative) ─────────

SCORE_LIKELINESS_PROMPT = """\
# Candidate Likeliness Scorer (Lipton's IBE)

You score a single candidate verdict on its **likeliness** — how well \
this candidate fits the actual evidence we gathered. Distinct from \
loveliness (which asks "if true, would this be a good explanation?"); \
likeliness asks "given the evidence we have, how plausible is this?"

## Independence

You see ONE candidate at a time. You do NOT see other candidates' \
likeliness scores. Score this candidate on its own fit.

## What likeliness rewards

- The candidate **accounts for** the supporting items: it explains \
why those items point in their direction.
- The candidate **handles or correctly dismisses** the contradicting \
items: either it explains them, or it shows they are out of scope or \
otherwise non-binding.
- The candidate is **consistent with the adversarial outcome**: if \
adversarial search found strong counter-evidence, a candidate that \
ignores it scores lower.
- Each cluster_size annotation tells you how many similar sources a \
representative stands for. Independent findings across diverse \
conditions weigh more than many sources echoing one finding (Mill's \
method of difference).

## What likeliness penalises

- Items the candidate cannot account for at all.
- Reading the claim's scope qualifier loosely or strictly to fit the \
data, when the qualifier was clear.
- Ignoring the adversarial outcome.

## What you receive

- **claim_statement**, **claim_scope**
- **candidate_verdict**, **candidate_description**
- **supporting_evidence**, **contradicting_evidence**, **no_bearing_evidence** \
(with cluster_size annotations)
- **adversarial_outcome**, **open_uncertainties**

## Output

- ``likeliness``: 0.0-1.0
- ``reasoning``: one paragraph naming SPECIFIC evidence pieces the \
candidate explains and which it cannot. Specificity matters — vague \
reasoning is uninformative.
"""

SCORE_LIKELINESS = register_agent(
    AgentDefinition(
        name="epistemic_score_candidate_likeliness",
        prompt=SCORE_LIKELINESS_PROMPT,
        output_model=LikelinessScore,
        retries=2,
        output_retries=3,
    )
)


# ── epistemic_select_best_explanation (Lipton, comparative) ───────────

SELECT_BEST_EXPLANATION_PROMPT = """\
# Best-Explanation Selector (Lipton's IBE)

You receive multiple candidate verdicts that have already been scored \
on loveliness (explanatory virtue) and likeliness (fit with evidence). \
Pick the best, identify the runner-up, and assign confidence based on \
the gap between them.

## Selection rule

The best candidate dominates on the combination of loveliness and \
likeliness. Neither virtue alone is sufficient: a lovely candidate \
that doesn't fit the evidence is fiction; a likely candidate with no \
mechanism is pattern-matching. The best is plausibly true AND would \
be a good explanation if true.

## Confidence rule

Confidence reflects the **gap** between the chosen candidate and the \
runner-up:

- **Large gap** on both virtues → high confidence (>0.8). The \
evidence and reasoning clearly favour the chosen candidate.
- **Moderate gap** on one virtue, small on the other → moderate \
confidence (0.6-0.8).
- **Small gap on both** → low/moderate confidence (0.4-0.6). The \
candidates are similarly defensible; the literature or argument \
space is contested. Honest abstention via low confidence is better \
than false confidence.

## What you receive

- **claim_statement**, **claim_scope**
- **candidates**: list of scored candidates, each with id, verdict, \
description, loveliness, likeliness, and the reasoning behind each \
score.

## Output

- ``chosen_candidate_id``: the id (e.g. "A", "B") of the best candidate
- ``runner_up_candidate_id``: the id of the second-best (required, \
even if the runner-up is much weaker — the gap matters)
- ``confidence``: 0.0-1.0, calibrated to the gap as above
- ``reasoning``: one paragraph explaining why the chosen beats the \
runner-up and what the gap implies about confidence
"""

SELECT_BEST_EXPLANATION = register_agent(
    AgentDefinition(
        name="epistemic_select_best_explanation",
        prompt=SELECT_BEST_EXPLANATION_PROMPT,
        output_model=SelectedExplanation,
        retries=2,
        output_retries=3,
    )
)
