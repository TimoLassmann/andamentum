"""Second opinion on a near-zero evidence quality score (llm_judge, FAST path).

``ExtractEvidenceOperation._score_evidence`` Path 2 blends four agent-produced
floats into one quality score and, below ``0.10``, invalidates the evidence
outright (a Truth-Maintenance-System destroy). That is a **categorical cliff
driven by a single sampled point value**: one unlucky draw on four
free-floating numbers and a piece of real evidence is erased, with nothing
downstream able to tell erasure from absence.

This module is the gate on that cliff. It asks ``llm_judge.judge_score`` — the
system's uncertainty-carrying judge — whether the source text is genuinely
substanceless, and returns the verdict together with the DERIVED doubt and
``needs_review`` signals. The caller destroys the evidence only when that
second opinion confidently agrees. Otherwise the evidence survives, keeping
its low ``quality_score``, so it is still heavily down-weighted in
``scrutiny_weight._item_masses`` and the counting posterior. Attenuation, not
destruction — the project's "process flags filter inquiry, not output"
doctrine, applied at the evidence layer.

FAST, NOT PANEL. ``model`` is a bare ``str``, so ``judge_score`` takes its
single-judge, temperature-0 path: one call, reproducible. A panel would be K
sequential calls on a tail path that fires once per near-zero item, and its
contract is an *agreement* poll rather than a *reproducible gate* — the wrong
shape for a destroy/retain decision that must give the same answer on a re-run.

ONE CRITERION, deliberately. ``judge_score`` fires one model call per
criterion and rolls them up equal-weight. Mapping the four quality axes to
four criteria would be a 4x cost blowup here AND would silently replace the
deliberate 0.35 / 0.25 / 0.25 / 0.15 blend the agent scorer applies, with an
equal-weight mean nobody chose.

WHERE llm_judge IS USED IN epistemic — THREE SITES, ONE SHAPE. This module
was the first; there are now three, and all three are the same shape (FAST
path, exactly one ``Criterion``, gated so they fire only on judgments that
are already borderline or entropy-blind, and suspend-only — none of them can
change a verdict):

1. **this module** — the near-zero-quality TMS destroy gate (per evidence
   item, tail path only);
2. ``direction_opinion`` — an independent CONFIRMATION of a one-hot,
   directional evidence judgment, during scrutiny (bounded per pass and per
   claim);
3. ``scrutiny_opinion`` — an independent read on a BORDERLINE deterministic
   evidence weight, per claim.

CONVERSION vs. CONFIRMATION — the distinction that keeps the argument below
true. ``judge_evidence`` is STILL NOT CONVERTED, for exactly the original
reasons: it forces a four-step scope analysis and enforces ``in_scope`` ↔
argmax over ``{supports, contradicts, no_bearing}`` via a retry-triggering
cross-field validator, none of which the generic three-label score schema can
express. Replacing it would be a regression dressed as an upgrade. What site
(2) adds is a layer ABOVE it: the primary judge has already committed to a
direction, and "does this text genuinely support [/contradict] the claim" IS
a meets/partial/fails question. The scope analysis is not re-done and not
lost — it is the proposition being confirmed. Site (3) is the same move at
the claim level, on a question ("is this enough, on-scope evidence to
establish the claim") that is natively a quality axis.

What all three share, and what a fourth site would have to share, is that the
judgment being put to llm_judge has NO DOMAIN INVARIANT TO LOSE.

This module mutates nothing and raises rather than degrading: a judge failure
propagates to ``_execute_op``, which records it as a loud quarantine.

Architecture: Layer 1 (framework-agnostic; async driver + pure assembly).
"""

from __future__ import annotations

from dataclasses import dataclass

from andamentum.llm_judge import Criterion, judge_score
from andamentum.llm_judge.signals import normalize_three

# Exactly one criterion — see the module docstring for why not four.
QUALITY_CRITERIA: list[Criterion] = [
    Criterion(
        name="substantive_content",
        description=(
            "The text contains substantive, checkable content that could bear "
            "on a scientific claim — rather than being empty, boilerplate, an "
            "error or paywall page, or navigation chrome."
        ),
    )
]


@dataclass(frozen=True)
class QualitySecondOpinion:
    """One judge's read on whether a near-zero-scored source is really empty.

    ``overall`` is the LABEL to decide on; ``expected_score`` is the continuous
    companion for ranking (llm_judge's documented rule: label for decisions,
    scalar for ranking). ``doubt`` and ``needs_review`` are DERIVED by
    ``llm_judge``, never asked of the model — ``needs_review`` already folds in
    ``doubt >= NEEDS_REVIEW_DOUBT_THRESHOLD``, which is why this module
    introduces no numeric threshold of its own.
    """

    distribution: list[float]
    overall: str
    expected_score: float
    doubt: float
    needs_review: bool
    summary: str


async def confirm_low_quality(
    *,
    content: str,
    source_ref: str,
    claim_statement: str,
    agent_score: float,
    model: str,
) -> QualitySecondOpinion:
    """Ask one judge whether ``content`` really is substanceless.

    Every parameter is keyword-only: ``content`` / ``source_ref`` /
    ``claim_statement`` / ``model`` are four same-typed strings, and
    transposing any pair would silently judge the wrong text against the
    wrong context.

    Args:
        content: The extracted source text the agent scored near zero.
        source_ref: The source identifier, surfaced to the judge as context.
        claim_statement: The claim this evidence was gathered for, so
            "could bear on a scientific claim" is asked about the right claim.
        agent_score: The agent's blended quality score, included in the
            context purely as provenance — the judge is NOT told to agree
            with it, only what prompted the second look.
        model: A single pydantic-ai model id. A ``str`` (not a sequence) is
            required: it selects llm_judge's FAST path.

    Returns:
        The judge's verdict, with derived doubt / needs_review.

    Raises:
        Whatever ``judge_score`` raises. This function never swallows a
        failure into a default answer — a silently-defaulted second opinion
        would be worse than the point value it replaces.
    """
    context = (
        f"This text was retrieved as evidence for the claim: {claim_statement or '(none)'}\n"
        f"Source: {source_ref}\n"
        f"An automated scorer rated its quality {agent_score:.3f} (near zero), "
        f"which would normally discard it. Judge the text on its own terms."
    )
    result = await judge_score(
        content,
        criteria=QUALITY_CRITERIA,
        context=context,
        model=model,
    )
    row = result.per_criterion[0]
    return QualitySecondOpinion(
        distribution=normalize_three(row.meets, row.partial, row.fails),
        overall=result.overall,
        expected_score=result.expected_score,
        doubt=result.doubt,
        needs_review=result.needs_review,
        summary=row.reasoning,
    )


__all__ = [
    "QUALITY_CRITERIA",
    "QualitySecondOpinion",
    "confirm_low_quality",
]
