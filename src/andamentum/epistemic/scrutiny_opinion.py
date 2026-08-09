"""Second opinion on a BORDERLINE scrutiny reading (llm_judge, FAST path).

``scrutiny_weight.compute_evidence_weight`` maps quality- and
corroboration-weighted directional mass onto ``strong | moderate | weak |
conflicting`` through three cutoffs. It is pure, deterministic and free —
and that is exactly why a reading sitting a hair from one of those cutoffs
is fragile in a way the number itself cannot report: the label is then an
artefact of where the cutoff sits, not of the evidence.

This module is the second opinion on that case. When
``scrutiny_weight.is_borderline`` says the label would flip under a ±20 %
relative perturbation, one independent judge is asked the claim-level
question the mass cannot answer — *does this body of evidence establish the
claim as stated?* — and the answer is recorded as a ``JudgeOpinion`` beside
the verdict.

**IT NEVER CHANGES THE VERDICT.** ``compute_evidence_weight``,
``EvidenceWeight``, ``_verdict_from_weight``, the blocking-issue downgrade
and ``Claim.scrutiny_doubt`` are all untouched. That is not timidity: it is
what preserves ``ConcludeAtCycleCapOperation``'s documented contract that
the cap "does not introduce a second, divergent reading" — the cap re-runs
the same pure function and therefore reads the identical label. A judge that
could move the label would make scrutiny and the cap disagree about the same
claim. Instead, a confident disagreement or a doubtful agreement feeds the
EXISTING Gate 0 ``Demand`` and buys the objective one bounded extra round of
inquiry (P7). Suspicion-raiser, never a verdict.

WHY meets/partial/fails IS THE RIGHT SCHEMA HERE, AND STILL WRONG FOR
``judge_evidence``. ``quality_doubt.py`` refuses to convert the primary
evidence-claim judge because that judgment is a DIRECTION axis
(``supports | contradicts | no_bearing`` plus ``in_scope``), which three
quality labels cannot express. That argument is untouched and still holds.
The question asked here is natively a quality axis — "is this enough,
on-scope evidence to establish the claim" — which is precisely what
llm_judge's schema says.

THE DETERMINISTIC LABEL IS WITHHELD FROM THE JUDGE. Telling it that the
system already read the evidence as "moderate" would anchor the very
opinion that exists to check that reading. The label appears only in the
stored ``JudgeOpinion.reference``, where the dissent comparison is made
afterwards, in code.

FAST, NOT PANEL — ``model`` is a bare ``str``. ONE CRITERION — ``judge_score``
fires one model call per criterion, so the whole second opinion is exactly
one call. Both choices copy ``quality_doubt.py`` verbatim, for its stated
reasons.

This module mutates nothing and raises rather than degrading.

Architecture: Layer 1 (framework-agnostic; async driver + pure assembly).
"""

from __future__ import annotations

from andamentum.llm_judge import Criterion, judge_score
from andamentum.llm_judge.signals import normalize_three

from .entities.judge_opinion import JudgeOpinion

# Exactly one criterion — see the module docstring for why not four.
SCRUTINY_CRITERIA: list[Criterion] = [
    Criterion(
        name="evidence_establishes_the_claim",
        description=(
            "The evidence, taken as a body, establishes the claim as stated "
            "— rather than being off-scope, too thin to establish it, or "
            "pointing both ways."
        ),
    )
]

#: The weight labels ``ScrutiniseClaimOperation._verdict_from_weight`` maps
#: to a passing verdict. Named here so the dissent rule reads the same
#: standard the verdict does rather than re-listing it.
_PASSING_LABELS = frozenset({"strong", "moderate"})


def reading_dissents(*, weight_label: str, overall: str) -> bool:
    """Does ``overall`` contradict the deterministic reading ``weight_label``?

    Pure, and deliberately asymmetric-free: a "pass" reading is dissented
    from by a confident ``fails``, a "not pass" reading by a confident
    ``meets``. ``partial`` NEVER dissents — a hedge is not a contradiction,
    and treating it as one would fire Gate 0 on every merely-cautious judge.

    Both parameters are keyword-only: they are same-typed strings drawn from
    two different closed vocabularies, and transposing them would silently
    compare a weight label against the ``meets`` set.
    """
    reading_is_pass = weight_label in _PASSING_LABELS
    if reading_is_pass:
        return overall == "fails"
    return overall == "meets"


async def second_opinion_on_reading(
    *,
    evidence_body: str,
    claim_statement: str,
    claim_scope: str,
    weight_label: str,
    model: str,
) -> JudgeOpinion:
    """Ask one judge whether the evidence body establishes the claim.

    Every parameter is keyword-only: ``evidence_body`` / ``claim_statement``
    / ``claim_scope`` / ``weight_label`` / ``model`` are five same-typed
    strings, and transposing any pair would judge the wrong text against the
    wrong context — invisibly to pyright and to every test that does not
    exercise that exact path.

    Args:
        evidence_body: The already-formatted representative evidence
            summaries scrutiny is reasoning over, joined. Reused from the
            caller — this function never re-selects or re-embeds.
        claim_statement: The claim the evidence was gathered for.
        claim_scope: The claim's scope, so "on-scope" is asked about the
            right conditions.
        weight_label: The deterministic reading this opinion is taken
            AGAINST. It is recorded on the result and used for the dissent
            comparison; it is deliberately NOT shown to the judge.
        model: A single pydantic-ai model id. A ``str`` (not a sequence) is
            required: it selects llm_judge's FAST path. A one-element LIST
            would silently select PANEL at sampling temperature — ``_is_panel``
            keys on TYPE, not length.

    Returns:
        The judge's opinion, with derived doubt / confidence / needs_review
        and the stored ``dissents`` comparison.

    Raises:
        Whatever ``judge_score`` raises. This function never swallows a
        failure into a default answer — a silently-defaulted second opinion
        would be worse than not asking.
    """
    context = (
        f"Claim under scrutiny: {claim_statement or '(none)'}\n"
        f"Scope (the conditions under which the claim is asserted): "
        f"{claim_scope or 'General'}\n"
        "Below is the evidence gathered for this claim. Judge whether, taken "
        "as a body, it establishes the claim as stated."
    )
    result = await judge_score(
        evidence_body,
        criteria=SCRUTINY_CRITERIA,
        context=context,
        model=model,
    )
    row = result.per_criterion[0]
    return JudgeOpinion(
        subject=(
            f"Does the gathered evidence establish the claim "
            f"'{claim_statement[:120]}' as stated?"
        ),
        reference=weight_label,
        distribution=normalize_three(row.meets, row.partial, row.fails),
        overall=result.overall,
        expected_score=result.expected_score,
        summary=row.reasoning,
        dissents=reading_dissents(weight_label=weight_label, overall=result.overall),
    )


__all__ = [
    "SCRUTINY_CRITERIA",
    "reading_dissents",
    "second_opinion_on_reading",
]
