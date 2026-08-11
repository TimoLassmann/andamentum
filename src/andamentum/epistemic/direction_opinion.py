"""Independent CONFIRMATION of a one-hot evidence direction (llm_judge, FAST).

THE HOLE THIS CLOSES. ``Claim.scrutiny_doubt`` — the number Gate 0 escalates
on — is the directional-mass-weighted mean of each judgment's normalized
entropy. ``thresholds.JUDGMENT_ONE_HOT_THRESHOLD`` records the measured fact
that small models emit a one-hot histogram 53–82 % of the time, and
``judgment_signal.distribution_is_one_hot`` exists precisely to mark those
distributions as uninformative. Put the two together: a claim whose evidence
judgments are ALL one-hot has ``scrutiny_doubt ≈ 0``, so Gate 0 is
structurally dead on the majority of the arm the paper measures. Re-reading
the same distribution cannot fix that — the distribution is degenerate. Only
a SECOND, INDEPENDENT elicitation can, and that is what ``llm_judge`` is.

CONFIRMATION, NOT CONVERSION. ``epistemic_judge_evidence`` is untouched —
same agent, same four-step scope-then-direction prompt, same
``in_scope`` ↔ argmax cross-field validator, same five call sites.
``quality_doubt.py``'s argument for not converting it (the direction axis
carries ``in_scope`` and the ``contradicts`` / ``no_bearing`` distinction,
which three quality labels cannot express) is correct and stands. What this
module adds sits ABOVE it: the primary judge has already committed to a
direction, and *"read on its own terms, does this text genuinely support
[/contradict] the claim?"* IS a meets/partial/fails question. The scope
analysis is not re-done and not lost — it is the proposition being confirmed.

WHERE IT FIRES, AND WHY NOWHERE ELSE. One site:
``ScrutiniseClaimOperation.execute``, over the already-materialised
``eligible_representative_evidence`` set, in a strictly sequential loop,
before the single ``compute_evidence_weight`` call. Not at the five primary
judging call sites, for three independently disqualifying reasons: they are
per-item and unbounded (13–88 items per claim); three of them judge inside an
``asyncio.gather`` and ``llm_judge`` does NOT share ``core.AgentRunner``'s
semaphore, so a call there would fire N concurrent local-model inferences;
and five sites is five places to drift.

THE SELECTOR IS THE INFORMATION ARGUMENT, NOT JUST A COST TRICK. Candidates
must be one-hot AND directional — the predicate
``reproducibility.warrants_reproducibility_check`` already defines, IMPORTED
rather than re-implemented. One-hot means entropy is uninformative (the blind
spot); directional means the item moved the posterior. A graded judgment
already reports its own doubt, and that doubt already reaches Gate 0, so
paying a call to hear "yes, unsure" buys nothing.

HOW THIS DIFFERS FROM ``reproducibility.py``. Complementary, not duplicative.
Reproducibility asks *"does the SAME judge survive paraphrase?"* — at the
decisive-posterior gate, a binary flip, up to four full ``judge_evidence``
round-trips. This asks *"does an INDEPENDENT judge, with a different schema,
agree?"* — during scrutiny, where the loop-back is still cheap, graded, one
round-trip. Both target the one-hot blind spot; both are suspend-only.

NOTHING HERE CHANGES A VERDICT. ``support_judgment``,
``judgment_distribution``, ``_item_masses`` and the weight label are never
written by this path. A confidently-dissenting confirmation raises
``Claim.judgment_dissent_count``, which buys the objective one bounded extra
round of inquiry via the existing Gate 0 ``Demand`` — and nothing else.

FAST, NOT PANEL — bare ``str`` model. ONE CRITERION — one model call per
confirmation. Both copy ``quality_doubt.py`` verbatim.

Architecture: Layer 1 (framework-agnostic; async driver + pure selection).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from andamentum.llm_judge import Criterion, judge_score
from andamentum.llm_judge.signals import normalize_three

from .entities.judge_opinion import JudgeOpinion
from .reproducibility import warrants_reproducibility_check
from .scrutiny_weight import item_directional_mass

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .entities import Evidence

# One criterion per stance — see the module docstring for why not four axes.
# The proposition is put AFFIRMATIVELY in both cases (does it support / does
# it contradict), which is what makes ``fails`` mean "dissents from the
# primary direction" rather than "agrees with the opposite one".
SUPPORTS_CRITERION = Criterion(
    name="text_supports_the_claim",
    description=(
        "The source text, read on its own terms, genuinely supports the claim "
        "stated in the context — rather than being about something else, or "
        "pointing the other way."
    ),
)

CONTRADICTS_CRITERION = Criterion(
    name="text_contradicts_the_claim",
    description=(
        "The source text, read on its own terms, genuinely contradicts the "
        "claim stated in the context — rather than being about something "
        "else, or pointing the other way."
    ),
)

_CRITERION_BY_STANCE: dict[str, Criterion] = {
    "supports": SUPPORTS_CRITERION,
    "contradicts": CONTRADICTS_CRITERION,
}


def select_confirmation_candidates(
    eligible: Sequence["Evidence"],
    *,
    per_pass_limit: int,
    remaining_budget: int,
) -> list["Evidence"]:
    """The load-bearing, unconfirmed, entropy-blind items worth one call each.

    PURE — no I/O, no model call — so the whole cost bound is testable
    without a runner.

    Filter:
      * ``reproducibility.warrants_reproducibility_check`` — one-hot AND
        directional. IMPORTED, never re-implemented: one selectivity
        definition, two consumers, pinned by a monkeypatch test.
      * ``direction_opinion is None`` — an Evidence is confirmed AT MOST
        ONCE, EVER. This persisted guard is what collapses the
        four-scrutiny-passes multiplier; without it a re-pass would re-pay
        for every item it already judged.

    Rank by ``scrutiny_weight.item_directional_mass`` — literally "how much
    did this item move the verdict" — with a deterministic
    ``(-mass, source_ref, entity_id)`` tie-break so the same evidence set
    always yields the same spend across re-runs.

    Both bounds are keyword-only: two same-typed ints whose transposition
    would silently widen or collapse the budget.
    """
    candidates = [
        ev
        for ev in eligible
        if warrants_reproducibility_check(ev) and ev.direction_opinion is None
    ]
    candidates.sort(
        key=lambda ev: (
            -item_directional_mass(ev),
            ev.source_ref or "",
            ev.entity_id,
        )
    )
    cap = min(per_pass_limit, remaining_budget)
    if cap <= 0:
        return []
    return candidates[:cap]


async def confirm_direction(
    *,
    content: str,
    source_ref: str,
    claim_statement: str,
    claim_scope: str,
    stance: str,
    model: str,
) -> JudgeOpinion:
    """Ask one independent judge to confirm the primary judge's direction.

    Every parameter is keyword-only: ``content`` / ``source_ref`` /
    ``claim_statement`` / ``claim_scope`` / ``stance`` / ``model`` are six
    same-typed strings, and transposing any pair would judge the wrong text
    against the wrong context, invisibly.

    Args:
        content: The extracted source text the primary judge read.
        source_ref: The source identifier, surfaced to the judge as context.
        claim_statement: The claim the direction was judged against.
        claim_scope: The claim's scope, so "on its own terms" is asked under
            the right conditions.
        stance: The primary judge's committed direction — ``"supports"`` or
            ``"contradicts"``. It selects which criterion is asked and is
            recorded as the opinion's ``reference``.
        model: A single pydantic-ai model id. A ``str`` (not a sequence) is
            required: it selects llm_judge's FAST path. A one-element LIST
            would silently select PANEL at sampling temperature —
            ``_is_panel`` keys on TYPE, not length.

    Returns:
        The judge's opinion. ``dissents`` is ``overall == "fails"``: the
        proposition was put affirmatively, so a "fails" IS a dissent from the
        direction the primary judge committed to. A ``partial`` is a hedge,
        not a contradiction, and does not dissent.

    Raises:
        ValueError: if ``stance`` is not a directional verdict. The selector
            only ever produces directional candidates, so reaching this is a
            caller bug, not a data condition — it fails loud rather than
            silently confirming the wrong proposition.
        Whatever ``judge_score`` raises — propagated, never swallowed into an
            unconfirmed verdict.
    """
    criterion = _CRITERION_BY_STANCE.get(stance)
    if criterion is None:
        raise ValueError(
            f"confirm_direction expects a directional stance "
            f"({sorted(_CRITERION_BY_STANCE)}), got {stance!r}. Candidates "
            "come from select_confirmation_candidates, which filters to "
            "directional judgments — a non-directional stance here means the "
            "caller bypassed that selector."
        )

    context = (
        f"Claim: {claim_statement or '(none)'}\n"
        f"Scope (the conditions under which the claim is asserted): "
        f"{claim_scope or 'General'}\n"
        f"Source: {source_ref}\n"
        "Below is the source text. Judge it on its own terms against the "
        "criterion."
    )
    result = await judge_score(
        content,
        criteria=[criterion],
        context=context,
        model=model,
    )
    row = result.per_criterion[0]
    return JudgeOpinion(
        subject=(
            f"Does this source {stance} the claim "
            f"'{claim_statement[:120]}'?"
        ),
        reference=stance,
        distribution=normalize_three(row.meets, row.partial, row.fails),
        overall=result.overall,
        expected_score=result.expected_score,
        summary=row.reasoning,
        dissents=result.overall == "fails",
    )


__all__ = [
    "SUPPORTS_CRITERION",
    "CONTRADICTS_CRITERION",
    "select_confirmation_candidates",
    "confirm_direction",
]
