"""Sequential fan-out and agreement-gate aggregation. Plain Python — no
graph engine, nothing to route between (see the module docstring in
``__init__.py`` for why this module is still dialect-conforming without one).

Every model call in this file runs strictly sequentially — a single
``for``/``await`` loop, never ``asyncio.gather`` — because Ollama serialises
inference per-process anyway, and running two local-model calls concurrently
just queues them upstream and risks timeout cascades (the andamentum-wide
rule: never run two Ollama calls concurrently).

Panel pooling does NOT boost accuracy over the best single judge — its only
validated value is an AGREEMENT GATE: unanimous panels are trusted more,
split panels raise ``needs_review``. This module never fabricates a
"smarter" pooled answer; the authoritative verdict is always the panel
MAJORITY VOTE, and ``confidence``/``doubt`` are aggregated conservatively
(min-confidence, max-doubt — mirroring
``andamentum.epistemic.operations.integration`` K-agreement).
"""

from __future__ import annotations

from . import elicit, signals
from .schemas import Criterion, CriterionScore, JudgeVote

# ── Per-judge routines (shared by fast and panel paths) ─────────────────


async def _judge_score_once(
    output: str,
    criteria: list[Criterion],
    context: str | None,
    *,
    model: str,
    temperature: float,
) -> tuple[list[CriterionScore], list[list[float]], list[float], str, float, float]:
    """Run one judge through every criterion, sequentially, and roll up.

    Returns ``(criterion_scores, per_criterion_dists, overall_dist, verdict,
    confidence, doubt)`` for this single judge. ``overall_dist`` is this
    judge's rolled-up belief vector over SCORE_LABELS; the panel needs it to
    ask each judge how much it believed the verdict the PANEL ended up
    reporting, which is not answerable from the judge's own scalar confidence.
    """
    criterion_scores: list[CriterionScore] = []
    dists: list[list[float]] = []
    for criterion in criteria:  # sequential — never fan out concurrently
        d = await elicit.elicit_criterion_score(
            output, criterion, context, model=model, temperature=temperature
        )
        dist = signals.normalize_three(d.meets, d.partial, d.fails)
        # Publish the NORMALIZED row, not the model's raw triple. Models are
        # asked for points summing to 100 but drift (a small model reading the
        # three fields as independent confidences happily returns 60/50/30).
        # The verdict is derived from `dist`, so publishing the raw numbers
        # would hand consumers a row that disagrees with the verdict it
        # produced — meets/100 = 0.60 where the actual belief was 0.43 — and
        # would make the fast path contradict the panel path, which reports
        # its pooled mean normalized. Same judge, one contract.
        meets, partial, fails = signals.to_percentages(dist)
        criterion_scores.append(
            CriterionScore(
                criterion=criterion.name,
                reasoning=d.reasoning,
                meets=meets,
                partial=partial,
                fails=fails,
            )
        )
        dists.append(dist)
    overall_dist, verdict, confidence, doubt = signals.roll_up_score(dists)
    return criterion_scores, dists, overall_dist, verdict, confidence, doubt


async def _judge_compare_once(
    output_a: str,
    output_b: str,
    criteria: list[Criterion],
    context: str | None,
    *,
    model: str,
    temperature: float,
) -> tuple[list[float], str, float, float, bool, str]:
    """Run one judge through BOTH presentation orders, sequentially.

    Returns ``(order_averaged_dist, winner, confidence, doubt,
    order_consistent, reasoning)`` for this single judge. ``reasoning`` is
    the A-first call's reasoning (the AB order is the canonical one to
    surface to the caller).
    """
    dist_ab_call = await elicit.elicit_pairwise(
        output_a,
        output_b,
        context,
        criteria,
        model=model,
        order="AB",
        temperature=temperature,
    )
    dist_ba_call = await elicit.elicit_pairwise(
        output_a,
        output_b,
        context,
        criteria,
        model=model,
        order="BA",
        temperature=temperature,
    )
    normalized_ab = signals.normalize_three(*dist_ab_call.to_row())
    normalized_ba = signals.normalize_three(*dist_ba_call.to_row())
    canon_ab = signals.canonicalize(normalized_ab, "AB")
    canon_ba = signals.canonicalize(normalized_ba, "BA")
    avg = signals.order_average(canon_ab, canon_ba)
    winner = signals.compare_verdict(avg)
    # Confidence is the mass on the winner ACTUALLY REPORTED, not the raw
    # maximum. They differ on an even a/b split, where `winner` is 'tie' but
    # the maximum sits on 'a' and 'b': reporting max_mass there would claim
    # 0.45 confidence in a 'tie' verdict holding only 0.10 of the belief.
    confidence = signals.mass_on(avg, signals.COMPARE_LABELS, winner)
    doubt = signals.normalized_entropy(avg)
    oc = signals.order_consistent(canon_ab, canon_ba)
    return avg, winner, confidence, doubt, oc, dist_ab_call.reasoning


# ── FAST (single judge) wrappers ─────────────────────────────────────────


async def run_score_fast(
    output: str, criteria: list[Criterion], context: str | None, *, model: str
) -> "ScoreResultParts":
    """Single-judge judge_score. Returns the parts ``__init__.py`` assembles
    into a :class:`~andamentum.llm_judge.schemas.ScoreResult`."""
    (
        criterion_scores,
        _dists,
        _overall_dist,
        verdict,
        confidence,
        doubt,
    ) = await _judge_score_once(
        output, criteria, context, model=model, temperature=elicit.FAST_TEMPERATURE
    )
    needs_review = signals.needs_review_score(unanimous=True, doubt=doubt)
    return ScoreResultParts(
        per_criterion=criterion_scores,
        overall=verdict,
        confidence=confidence,
        doubt=doubt,
        needs_review=needs_review,
        judges=None,
    )


async def run_compare_fast(
    output_a: str,
    output_b: str,
    criteria: list[Criterion],
    context: str | None,
    *,
    model: str,
) -> "CompareResultParts":
    """Single-judge judge_compare (both orders always run). Returns the
    parts ``__init__.py`` assembles into a
    :class:`~andamentum.llm_judge.schemas.CompareResult`."""
    _avg, winner, confidence, doubt, oc, reasoning = await _judge_compare_once(
        output_a,
        output_b,
        criteria,
        context,
        model=model,
        temperature=elicit.FAST_TEMPERATURE,
    )
    needs_review = signals.needs_review_compare(oc, unanimous=True, doubt=doubt)
    return CompareResultParts(
        reasoning=reasoning,
        winner=winner,
        confidence=confidence,
        doubt=doubt,
        order_consistent=oc,
        needs_review=needs_review,
        judges=None,
    )


# ── PANEL wrappers ───────────────────────────────────────────────────────


async def run_score_panel(
    output: str, criteria: list[Criterion], context: str | None, *, models: list[str]
) -> "ScoreResultParts":
    """Panel judge_score: every model in ``models`` scored sequentially at
    :data:`andamentum.llm_judge.elicit.PANEL_SAMPLING_TEMPERATURE`.

    ``overall`` is the majority vote across judges (authoritative, per the
    locked design). ``per_criterion`` is the panel-MEAN distribution per
    criterion — informational only; see ``ScoreResult`` docstring for why
    the mean and the majority can occasionally diverge on a close split.
    """
    verdicts: list[str] = []
    doubts: list[float] = []
    all_dists: list[list[list[float]]] = []  # per judge: list of per-criterion dists
    overall_dists: list[list[float]] = []  # per judge: rolled-up belief vector
    judges: list[JudgeVote] = []

    for model in models:  # sequential across judges — never concurrent
        (
            _scores,
            dists,
            overall_dist,
            verdict,
            confidence,
            doubt,
        ) = await _judge_score_once(
            output,
            criteria,
            context,
            model=model,
            temperature=elicit.PANEL_SAMPLING_TEMPERATURE,
        )
        verdicts.append(verdict)
        doubts.append(doubt)
        all_dists.append(dists)
        overall_dists.append(overall_dist)
        judges.append(
            JudgeVote(model=model, verdict=verdict, confidence=confidence, doubt=doubt)
        )

    majority, unanimous = signals.panel_majority(verdicts, signals.SCORE_TIEBREAK)
    # Ask each judge how much it believed the verdict the PANEL is reporting —
    # not how confident it was in its own. On a split panel those are different
    # questions, and only the first one is what `confidence` claims to answer.
    # (JudgeVote.confidence above stays each judge's belief in ITS OWN verdict:
    # that is the per-judge fact the caller wants when reading judges[].)
    panel_confidence = signals.panel_confidence(
        [signals.mass_on(d, signals.SCORE_LABELS, majority) for d in overall_dists]
    )
    panel_doubt = signals.panel_doubt(doubts)
    needs_review = signals.needs_review_score(unanimous, panel_doubt)

    n_judges = len(models)
    pooled_criterion_scores: list[CriterionScore] = []
    for i, criterion in enumerate(criteria):
        pooled_dist = signals.mean_distributions(
            [judge_dists[i] for judge_dists in all_dists]
        )
        meets, partial, fails = signals.to_percentages(pooled_dist)
        pooled_criterion_scores.append(
            CriterionScore(
                criterion=criterion.name,
                reasoning=f"Panel mean of {n_judges} judges; see judges[] for individual votes.",
                meets=meets,
                partial=partial,
                fails=fails,
            )
        )

    return ScoreResultParts(
        per_criterion=pooled_criterion_scores,
        overall=majority,
        confidence=panel_confidence,
        doubt=panel_doubt,
        needs_review=needs_review,
        judges=judges,
    )


async def run_compare_panel(
    output_a: str,
    output_b: str,
    criteria: list[Criterion],
    context: str | None,
    *,
    models: list[str],
) -> "CompareResultParts":
    """Panel judge_compare: every model in ``models`` compared sequentially
    (both orders each) at :data:`andamentum.llm_judge.elicit.PANEL_SAMPLING_TEMPERATURE`.

    Panel ``order_consistent`` is ``all(judge.order_consistent)`` — a single
    flip anywhere in the panel is enough to flag it.

    A HUNG panel (judges split evenly between 'a' and 'b') reports
    ``winner='tie'``, never 'a'. See :func:`signals.panel_majority` — breaking
    such a tie toward a side would hand the win to whichever output the caller
    passed first, reintroducing the position bias the both-orders machinery
    exists to cancel.
    """
    winners: list[str] = []
    doubts: list[float] = []
    order_flags: list[bool] = []
    avgs: list[list[float]] = []  # per judge: canonical [pa, ptie, pb]
    judges: list[JudgeVote] = []
    reasoning = ""

    for i, model in enumerate(models):  # sequential across judges
        (
            avg,
            winner,
            confidence,
            doubt,
            oc,
            judge_reasoning,
        ) = await _judge_compare_once(
            output_a,
            output_b,
            criteria,
            context,
            model=model,
            temperature=elicit.PANEL_SAMPLING_TEMPERATURE,
        )
        winners.append(winner)
        doubts.append(doubt)
        order_flags.append(oc)
        avgs.append(avg)
        judges.append(
            JudgeVote(
                model=model,
                verdict=winner,
                confidence=confidence,
                doubt=doubt,
                order_consistent=oc,
            )
        )
        if i == 0:
            reasoning = judge_reasoning

    majority, unanimous = signals.panel_majority(
        winners,
        signals.COMPARE_TIEBREAK,
        undecided=signals.COMPARE_UNDECIDED,
    )
    # Each judge's belief in the winner the PANEL is reporting — see the note
    # in run_score_panel; the same reasoning applies verbatim here.
    panel_confidence = signals.panel_confidence(
        [signals.mass_on(a, signals.COMPARE_LABELS, majority) for a in avgs]
    )
    panel_doubt = signals.panel_doubt(doubts)
    panel_order_consistent = all(order_flags)
    needs_review = signals.needs_review_compare(
        panel_order_consistent, unanimous, panel_doubt
    )

    return CompareResultParts(
        reasoning=reasoning,
        winner=majority,
        confidence=panel_confidence,
        doubt=panel_doubt,
        order_consistent=panel_order_consistent,
        needs_review=needs_review,
        judges=judges,
    )


# ── Plain result-part carriers ───────────────────────────────────────────
#
# Flat carriers (not part of the public schema surface) so __init__.py can
# build the final pydantic ScoreResult/CompareResult in one place. Kept as
# simple attribute-holding classes rather than re-importing the public
# pydantic models here, to keep this module's only pydantic dependency on
# the already-imported schemas.


class ScoreResultParts:
    """Everything :func:`andamentum.llm_judge.judge_score` needs to build a
    :class:`~andamentum.llm_judge.schemas.ScoreResult`."""

    def __init__(
        self,
        *,
        per_criterion: list[CriterionScore],
        overall: str,
        confidence: float,
        doubt: float,
        needs_review: bool,
        judges: list[JudgeVote] | None,
    ) -> None:
        self.per_criterion = per_criterion
        self.overall = overall
        self.confidence = confidence
        self.doubt = doubt
        self.needs_review = needs_review
        self.judges = judges


class CompareResultParts:
    """Everything :func:`andamentum.llm_judge.judge_compare` needs to build a
    :class:`~andamentum.llm_judge.schemas.CompareResult`."""

    def __init__(
        self,
        *,
        reasoning: str,
        winner: str,
        confidence: float,
        doubt: float,
        order_consistent: bool,
        needs_review: bool,
        judges: list[JudgeVote] | None,
    ) -> None:
        self.reasoning = reasoning
        self.winner = winner
        self.confidence = confidence
        self.doubt = doubt
        self.order_consistent = order_consistent
        self.needs_review = needs_review
        self.judges = judges


__all__ = [
    "run_score_fast",
    "run_compare_fast",
    "run_score_panel",
    "run_compare_panel",
    "ScoreResultParts",
    "CompareResultParts",
]
