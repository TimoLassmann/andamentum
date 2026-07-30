"""Public boundary types for ``andamentum.llm_judge``.

Every model here is flat, or a list of flat models — at most one level of
nesting, per the andamentum-wide rule that small local models fill bounded,
shallow schemas reliably. None of these types is ever filled directly by an
LLM call: the model only ever fills the private, single-distribution
elicitation schemas in :mod:`andamentum.llm_judge.elicit`; the types below
are assembled deterministically by :mod:`andamentum.llm_judge.panel` /
:mod:`andamentum.llm_judge.signals` from one or more of those calls.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Criterion(BaseModel):
    """One axis a judge scores an output against.

    ``description`` is a single sentence — enough for a small model to
    understand the axis without a paragraph of instructions per call.
    """

    name: str = Field(description="Short criterion name, e.g. 'correctness'.")
    description: str = Field(
        description="One sentence describing what this criterion means."
    )


class CriterionScore(BaseModel):
    """One judge's verbalized distribution for one criterion.

    ``meets`` + ``partial`` + ``fails`` sum to 100 — a verbalized
    distribution over three discrete outcomes, not a bare confidence
    number. ``reasoning`` is written by the model BEFORE the numbers
    (derive-then-judge) — the ordering that experiments/dirichlet_confidence
    and experiments/pairwise_judge both found essential for elicitation
    quality; the field order here mirrors that (reasoning first).
    """

    criterion: str = Field(description="The Criterion.name this score is for.")
    reasoning: str = Field(description="Reasoning written BEFORE the numbers below.")
    meets: int = Field(
        ge=0, le=100, description="Belief points that the output meets this criterion."
    )
    partial: int = Field(
        ge=0, le=100, description="Belief points that the output partially meets it."
    )
    fails: int = Field(
        ge=0, le=100, description="Belief points that the output fails this criterion."
    )


class JudgeVote(BaseModel):
    """One panel member's vote — flat, reused by both ``judge_score`` and
    ``judge_compare`` panels.

    Field meaning depends on the mode:

    - ``judge_score`` panel: ``verdict`` is one of ``'meets'``/``'partial'``/
      ``'fails'`` (this judge's own per-criterion roll-up); ``order_consistent``
      is always ``None`` (order-swap has no meaning for a single-output score).
    - ``judge_compare`` panel: ``verdict`` is one of ``'a'``/``'b'``/``'tie'``
      (this judge's order-averaged winner); ``order_consistent`` records
      whether THIS judge agreed with itself across the A-first and B-first
      presentation orders.
    """

    model: str = Field(description="The model id this judge ran as.")
    verdict: str = Field(
        description="This judge's verdict: score labels for judge_score, "
        "compare labels ('a'/'b'/'tie') for judge_compare."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="This judge's own confidence (max mass)."
    )
    doubt: float = Field(
        ge=0.0, le=1.0, description="This judge's own doubt (normalized entropy)."
    )
    order_consistent: bool | None = Field(
        default=None,
        description="judge_compare only: did this judge agree with itself across both orders?",
    )


class ScoreResult(BaseModel):
    """Result of :func:`andamentum.llm_judge.judge_score`.

    ``per_criterion`` is ASSEMBLED by code from one or more single-criterion
    model calls — the model never fills this list in one shot (small models
    are unreliable at multi-item, sum-to-100-per-item schemas). In fast mode
    each entry is one judge's own distribution; in panel mode each entry is
    the panel MEAN distribution across judges (with a deterministic
    ``reasoning`` note), rendered back to integers via
    ``signals.to_percentages`` (largest-remainder, so the row still sums to
    exactly 100). The panel mean is purely informational — the authoritative
    ``overall`` verdict is always the panel MAJORITY VOTE (see
    :mod:`andamentum.llm_judge.signals`), which can occasionally diverge from
    the pooled-mean argmax on a close split.

    ``overall`` is the equal-weight roll-up across criteria (the ``Criterion``
    schema has no weight field, so weighting is forced equal). This means a
    confident 'fails' on one important criterion (e.g. correctness) can be
    diluted by 'meets' on several others, yielding an overall 'meets' or
    'partial' that undersells a serious fault. Mitigation is in the prompt
    (judges are told a confident factual/logical error is a serious fault
    WITHIN a criterion), not in the roll-up weighting.

    ``confidence`` and ``doubt`` are DERIVED, never asked of the model
    directly. ``confidence`` is the probability mass on the verdict ACTUALLY
    REPORTED — in fast mode, this judge's mass on ``overall``; in panel mode,
    the mass the LEAST CONVINCED judge put on ``overall`` (conservative, per
    epistemic K-agreement). Note what this is NOT: it is not the minimum of
    each judge's confidence in its OWN verdict. On a split panel those differ,
    and only the former is meaningful — a dissenter's 0.80 belief in 'partial'
    says nothing about the panel's confidence in 'meets', and publishing it as
    such would attach one label's belief to a different label. ``doubt`` is
    the normalized Shannon entropy of the verdict distribution (panel: the MAX
    across judges — a suspicion-raiser is never dampened by more confident
    judges).

    DOUBT CAVEAT (read before using ``doubt`` for anything): doubt is a
    SUSPICION-RAISER ONLY. High doubt is a reason to escalate / route for a
    second look. Low doubt NEVER proves the verdict is correct — it only
    means the judge (or panel) was not conflicted. The signal is validated
    as informative on capable (~12B+/mini-class) models and is noise on
    nano-class models; this module does not branch behaviour on model tier,
    so treat doubt with proportionally more skepticism on very small models.

    ``needs_review`` follows one explicit, uniform predicate (see
    ``signals.needs_review_score``): panel split (not unanimous) OR
    doubt >= a fixed threshold. It is a routing hint, never a block —
    the result is always returned.

    ``expected_score`` is the CONTINUOUS companion to ``overall``: the
    expectation of the (equal-weight, criterion-averaged) belief
    distribution under ``meets``=1, ``partial``=0.5, ``fails``=0, so it lands
    in ``[0, 1]`` with higher = better. The benchmark in
    ``benchmarks/judge_scoring`` found this scalar a far stronger RANKING
    signal than the argmax label (AUROC ≈ 0.98 vs 76-83% argmax accuracy on
    the same elicited numbers) — it costs zero extra model calls, being read
    off the distribution already in hand. USE THE LABEL FOR A DECISION, THE
    SCALAR FOR RANKING / ROUTING / THRESHOLDING. On a close panel split the
    argmax of ``expected_score`` can disagree with ``overall`` (which is the
    majority VOTE, not the pooled-mean argmax) — that is the same
    label-vs-continuous divergence documented for ``per_criterion``, not a
    bug.
    """

    per_criterion: list[CriterionScore]
    overall: Literal["meets", "partial", "fails"]
    confidence: float = Field(ge=0.0, le=1.0)
    doubt: float = Field(ge=0.0, le=1.0)
    needs_review: bool
    expected_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Continuous score in [0,1] (meets=1, partial=0.5, fails=0) — "
        "a ranking signal; use `overall` for the decision.",
    )
    judges: list[JudgeVote] | None = None


class CompareResult(BaseModel):
    """Result of :func:`andamentum.llm_judge.judge_compare`.

    Both presentation orders (output_a-first and output_b-first) are always
    run, even in fast mode, and the winner is computed from the
    ORDER-AVERAGED histogram — never from a single order — because a bare
    single-order call is vulnerable to position bias. ``order_consistent``
    records whether the two orders agreed on the winner (using the same
    conservative tie-break as everything else here); a flip is itself a
    doubt signal and forces ``needs_review=True``.

    ``winner`` ALWAYS holds a value (``'a'``, ``'b'``, or ``'tie'``) — on a
    panel it is the majority vote across judges, with a documented
    conservative tie-break (never a silent default). A HUNG panel — judges
    split evenly between 'a' and 'b' — reports ``'tie'``, NEVER 'a'. Awarding
    a hung panel to 'a' would mean awarding it to whichever output the caller
    happened to pass first, reintroducing at the panel layer the very position
    bias that running both presentation orders exists to cancel. A single
    judge with an even a/b mass split already reports 'tie'; a hung panel is
    the same situation and answers the same way.

    ``confidence``/``doubt`` follow the same derivation and the same
    SUSPICION-RAISER-ONLY caveat as :class:`ScoreResult` — see that
    docstring; it is not repeated verbatim here to avoid drift, but it
    applies identically.

    ``needs_review`` follows ``signals.needs_review_compare``: order
    inconsistency (any judge, or the fast-mode single judge) OR panel split
    OR doubt above threshold. Never blocks the caller.

    ``expected_preference`` is the CONTINUOUS companion to ``winner``:
    ``E[preference for A]`` in ``[0, 1]`` — the expectation of the
    order-averaged (and, for a panel, judge-pooled) ``[pa, ptie, pb]``
    histogram under ``a``=1, ``tie``=0.5, ``b``=0. Above 0.5 favours A, below
    0.5 favours B, exactly 0.5 is indifferent. The benchmark in
    ``benchmarks/judge_scoring`` found it edges out the argmax on accuracy by
    converting ties into graded decisions, and — like ``expected_score`` on
    :class:`ScoreResult` — it is read off the distribution already computed,
    at zero extra model-call cost. USE ``winner`` FOR A DECISION, THIS SCALAR
    FOR RANKING / THRESHOLDING. It can land on the 'wrong' side of 0.5 from a
    hung-panel ``winner='tie'`` — the scalar exposes the lean the conservative
    tie-break deliberately refuses to call.
    """

    reasoning: str = Field(
        description="The judge's reasoning for the A-first order call."
    )
    winner: Literal["a", "b", "tie"]
    confidence: float = Field(ge=0.0, le=1.0)
    doubt: float = Field(ge=0.0, le=1.0)
    order_consistent: bool
    needs_review: bool
    expected_preference: float = Field(
        ge=0.0,
        le=1.0,
        description="E[preference for A] in [0,1] (a=1, tie=0.5, b=0) — a ranking "
        "signal; use `winner` for the decision.",
    )
    judges: list[JudgeVote] | None = None


__all__ = [
    "Criterion",
    "CriterionScore",
    "JudgeVote",
    "ScoreResult",
    "CompareResult",
]
