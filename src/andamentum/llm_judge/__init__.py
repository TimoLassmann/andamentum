"""andamentum.llm_judge — a dead-simple LLM-as-judge primitive.

Two things:

1. ``await judge_score(output, *, criteria=None, context=None, model=...)``
   — score ONE output against criteria, per-criterion, with confidence and
   doubt.
2. ``await judge_compare(output_a, output_b, *, criteria=None, context=None,
   model=...)`` — compare TWO outputs (which is better), with confidence
   and doubt.

Both are single async entry points returning a bounded, flat-ish pydantic
schema (:class:`ScoreResult` / :class:`CompareResult`) — mirroring
``andamentum.vision_critique``'s shape: no ``pydantic-graph``, no
``graph.py``. There is nothing to route between here (one call, or a
sequential fan-out over judges/criteria with a single deterministic
roll-up) — so, per the dialect (Law 2 / Law 9), a graph engine would be
pure ceremony, not a missing piece. This module is still fully
dialect-conforming without one.

MODE LIVES IN ONE PARAMETER. ``model`` is the single mode switch:

- a ``str`` -> FAST path, one judge, temperature 0 (deterministic,
  reproducible).
- a ``Sequence[str]`` -> PANEL/CONSENSUS path, one call per model,
  strictly sequential, at a higher sampling temperature (see the
  temperature note below) — heterogeneous models or the SAME model
  repeated both work (the panel mechanics don't care which).

There is exactly one code path per function; the branch on ``model``'s
type is the mode axis the public API deliberately exposes, not a hidden
per-model-tier special case.

FAST vs. PANEL TEMPERATURE: FAST calls run at temperature 0 for
reproducibility. PANEL calls run at a higher, named sampling temperature
(``andamentum.llm_judge.elicit.PANEL_SAMPLING_TEMPERATURE``) — at
temperature 0, calling the same model K times returns K identical votes,
so a same-model-repeated panel's agreement gate would be trivially
unanimous and never raise ``needs_review``. This is why ``llm_judge`` owns
its own pydantic-ai ``Agent`` construction in ``elicit.py`` rather than
routing through ``core.AgentRunner`` (which does not expose
``model_settings``).

PANEL POOLING IS A FILTER, NOT AN ACCURACY BOOST. Averaging judges together
does not produce a "smarter" answer than the best single judge would have
given — the validated value of a panel is purely an AGREEMENT GATE:
unanimous votes are trusted more (conservative min-confidence
aggregation), a split panel sets ``needs_review=True`` and returns a
best-effort majority verdict. The caller is NEVER blocked — a panel result
is always returned, disagreement or not.

DOUBT IS A SUSPICION-RAISER ONLY. ``doubt`` (normalized Shannon entropy of
the verbalized verdict distribution) is a routing hint: high doubt is a
reason to escalate to a second look. Low doubt NEVER proves the verdict is
correct — it only means the judge was not internally conflicted. The
signal is validated as informative on capable (~12B+/mini-class) models
and is noise on nano-class models; this module applies one uniform formula
regardless of model tier (no tier branching), so weight ``doubt``
proportionally less when the judge is a very small model.

Every LLM-calling function in this module takes ``model=`` as a
keyword-only argument. There is no hidden default and no global config.

Example (fast path)::

    from andamentum.llm_judge import judge_score

    result = await judge_score(
        "Paris is the capital of France.",
        context="What is the capital of France?",
        model="anthropic:claude-haiku-4-5",
    )
    print(result.overall, result.confidence, result.doubt)

Example (panel path — heterogeneous models, or the same model repeated)::

    from andamentum.llm_judge import judge_compare

    result = await judge_compare(
        "The answer is 4.",
        "The answer is 5.",
        context="What is 2 + 2?",
        model=["anthropic:claude-haiku-4-5", "openai:gpt-5.4-nano"],
    )
    if result.needs_review:
        ...  # route for a second look
"""

from __future__ import annotations

from typing import Sequence

from . import panel
from .criteria import DEFAULT_CRITERIA
from .schemas import CompareResult, Criterion, CriterionScore, JudgeVote, ScoreResult

__all__ = [
    "judge_score",
    "judge_compare",
    "Criterion",
    "CriterionScore",
    "ScoreResult",
    "CompareResult",
    "JudgeVote",
    "DEFAULT_CRITERIA",
]


def _normalize_models(model: str | Sequence[str]) -> list[str]:
    """A bare ``str`` is one fast judge; any other ``Sequence[str]`` (even
    length 1) is a panel — the caller's choice of a list literal is
    respected as an explicit request for the panel code path."""
    if isinstance(model, str):
        return [model]
    return list(model)


def _is_panel(model: str | Sequence[str]) -> bool:
    """Mode axis lives in the *type* of ``model``, not its normalized
    length: a bare ``str`` is always the fast path; any ``Sequence[str]``
    — including a one-element list — is always the panel path. This is
    what lets a caller force the panel code path (sequential fan-out,
    sampling temperature, a ``judges`` list, an agreement gate) with a
    single model repeated, or even a literal one-element list."""
    return not isinstance(model, str)


def _validate_inputs(models: list[str], criteria: list[Criterion]) -> None:
    """Reject degenerate inputs AT THE BOUNDARY, before any model call.

    Without this, an empty ``model`` or ``criteria`` sequence sails past the
    entry point and dies several frames down in the pure-math layer with
    something like ``max() iterable argument is empty`` — an error that names
    nothing the caller actually did wrong. A judge is only useful if its
    failures are legible, so the diagnosis belongs here, where the mistake was
    made. These are caller errors, so they raise rather than degrade into a
    default verdict — this module never invents a score it did not derive.
    """
    if not models:
        raise ValueError(
            "model must name at least one judge: pass a model id (fast path) "
            "or a non-empty sequence of model ids (panel path)"
        )
    if any(not m.strip() for m in models):
        raise ValueError(f"model ids must be non-empty strings, got {models!r}")
    if not criteria:
        raise ValueError(
            "criteria must contain at least one Criterion — pass criteria=None "
            "to use the built-in DEFAULT_CRITERIA, not an empty list"
        )


async def judge_score(
    output: str,
    *,
    criteria: list[Criterion] | None = None,
    context: str | None = None,
    model: str | Sequence[str],
) -> ScoreResult:
    """Score ``output`` against ``criteria``, one verbalized distribution
    per criterion, rolled up to one overall verdict.

    Args:
        output: The text to judge.
        criteria: Criteria to score against. ``None`` -> :data:`DEFAULT_CRITERIA`
            (six short, domain-agnostic axes: correctness, completeness,
            instruction-following, sound reasoning, clarity, groundedness).
        context: The task/prompt ``output`` was answering. Optional, but
            strongly used by the prompt when present.
        model: A model id (fast, single judge, temperature 0) or a sequence
            of model ids (panel, sequential, sampling temperature — see
            module docstring). No default.

    Returns:
        A :class:`ScoreResult`. See its docstring for the equal-weight
        roll-up limitation and the doubt-signal caveat — both apply here.

    Raises:
        ValueError: on a degenerate input — an empty ``model`` sequence, a
            blank model id, or an empty ``criteria`` list (pass
            ``criteria=None`` for the defaults, not ``[]``).
        Whatever the underlying pydantic-ai call raises after the built-in
        ``PromptedOutput`` fallback (e.g. ``UnexpectedModelBehavior``) —
        this function never swallows a failure into a default answer.
    """
    resolved_criteria = criteria if criteria is not None else DEFAULT_CRITERIA
    is_panel = _is_panel(model)
    models = _normalize_models(model)
    _validate_inputs(models, resolved_criteria)
    if is_panel:
        parts = await panel.run_score_panel(
            output, resolved_criteria, context, models=models
        )
    else:
        parts = await panel.run_score_fast(
            output, resolved_criteria, context, model=models[0]
        )
    return ScoreResult(
        per_criterion=parts.per_criterion,
        overall=parts.overall,  # type: ignore[arg-type]
        confidence=parts.confidence,
        doubt=parts.doubt,
        needs_review=parts.needs_review,
        expected_score=parts.expected_score,
        judges=parts.judges,
    )


async def judge_compare(
    output_a: str,
    output_b: str,
    *,
    criteria: list[Criterion] | None = None,
    context: str | None = None,
    model: str | Sequence[str],
) -> CompareResult:
    """Compare ``output_a`` and ``output_b`` — which is better.

    Both presentation orders are always run (A-first and B-first), even in
    fast mode, and the winner is computed from the order-averaged
    histogram — a single-order call is vulnerable to position bias.

    Args:
        output_a: The first candidate output.
        output_b: The second candidate output.
        criteria: Criteria the judge should weigh (folded into the prompt's
            "criteria that matter" framing). ``None`` -> :data:`DEFAULT_CRITERIA`.
        context: The task/prompt both outputs were answering. Optional, but
            strongly used by the prompt when present.
        model: A model id (fast, single judge, temperature 0) or a sequence
            of model ids (panel, sequential, sampling temperature — see
            module docstring). No default.

    Returns:
        A :class:`CompareResult`. ``winner`` always holds a value
        (``'a'``/``'b'``/``'tie'``). See its docstring for the
        ``order_consistent`` semantics and the doubt-signal caveat.

    Raises:
        ValueError: on a degenerate input — an empty ``model`` sequence, a
            blank model id, or an empty ``criteria`` list (pass
            ``criteria=None`` for the defaults, not ``[]``).
        Whatever the underlying pydantic-ai call raises after the built-in
        ``PromptedOutput`` fallback — this function never swallows a
        failure into a default answer.
    """
    resolved_criteria = criteria if criteria is not None else DEFAULT_CRITERIA
    is_panel = _is_panel(model)
    models = _normalize_models(model)
    _validate_inputs(models, resolved_criteria)
    if is_panel:
        parts = await panel.run_compare_panel(
            output_a, output_b, resolved_criteria, context, models=models
        )
    else:
        parts = await panel.run_compare_fast(
            output_a, output_b, resolved_criteria, context, model=models[0]
        )
    return CompareResult(
        reasoning=parts.reasoning,
        winner=parts.winner,  # type: ignore[arg-type]
        confidence=parts.confidence,
        doubt=parts.doubt,
        order_consistent=parts.order_consistent,
        needs_review=parts.needs_review,
        expected_preference=parts.expected_preference,
        judges=parts.judges,
    )
