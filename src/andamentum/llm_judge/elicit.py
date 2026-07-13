"""The single model-call layer, one call per verbalized distribution.

This module — not ``andamentum.core.agents`` — owns pydantic-ai ``Agent``
construction for ``llm_judge``, because ``core.AgentRunner`` /
``run_agent_with_fallback`` do not expose ``model_settings`` (temperature),
and this module needs temperature as a first-class, named parameter: FAST
(single-judge) calls run at temperature 0 for reproducibility, while PANEL
calls run at :data:`PANEL_SAMPLING_TEMPERATURE` so that a same-model-repeated
panel actually produces varying votes — at temperature 0, K calls to the
same model return K identical outputs and the panel's only validated value
(the agreement gate) collapses to noise. This is a mode-axis parameter (fast
vs. panel — an axis the public API already branches on via the ``model``
argument shape), not a hidden per-model-tier branch.

Both elicited schemas here are private and flat (reasoning first, then a
sum-to-100 triple) — the model NEVER fills the public list-shaped
``ScoreResult``/``CompareResult`` types directly; those are assembled by
:mod:`andamentum.llm_judge.panel` from one or more calls into this module.

``elicit_criterion_score`` and ``elicit_pairwise`` are the two functions
tests monkeypatch to run fully offline — treat their signatures as a stable
seam.

Failure handling is loud by design: a failed tool-based call falls back to
``PromptedOutput`` exactly once; if that also fails, the exception
propagates to the caller. Nothing here swallows an error to return a
default value — a swallowed failure here would become a silent wrong
answer.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.models import Model
from pydantic_ai.output import PromptedOutput
from pydantic_ai.settings import ModelSettings

from andamentum.core import resolve_model

from .prompts import (
    SCORE_INSTRUCTIONS,
    build_compare_instructions,
    build_compare_prompt,
    build_score_prompt,
)
from .schemas import Criterion

T = TypeVar("T", bound=BaseModel)

# FAST (single judge) is deterministic and reproducible. PANEL sampling uses
# a non-zero temperature so repeated draws of the same model actually vary —
# see module docstring for why this is load-bearing, not a style choice.
FAST_TEMPERATURE = 0.0
PANEL_SAMPLING_TEMPERATURE = 0.7


# The field descriptions below are not decoration: they are compiled into the
# JSON schema the model actually fills, and on the PromptedOutput path that
# schema IS the instruction the model sees. Leaving them bare (`Field(ge=0,
# le=100)`) states the range but never states that the three are ONE
# distribution summing to 100 — which is exactly the misreading that makes a
# small model return three independent confidences like 60/50/30.


class _CriterionDist(BaseModel):
    """Private elicitation schema for one judge_score call: one criterion,
    reasoning first, then a meets/partial/fails distribution."""

    reasoning: str = Field(description="Reasoning written BEFORE the numbers below.")
    meets: int = Field(
        ge=0,
        le=100,
        description="Belief points that the output MEETS this criterion. "
        "meets + partial + fails must sum to exactly 100.",
    )
    partial: int = Field(
        ge=0,
        le=100,
        description="Belief points that the output PARTIALLY meets this criterion. "
        "meets + partial + fails must sum to exactly 100.",
    )
    fails: int = Field(
        ge=0,
        le=100,
        description="Belief points that the output FAILS this criterion. "
        "meets + partial + fails must sum to exactly 100.",
    )


class _PairwiseDist(BaseModel):
    """Private elicitation schema for one judge_compare call: one
    presentation order, reasoning first, then a response_1_better/tie/
    response_2_better distribution (position-neutral labels)."""

    reasoning: str = Field(description="Reasoning written BEFORE the numbers below.")
    response_1_better: int = Field(
        ge=0,
        le=100,
        description="Belief points that Response 1 is better. The three numbers "
        "must sum to exactly 100.",
    )
    tie: int = Field(
        ge=0,
        le=100,
        description="Belief points that the two responses are equally good. The "
        "three numbers must sum to exactly 100.",
    )
    response_2_better: int = Field(
        ge=0,
        le=100,
        description="Belief points that Response 2 is better. The three numbers "
        "must sum to exactly 100.",
    )

    def to_row(self) -> list[float]:
        """Raw ``[p1, ptie, p2]`` over the shown Response 1 / Response 2."""
        return [
            float(self.response_1_better),
            float(self.tie),
            float(self.response_2_better),
        ]


# ── Caches ───────────────────────────────────────────────────────────────
#
# TWO caches, because the two things being cached have very different
# lifetimes and very different costs to get wrong.
#
# The RESOLVED MODEL owns the transport. `core.resolve_model` is not itself
# cached and is not free: for `ollama:` it builds a fresh OllamaModel +
# OllamaProvider (and with it a new httpx.AsyncClient) on every call, and for
# `bedrock:` a fresh boto3 session and client. Only `openai:`/`anthropic:`
# pass through as a bare string. So resolving per agent-cache-miss would mint,
# and never close, one connection pool per miss — a file-descriptor leak on
# precisely the local-Ollama path this module is designed around. It is keyed
# by model id alone, which is naturally bounded: a caller uses a handful of
# judges, not an unbounded set.
_MODEL_CACHE: dict[str, Model | str] = {}

# The AGENT is a pure wrapper over a resolved model — it holds no transport,
# so evicting one is harmless. It is worth caching anyway (construction costs
# ~10ms, and a 3-model panel over 6 criteria would otherwise build 18+), but
# it must be BOUNDED: `build_compare_instructions` folds the criterion names
# into the instructions string, so the key varies with the caller's criteria
# set. An unbounded dict therefore grows without limit in a long-running
# scorer that sees many criteria sets. A small LRU keeps the hit rate for the
# access pattern that actually matters (the same criteria set reused across
# the criteria of one call, and across panel repeats) while making unbounded
# growth impossible.
_AGENT_CACHE_MAXSIZE = 64
_AGENT_CACHE: OrderedDict[tuple[str, str, str, bool], Agent[Any, Any]] = OrderedDict()

# Statuses on which retrying with PromptedOutput is a plausible remedy: the
# request itself was rejected as malformed, which a tool/function-calling
# schema the provider dislikes can genuinely cause. Everything else — 401
# (auth), 429 (rate limit), 5xx (server) — has nothing to do with output mode.
# Retrying those doubles the billed requests on every hard failure, and turns
# a 429 into an immediate second knock on an endpoint that just said "slow
# down". A context-length 400 is the one case where the retry is arguably
# worse (PromptedOutput inlines the JSON schema, making the prompt LONGER),
# but it is not separable by status code, and the tool-schema case is the one
# this fallback exists for.
_FALLBACK_HTTP_STATUSES = frozenset({400, 422})


def _get_resolved_model(model: str) -> Model | str:
    cached = _MODEL_CACHE.get(model)
    if cached is None:
        cached = resolve_model(model)
        _MODEL_CACHE[model] = cached
    return cached


def _get_agent(
    model: str, instructions: str, output_type: type[T], *, prompted: bool
) -> Agent[Any, Any]:
    key = (model, instructions, output_type.__name__, prompted)
    cached = _AGENT_CACHE.get(key)
    if cached is not None:
        _AGENT_CACHE.move_to_end(key)
        return cached
    resolved = _get_resolved_model(model)
    built_output_type = PromptedOutput(output_type) if prompted else output_type
    agent = Agent(resolved, instructions=instructions, output_type=built_output_type)
    _AGENT_CACHE[key] = agent
    if len(_AGENT_CACHE) > _AGENT_CACHE_MAXSIZE:
        _AGENT_CACHE.popitem(last=False)  # evict least-recently-used
    return agent


def _should_fall_back(exc: Exception) -> bool:
    """Is ``exc`` plausibly a STRUCTURED-OUTPUT failure that PromptedOutput
    could fix — as opposed to a failure that a second identical request cannot
    possibly help with?"""
    if isinstance(exc, UnexpectedModelBehavior):
        return True
    if isinstance(exc, ModelHTTPError):
        return exc.status_code in _FALLBACK_HTTP_STATUSES
    return False


async def _run(
    model: str,
    instructions: str,
    prompt: str,
    output_type: type[T],
    temperature: float,
) -> T:
    """Run one elicitation call: tool-based output, falling back to
    ``PromptedOutput`` exactly once when — and only when — the failure is one
    PromptedOutput could actually fix (see :func:`_should_fall_back`). Any
    other exception, or a failure of the fallback itself, propagates — this
    function never swallows an error.
    """
    settings = ModelSettings(temperature=temperature)
    try:
        agent = _get_agent(model, instructions, output_type, prompted=False)
        result = await agent.run(prompt, model_settings=settings)
        return result.output
    except (UnexpectedModelBehavior, ModelHTTPError) as exc:
        if not _should_fall_back(exc):
            raise
        fallback_agent = _get_agent(model, instructions, output_type, prompted=True)
        result = await fallback_agent.run(prompt, model_settings=settings)
        return result.output


async def elicit_criterion_score(
    output: str,
    criterion: Criterion,
    context: str | None,
    *,
    model: str,
    temperature: float,
) -> _CriterionDist:
    """One judge_score call: score ``output`` against a single ``criterion``."""
    prompt = build_score_prompt(output, criterion, context)
    return await _run(model, SCORE_INSTRUCTIONS, prompt, _CriterionDist, temperature)


async def elicit_pairwise(
    output_1: str,
    output_2: str,
    context: str | None,
    criteria: list[Criterion],
    *,
    model: str,
    order: Literal["AB", "BA"],
    temperature: float,
) -> _PairwiseDist:
    """One judge_compare call for one presentation order.

    ``output_1``/``output_2`` are always the caller's canonical
    output_a/output_b, in that fixed argument position — the ``order``
    argument (not argument reordering) determines whether ``output_1`` is
    shown as Response 1 (``'AB'``) or Response 2 (``'BA'``), matching the
    validated ``experiments/pairwise_judge`` pattern. The caller
    (:mod:`andamentum.llm_judge.panel`) always invokes this the same way for
    both orders, differing only in the ``order`` keyword.
    """
    if order == "AB":
        r1, r2 = output_1, output_2
    elif order == "BA":
        r1, r2 = output_2, output_1
    else:
        raise ValueError(f"order must be 'AB' or 'BA', got {order!r}")
    instructions = build_compare_instructions(criteria)
    prompt = build_compare_prompt(r1, r2, context)
    return await _run(model, instructions, prompt, _PairwiseDist, temperature)


__all__ = [
    "FAST_TEMPERATURE",
    "PANEL_SAMPLING_TEMPERATURE",
    "elicit_criterion_score",
    "elicit_pairwise",
]
