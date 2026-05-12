"""Description-driven evidence-provider dispatch.

Replaces the legacy three-agent chain
(``epistemic_select_provider`` + ``epistemic_rank_providers`` +
``epistemic_formulate_query``) with one generic per-provider dispatch
agent. Provider knowledge is read from each provider's class attributes
(``description``, ``query_guidance``, ``query_examples``) at runtime;
this module never hard-codes provider names or syntaxes.

Public API:

- ``DispatchResult`` — bundle of (queries, reasoning, confidence)
  produced by one dispatch call.
- ``formulate_provider_query(claim, provider, *, agent_runner)`` —
  ask the dispatch agent whether ``provider`` can help with ``claim``,
  and if so, construct one or two native-syntax queries.
- ``select_candidates_by_embedding(claim, providers, *, top_k, ...)`` —
  optional pre-filter that narrows N providers to top-K by description
  similarity. At the current 10-provider catalogue, ``top_k`` defaults
  to "no pre-filter" (pass-through) per the PRD; the helper exists so
  large-catalogue activation is a one-flag change later.
- ``gather_evidence_new(claim, providers, *, agent_runner, ...)`` —
  end-to-end orchestrator: pre-filter → dispatch → gather → aggregate.
  Returns ``list[GatheredEvidence]`` in the same shape the legacy
  pipeline produces, so this is drop-in for the existing extract step.

Phase 2 of the description-driven-dispatch PRD
(``docs/superpowers/plans/2026-05-12-description-driven-provider-dispatch.md``).
The legacy path keeps working unchanged through Phase 4; this module
runs alongside as the opt-in alternative.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from andamentum.core.agents import AgentRunner

from .agents import get_agent
from .operations import GatheredEvidence

logger = logging.getLogger(__name__)


# ── DispatchResult ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class DispatchResult:
    """Result of one dispatch call for one provider.

    ``queries`` is the operational output:

    - Empty list (``[]``) means the dispatch agent decided the provider
      should abstain on this claim. Don't call ``provider.gather()``.
    - 1+ strings means the dispatch agent committed to those queries.
      The orchestrator calls ``provider.gather(q)`` for each ``q`` and
      aggregates the resulting evidence.

    ``reasoning`` and ``confidence`` are diagnostic, not load-bearing
    — they go into the per-claim trace for debugging and audit.
    """

    queries: list[str]
    reasoning: str
    confidence: float

    @property
    def abstained(self) -> bool:
        """True when the dispatch agent decided this provider cannot help."""
        return not self.queries


# ── Per-provider dispatch (the agent call) ─────────────────────────────


async def formulate_provider_query(
    *,
    claim: str,
    provider_name: str,
    provider: Any,
    agent_runner: AgentRunner,
) -> DispatchResult:
    """Run the dispatch agent for one provider against one claim.

    Reads ``provider.description``, ``provider.query_guidance``, and
    ``provider.query_examples`` from the provider's class attributes
    (Phase 1 contract). Returns a ``DispatchResult`` describing the
    agent's routing decision.

    On hard agent failure (network, malformed output the agent runner
    can't recover from), returns ``DispatchResult(queries=[], ...)``
    rather than raising — the orchestrator treats this as "the
    provider couldn't be dispatched-to in this run." This mirrors the
    "providers never raise" convention from CONTRIBUTING.md.

    Args:
        claim: The research claim or sub-claim to construct queries for.
        provider_name: Short identifier (matches the registration key).
        provider: An instance of a provider class. Must have
            ``description``, ``query_guidance``, ``query_examples``
            class attributes per the Phase 1 contract.
        agent_runner: An ``AgentRunner`` configured with the dispatch
            model. Typically shared across many dispatch calls in one
            claim's processing.

    Returns:
        ``DispatchResult`` with 0, 1, or 2 queries. Never raises.
    """
    description = getattr(provider, "description", "")
    query_guidance = getattr(provider, "query_guidance", "")
    examples: list[tuple[str, str | None]] = getattr(provider, "query_examples", [])

    examples_block = _render_examples(examples)

    defn = get_agent("epistemic_dispatch_provider")

    try:
        result = await agent_runner.run(
            defn,
            claim=claim,
            provider_name=provider_name,
            provider_description=description,
            query_guidance=query_guidance,
            query_examples=examples_block,
        )
    except Exception as e:
        logger.warning(
            "Dispatch agent failed for provider=%s claim=%r: %s — "
            "treating as abstain",
            provider_name,
            claim[:80],
            e,
        )
        return DispatchResult(
            queries=[],
            reasoning=f"Dispatch failed: {type(e).__name__}",
            confidence=0.0,
        )

    # Clamp queries to at most 2 — defensive, since the prompt asks
    # for ≤ 2 but some small models occasionally produce more.
    queries = [q for q in result.queries if isinstance(q, str) and q.strip()]
    if len(queries) > 2:
        queries = queries[:2]

    confidence = float(getattr(result, "confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    return DispatchResult(
        queries=queries,
        reasoning=str(result.reasoning),
        confidence=confidence,
    )


def _render_examples(examples: list[tuple[str, str | None]]) -> str:
    """Format provider.query_examples as an in-context block for the
    dispatch agent. Empty list → "(no examples)"; pairs are formatted
    so the abstain signal (``None`` query) is obvious."""
    if not examples:
        return "(no examples provided)"

    lines = []
    for claim_example, query in examples:
        if query is None:
            lines.append(
                f"- Claim: {claim_example}\n  Action: ABSTAIN — provider can't help"
            )
        else:
            lines.append(f"- Claim: {claim_example}\n  Query: {query}")
    return "\n".join(lines)


# ── Embedding pre-filter (pass-through at current scale) ───────────────


async def select_candidates_by_embedding(
    *,
    claim: str,
    providers: dict[str, Any],
    top_k: int | None = None,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Narrow ``providers`` to a top-K candidate set by description-
    embedding similarity to the claim.

    At the current 10-provider catalogue, the default is "no
    pre-filter" — ``top_k=None`` returns the full provider dict
    unchanged. The PRD reasoning: low-yield providers contribute
    calibration signal via their abstention pattern, and silently
    pruning them before the dispatch agent runs would lose that
    signal. Real pre-filter activation is deferred to a follow-up PR
    when the catalogue exceeds ~30 providers.

    The helper exists in this shape so the activation later is a
    one-flag change. The implementation below is the pass-through
    default; the embedding logic (commented out) is the eventual
    activation path.

    If the embedding service fails, this function falls back to
    pass-through rather than empty — empty would silently kill
    evidence gathering for the claim.
    """
    if top_k is None or top_k >= len(providers):
        # Pass-through: every provider is a candidate.
        return dict(providers)

    # Real pre-filter (commented; activate when catalogue is large):
    #
    #   from .embeddings import embed_texts
    #   try:
    #       claim_vec = (await embed_texts([claim], model=embedding_model))[0]
    #       descriptions = [
    #           (name, getattr(p, "description", ""))
    #           for name, p in providers.items()
    #       ]
    #       desc_vecs = await embed_texts(
    #           [d for _, d in descriptions], model=embedding_model
    #       )
    #       scored = [
    #           (name, _cosine(claim_vec, vec))
    #           for (name, _), vec in zip(descriptions, desc_vecs)
    #       ]
    #       scored.sort(key=lambda item: item[1], reverse=True)
    #       top_names = [name for name, _ in scored[:top_k]]
    #       return {name: providers[name] for name in top_names}
    #   except Exception as e:
    #       logger.warning(
    #           "Embedding pre-filter failed (%s); falling back to all providers",
    #           e,
    #       )
    #       return dict(providers)
    #
    # Until activated: the truncation below is deterministic by dict
    # iteration order. This branch is reached only if a caller explicitly
    # passes top_k < len(providers); current callers don't.
    return dict(list(providers.items())[:top_k])


# ── Orchestrator ───────────────────────────────────────────────────────


async def gather_evidence_new(
    *,
    claim: str,
    providers: dict[str, Any],
    agent_runner: AgentRunner,
    top_k: int | None = None,
    embedding_model: str | None = None,
) -> list[GatheredEvidence]:
    """Description-driven gather: dispatch each provider, then gather.

    End-to-end alternative to the legacy ``PlanTaskOperation`` →
    ``ExtractEvidence`` chain. Returns ``list[GatheredEvidence]`` in
    the same shape the legacy pipeline produces, so this is drop-in
    for the existing extract step.

    Flow:

    1. Embedding pre-filter (pass-through at current scale).
    2. Dispatch agent runs once per candidate provider, in parallel.
    3. For each (provider, queries) where queries is non-empty, call
       ``provider.gather(q)`` for each ``q`` in parallel.
    4. Aggregate all resulting evidence into one list.

    Providers that the dispatch agent decides to abstain on never
    have their HTTP-call layer reached, by design.

    Args:
        claim: Claim or sub-claim text.
        providers: ``{name: provider_instance}``. Each provider must
            satisfy the Phase 1 contract (description, query_guidance,
            query_examples class attributes plus ``gather`` method).
        agent_runner: ``AgentRunner`` for the dispatch agent calls.
        top_k: If set and less than ``len(providers)``, narrows the
            candidate set via embedding similarity. ``None`` is the
            no-pre-filter default at current scale.
        embedding_model: Embedding model id for the pre-filter. Only
            consulted when ``top_k`` triggers actual narrowing.

    Returns:
        List of ``GatheredEvidence`` from all providers that returned
        anything. May be empty if every provider abstained or every
        ``gather()`` returned empty.
    """
    candidates = await select_candidates_by_embedding(
        claim=claim,
        providers=providers,
        top_k=top_k,
        embedding_model=embedding_model,
    )

    # Step 2: dispatch in parallel across candidates.
    dispatch_results = await asyncio.gather(
        *(
            formulate_provider_query(
                claim=claim,
                provider_name=name,
                provider=p,
                agent_runner=agent_runner,
            )
            for name, p in candidates.items()
        ),
        return_exceptions=False,  # formulate_provider_query never raises
    )

    # Step 3: gather() per provider for each committed query.
    #
    # asyncio.gather flattens all (provider, query) pairs into one
    # parallel batch. Providers whose dispatch returned [] get zero
    # gather calls. We log the dispatch trace per provider for audit.
    gather_tasks: list[Any] = []
    trace: list[tuple[str, DispatchResult, str]] = []
    for (name, p), dispatch in zip(candidates.items(), dispatch_results):
        if not dispatch.queries:
            logger.debug(
                "Dispatch abstained: provider=%s reasoning=%s",
                name,
                dispatch.reasoning,
            )
            continue
        for query in dispatch.queries:
            gather_tasks.append(p.gather(query))
            trace.append((name, dispatch, query))

    if not gather_tasks:
        logger.info(
            "All %d candidate providers abstained for claim=%r",
            len(candidates),
            claim[:80],
        )
        return []

    raw_results = await asyncio.gather(*gather_tasks, return_exceptions=True)

    # Step 4: aggregate, logging any provider HTTP failures.
    aggregated: list[GatheredEvidence] = []
    for (name, _dispatch, query), result in zip(trace, raw_results):
        if isinstance(result, BaseException):
            logger.warning(
                "Provider %s gather failed for query=%r: %s",
                name,
                query,
                result,
            )
            continue
        aggregated.extend(result)

    logger.info(
        "Description-driven dispatch: %d/%d providers dispatched, %d evidence items",
        len([t for t in trace if t]),
        len(candidates),
        len(aggregated),
    )
    return aggregated


__all__ = [
    "DispatchResult",
    "formulate_provider_query",
    "select_candidates_by_embedding",
    "gather_evidence_new",
]
