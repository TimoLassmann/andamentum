"""Node: Consolidate — collapse redundant comments before synthesis.

Runs after ``AuthorQuestions``, before ``Synthesise``, on the UNION of the
LLM findings (``challenged_findings``) and the deterministic findings — the
one place deterministic findings finally enter the quality pipeline instead
of bypassing it.

Three tiers (PRD: docs/plans/2026-05-21-whetstone-consolidate-prd.md):

  1. Substrate (deterministic, cheap) — proposes candidates, never decides:
       • High-volume style flags roll up per (category, section).
       • Candidate edges between findings IN THE SAME SECTION: overlapping
         anchors OR similar claims (embedding cosine, strict threshold).
         Cross-section semantic merges are held off for now.
  2. Adjudication (LLM, flat binary) — for each candidate edge, "same or
     distinct?". Union-find over the "same" verdicts rebuilds merge groups
     transitively, so a small model never has to emit a partition.
  3. Merge (deterministic) — fold each confirmed-same group into one
     finding, recording cross-perspective corroboration (and bumping
     confidence when ≥2 perspectives agree).

Embeddings are REQUIRED: if Ollama is unreachable the embed call raises and
the run fails loud (per CONSTITUTION.md — no silent skip, no fallback).
Novelty findings are document-level (no real anchor) and are passed through
untouched.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic_graph import BaseNode, GraphRunContext

from ..agents import SameOrDistinct, build_pydantic_ai_agent
from ..deps import ReviewDeps
from ..schemas import Finding, ReviewResult
from ..state import ReviewState
from ..structural.consolidation import (
    anchor_overlap,
    merge_group,
    rollup_deterministic,
    union_find_groups,
)

if TYPE_CHECKING:
    from .synthesise import Synthesise


logger = logging.getLogger("andamentum.whetstone")

# Cosine threshold above which two same-section claims are a candidate for
# the same issue. This is a RECALL gate, not the decider — the LLM adjudicates
# every candidate as same/distinct. Set permissively (the LLM is the
# discriminator); within-section candidate counts are small so extra pairs are
# cheap. Not empirically calibrated for embeddinggemma — tune from a real run.
_SIMILARITY_THRESHOLD = 0.7

# Dropped from 6 → 2: pair adjudications fan out per candidate (often 60-90),
# and the resulting concurrent burst was saturating the OpenAI connection
# path (Connection-error waves in batches of N parallel calls).
_MAX_CONCURRENT_ADJUDICATIONS = 2


@dataclass
class Consolidate(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Collapse redundant findings into corroborated comments."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "Synthesise":
        ctx.state.current_phase = "consolidate"

        # Novelty findings are document-level, not anchored — leave them be.
        det_novelty, det_real = _partition_novelty(ctx.state.deterministic_findings)
        llm_novelty, llm_real = _partition_novelty(ctx.state.challenged_findings)

        # Tier 1a: roll up high-volume deterministic style flags.
        det_rolled = rollup_deterministic(det_real)

        pool = det_rolled + llm_real
        if len(pool) < 2:
            logger.info("[consolidate] <2 findings — nothing to consolidate")
            ctx.state.deterministic_findings = det_rolled + det_novelty
            ctx.state.challenged_findings = llm_real + llm_novelty
            from .synthesise import Synthesise

            return Synthesise()

        # Tier 1b: candidate edges (anchor overlap ∪ embedding similarity).
        vectors = await _embed_claims(ctx.deps, pool)
        candidates = _candidate_edges(pool, vectors)
        logger.info(
            "[consolidate] %d finding(s) → %d candidate pair(s)",
            len(pool),
            len(candidates),
        )

        # Tier 2: adjudicate each candidate pair (flat same/distinct).
        same_edges = await _adjudicate(ctx.deps, pool, candidates)
        ctx.state.llm_calls += len(candidates)

        # Tier 3: union-find → merge groups → fold each.
        groups = union_find_groups(len(pool), same_edges)
        merged = [merge_group([pool[i] for i in g]) for g in groups]

        det_out = [f for f in merged if f.source == "deterministic"]
        llm_out = [f for f in merged if f.source != "deterministic"]
        ctx.state.deterministic_findings = det_out + det_novelty
        ctx.state.challenged_findings = llm_out + llm_novelty
        logger.info(
            "[consolidate] done — %d finding(s) after merge "
            "(%d same-pair(s) of %d candidate(s))",
            len(merged),
            len(same_edges),
            len(candidates),
        )

        from .synthesise import Synthesise

        return Synthesise()


# ── Helpers ─────────────────────────────────────────────────────────────


def _partition_novelty(
    findings: list[Finding],
) -> tuple[list[Finding], list[Finding]]:
    """Split into (novelty, the rest). Novelty findings skip consolidation."""
    novelty = [f for f in findings if f.category == "novelty"]
    rest = [f for f in findings if f.category != "novelty"]
    return novelty, rest


def _claim_text(f: Finding) -> str:
    """The text we embed: the finding's claim (title + rationale), per decision 3."""
    return f"{f.title}\n{f.rationale}".strip()


async def _embed_claims(deps: ReviewDeps, pool: list[Finding]) -> list[list[float]]:
    """Embed every finding's claim. Ollama required — failure raises."""
    claims = [_claim_text(f) for f in pool]
    if deps.embedding_fn is not None:
        return await deps.embedding_fn(claims)
    from andamentum.core import embed_texts

    return await embed_texts(claims, model=deps.embedding_model)


def _section_key(f: Finding) -> str | None:
    """The section a finding lives in (its first quote's, else its first
    declared section). None when it has neither — such a finding can't be
    same-section matched."""
    if f.quotes:
        return f.quotes[0].section_id
    return f.sections_involved[0] if f.sections_involved else None


def _candidate_edges(
    pool: list[Finding], vectors: list[list[float]]
) -> list[tuple[int, int]]:
    """Pairs worth adjudicating — scoped to WITHIN a section for now.

    A pair is a candidate when the two findings live in the same section AND
    either their anchors overlap OR their claims are similar (cosine ≥
    threshold). Cross-section semantic merges are intentionally held off
    until within-section merging is proven — same-section is lower risk and
    covers the bulk of true duplicates (multiple lenses on one passage).

    Skips pairs that are both deterministic with the same category — those
    are roll-up territory, not genuine same/distinct judgements.
    """
    from andamentum.core import cosine_similarity

    edges: list[tuple[int, int]] = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            a, b = pool[i], pool[j]
            sec_a = _section_key(a)
            if sec_a is None or sec_a != _section_key(b):
                continue  # cross-section (or unanchored) — held off for now
            if (
                a.source == "deterministic"
                and b.source == "deterministic"
                and a.category == b.category
            ):
                continue
            if anchor_overlap(a, b) or (
                cosine_similarity(vectors[i], vectors[j]) >= _SIMILARITY_THRESHOLD
            ):
                edges.append((i, j))
    return edges


async def _adjudicate(
    deps: ReviewDeps,
    pool: list[Finding],
    candidates: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Run one binary same/distinct call per candidate pair (bounded parallel).

    Returns the subset of edges judged "same". A crashed call defaults to
    "distinct" (keep both — safer than hiding a real second issue), matching
    the loud-fail-safe pattern used by Challenge.
    """
    if not candidates:
        return []
    sem = asyncio.Semaphore(_MAX_CONCURRENT_ADJUDICATIONS)

    async def one(edge: tuple[int, int]) -> tuple[int, int] | None:
        i, j = edge
        async with sem:
            try:
                verdict = await _judge_pair(deps, pool[i], pool[j])
            except Exception as exc:
                logger.warning(
                    "[consolidate] pair (%d,%d) crashed: %s — keeping distinct",
                    i,
                    j,
                    exc,
                )
                return None
            return edge if verdict.relation == "same" else None

    results = await asyncio.gather(*[one(e) for e in candidates])
    return [e for e in results if e is not None]


def _section_of(f: Finding) -> str:
    return f.sections_involved[0] if f.sections_involved else "?"


def _quote_of(f: Finding) -> str:
    return repr(f.quotes[0].text) if f.quotes else "(none)"


async def _judge_pair(deps: ReviewDeps, a: Finding, b: Finding) -> SameOrDistinct:
    """One same/distinct call for two findings."""
    prompt = f"""FINDING A:
title:     {a.title}
section:   {_section_of(a)}
rationale: {a.rationale}
quote:     {_quote_of(a)}

FINDING B:
title:     {b.title}
section:   {_section_of(b)}
rationale: {b.rationale}
quote:     {_quote_of(b)}

Are these the SAME issue, or DISTINCT?"""

    agent = build_pydantic_ai_agent("consolidate", deps.model)
    result = await agent.run(prompt)
    return cast(SameOrDistinct, result.output)
