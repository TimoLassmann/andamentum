"""Cross-provider evidence deduplication by source_ref.

When the pipeline gathers evidence from multiple providers (pubmed,
openalex, europepmc, biorxiv, …), the same paper is often returned by
several. Without deduplication every duplicate gets its own Evidence
entity, gets content-extracted, and gets judged — paying linear LLM cost
per duplicate even though the second copy contributes no new information.

This module provides a post-extraction sweep that marks duplicates as
``invalidated=True``. Downstream filters (``count_supporting_sources``,
``count_support_contradict``, ``compute_posterior``) already exclude
invalidated evidence, so marking dupes here is enough — no changes
needed at the consumer side.

Dedupe key: a normalized ``source_ref`` (lowercase, DOI prefix stripped,
URL query/fragment stripped, trailing slash removed). Items with an empty
or null ``source_ref`` are not deduplicated against each other (no key →
no merge). Items already invalidated are skipped entirely (they don't
participate in dedupe).

When duplicates are found, the *winner* is the item with the most
``extracted_content`` (longest), breaking ties by oldest ``created_at``.
The winner accumulates ``also_found_by_source_types`` listing the other
providers that returned the same item, for transparency.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .repository import EpistemicRepository

logger = logging.getLogger(__name__)

_DOI_PREFIX_RE = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)
_PMID_PREFIX_RE = re.compile(r"^https?://(www\.)?(pubmed\.ncbi\.nlm\.nih\.gov|ncbi\.nlm\.nih\.gov/pubmed)/", re.IGNORECASE)


def normalize_source_ref(ref: str | None) -> str:
    """Normalize a source_ref string for stable dedupe-key matching.

    - Strips DOI URL prefixes (https://doi.org/, http://dx.doi.org/, …)
    - Strips PubMed URL prefixes (https://pubmed.ncbi.nlm.nih.gov/…)
    - Lowercases (DOIs are case-insensitive)
    - Strips trailing slash
    - Strips URL query params and fragment (everything after ? or #)
    - Returns "" for empty/None input

    Returns the normalized string. Empty input maps to "" — the caller
    should treat empty keys as non-deduplicable (every item with empty
    key keeps its own identity).
    """
    if not ref:
        return ""
    s = ref.strip()
    s = _DOI_PREFIX_RE.sub("", s)
    s = _PMID_PREFIX_RE.sub("", s)
    # Strip query string and fragment (URLs only — DOIs don't have these)
    for sep in ("?", "#"):
        idx = s.find(sep)
        if idx != -1:
            s = s[:idx]
    s = s.rstrip("/")
    return s.lower()


async def dedupe_evidence_by_source_ref(
    repo: "EpistemicRepository", objective_id: str
) -> tuple[int, int]:
    """Mark cross-provider duplicate evidence as invalidated.

    Walks all non-invalidated, extracted evidence for the objective.
    Groups by ``normalize_source_ref(source_ref)``. For each group with
    >1 item, keeps the item with the longest ``extracted_content`` (ties
    broken by oldest ``created_at``) and marks the rest as
    ``invalidated=True``, ``invalidation_reason="duplicate of <kept_id>
    (also returned by <providers>)"``.

    The kept item is unchanged; the audit trail of which providers also
    returned this paper is captured in each loser's
    ``invalidation_reason`` (and the operation log).

    Returns:
        (n_groups_with_dupes, n_marked_invalidated): how many distinct
        identifiers had duplicates, and how many evidence entities got
        invalidated as a result. Lets callers surface the cost saved.
    """
    from .entities import Evidence

    all_evidence = await repo.query("evidence", objective_id=objective_id)
    candidates: list[Evidence] = [
        e
        for e in all_evidence
        if isinstance(e, Evidence) and not e.invalidated and e.extracted
    ]

    # Group by normalized key. Empty keys produce no group — those items
    # are unique by definition (no identifier to match on).
    groups: dict[str, list[Evidence]] = {}
    for ev in candidates:
        key = normalize_source_ref(ev.source_ref)
        if not key:
            continue
        groups.setdefault(key, []).append(ev)

    n_groups_with_dupes = 0
    n_marked = 0

    for key, group in groups.items():
        if len(group) < 2:
            continue
        n_groups_with_dupes += 1

        # Winner: longest extracted_content, then oldest
        winner = max(
            group,
            key=lambda e: (
                len(e.extracted_content or ""),
                -(e.created_at.timestamp() if e.created_at else 0.0),
            ),
        )
        losers = [e for e in group if e is not winner]
        provider_list = sorted({e.source_type for e in losers if e.source_type})
        provider_blurb = (
            f" (also returned by {', '.join(provider_list)})"
            if provider_list
            else ""
        )

        for loser in losers:
            loser.invalidated = True
            loser.invalidation_reason = (
                f"Cross-provider duplicate of {winner.entity_id[:12]}"
                f"{provider_blurb}"
            )
            await repo.save(loser)
            n_marked += 1

        logger.info(
            "[dedupe_evidence] key=%s kept=%s (%s) marked %d duplicate(s) from %s",
            key,
            winner.entity_id[:12],
            winner.source_type,
            len(losers),
            provider_list,
        )

    if n_marked:
        logger.warning(
            "[dedupe_evidence] objective=%s: %d groups had duplicates; %d evidence marked invalidated",
            objective_id,
            n_groups_with_dupes,
            n_marked,
        )

    return n_groups_with_dupes, n_marked
