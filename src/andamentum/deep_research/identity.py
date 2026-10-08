"""Merge the records several sources return for one work into one candidate.

Keys come from ``Work.identity_keys`` in a fixed order: DOI, arXiv id, PMID,
PMCID, OpenAlex id, Semantic Scholar id, canonical URL, then normalised title
plus year. A shared hard key (every key but the title) means the same work, and
merging is transitive: a record carrying a DOI from one group and a PMID from
another joins them. The title key merges two records only when they hold no
identifier that disagrees (``works_compatible``), so a preprint and its
published version stay two works.

Output order is the order of each work's first appearance, so a run is
reproducible from its inputs.
"""

from __future__ import annotations

from collections.abc import Iterable

from andamentum.references import Work, merge_works, works_compatible

from .types import Candidate


def merge(candidates: Iterable[Candidate]) -> list[Candidate]:
    """One candidate per distinct work among *candidates*."""
    groups: dict[int, Candidate] = {}
    owner: dict[tuple[str, str], int] = {}
    for position, candidate in enumerate(candidates):
        hard = candidate.work.hard_keys()
        group_ids = sorted({owner[key] for key in hard if key in owner})
        if not group_ids:
            group_ids = _title_match(candidate, groups, owner)
        if not group_ids:
            groups[position] = candidate
            _register(position, candidate, owner)
            continue
        survivor = group_ids[0]
        combined = merge_pair(groups[survivor], candidate)
        for other in group_ids[1:]:
            combined = merge_pair(combined, groups.pop(other))
        groups[survivor] = combined
        # A merge keeps one value per field; every key any merged record
        # carried still leads to the surviving group.
        for key, group_id in owner.items():
            if group_id in group_ids:
                owner[key] = survivor
        _register(survivor, candidate, owner)
        _register(survivor, combined, owner)
    return [groups[position] for position in sorted(groups)]


def _title_match(
    candidate: Candidate,
    groups: dict[int, Candidate],
    owner: dict[tuple[str, str], int],
) -> list[int]:
    for kind, value in candidate.work.identity_keys():
        if kind != "title" or (kind, value) not in owner:
            continue
        group_id = owner[(kind, value)]
        if works_compatible(groups[group_id].work, candidate.work):
            return [group_id]
    return []


def _register(
    group_id: int, candidate: Candidate, owner: dict[tuple[str, str], int]
) -> None:
    for key in candidate.work.identity_keys():
        owner[key] = group_id


def merge_pair(first: Candidate, second: Candidate) -> Candidate:
    """Combine two records of one work: the work merged, the provenance unioned."""
    ranks = [r for r in (first.rank, second.rank) if r is not None]
    work = merge_works(first.work, second.work)
    return Candidate(
        work=work,
        sources=_union(first.sources, second.sources),
        snippet=_merged_snippet(first, second, work),
        queries=_union(first.queries, second.queries),
        subquestion_ids=_union(first.subquestion_ids, second.subquestion_ids),
        rank=min(ranks) if ranks else None,
        source_ranks=_best_ranks(first.source_ranks, second.source_ranks),
    )


def _merged_snippet(first: Candidate, second: Candidate, work: Work) -> str:
    """The longer snippet; but a snippet is the first characters of its record's
    abstract, so when the merged work is redistributable (a free abstract won)
    the snippet of a ``no_redistribute`` record is not carried into it."""
    options = [first, second]
    if work.redistributable:
        options = [c for c in options if c.work.redistributable] or options
    return max((c.snippet for c in options), key=len)


def _best_ranks(first: dict[str, int], second: dict[str, int]) -> dict[str, int]:
    best = dict(first)
    for name, position in second.items():
        best[name] = min(position, best.get(name, position))
    return best


def _union(first: list[str], second: list[str]) -> list[str]:
    return list(dict.fromkeys([*first, *second]))
