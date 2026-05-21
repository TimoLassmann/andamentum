"""Pure, deterministic substrate for comment consolidation.

This module knows nothing about LLMs, embeddings, or Ollama. It provides
the deterministic pieces the ``Consolidate`` node composes:

  • :func:`anchor_overlap` — do two findings point at overlapping spans?
  • :func:`union_find_groups` — connected components over "same" edges.
  • :func:`rollup_deterministic` — collapse high-volume style nitpicks
    (e.g. 73 passive-voice flags) into one per-(category, section) summary,
    keyed off COUNT, not category identity — so rare flags stay pinpointed.
  • :func:`merge_group` — fold a confirmed-same group of findings into one
    canonical finding, recording cross-perspective corroboration.

All functions are pure: they read findings and return new findings, never
mutating inputs. That keeps them unit-testable without a graph, a model,
or a network.
"""

from __future__ import annotations

from collections import defaultdict

from ..schemas import Finding

# A style category within one section collapses to a single summary comment
# only when it has at least this many instances there. Below it, each flag
# stays its own pinpoint comment. Keyed off count (general) rather than a
# hard-coded per-category allowlist (domain-specific). Tunable.
ROLLUP_MIN_COUNT = 3

_SEVERITY_RANK = {"minor": 1, "moderate": 2, "major": 3}
_CONFIDENCE_RANK = {"low": 1, "medium": 2, "high": 3}
_CONFIDENCE_BY_RANK = {1: "low", 2: "medium", 3: "high"}


def _first_span(f: Finding) -> tuple[str, int, int] | None:
    """Section id + char span of a finding's first quote, or None."""
    if not f.quotes:
        return None
    q = f.quotes[0]
    return q.section_id, q.char_start, q.char_end


def anchor_overlap(a: Finding, b: Finding) -> bool:
    """True when *a* and *b*'s first quotes share a section and overlap.

    Half-open intervals; touching-but-not-overlapping (``end == start``)
    does not count. Findings without a quote never anchor-overlap.
    """
    sa = _first_span(a)
    sb = _first_span(b)
    if sa is None or sb is None:
        return False
    sec_a, start_a, end_a = sa
    sec_b, start_b, end_b = sb
    if sec_a != sec_b:
        return False
    return start_a < end_b and start_b < end_a


def union_find_groups(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Connected components over ``range(n)`` given undirected *edges*.

    Returns one sorted list of member indices per component, components
    ordered by their smallest member. Singletons are included.
    """
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [sorted(members) for _, members in sorted(groups.items())]


def _perspectives_of(findings: list[Finding]) -> list[str]:
    """Distinct non-empty perspectives across *findings*, sorted."""
    seen: set[str] = set()
    for f in findings:
        if f.perspective:
            seen.add(f.perspective)
        seen.update(f.corroborated_by)
    return sorted(seen)


def merge_group(findings: list[Finding]) -> Finding:
    """Fold a confirmed-same group into one canonical finding.

    The canonical finding is the highest-severity, then highest-confidence
    member (its title / rationale / quotes are kept — the tightest anchor).
    Perspectives across the group are recorded in ``corroborated_by``; when
    ≥2 distinct perspectives independently raised the issue, confidence is
    bumped one tier (corroboration as signal). ``sections_involved`` is the
    union; ``source`` is ``challenged`` if any member came from an LLM,
    else ``deterministic``.

    A single-member group is returned unchanged.
    """
    if len(findings) == 1:
        return findings[0]

    canonical = max(
        findings,
        key=lambda f: (
            _SEVERITY_RANK.get(f.severity, 0),
            _CONFIDENCE_RANK.get(f.confidence, 0),
        ),
    )

    perspectives = _perspectives_of(findings)
    confidence = canonical.confidence
    if len(perspectives) >= 2:
        bumped = min(_CONFIDENCE_RANK[confidence] + 1, 3)
        confidence = _CONFIDENCE_BY_RANK[bumped]

    sections: list[str] = []
    for f in findings:
        for sid in f.sections_involved:
            if sid not in sections:
                sections.append(sid)

    any_llm = any(f.source in ("investigate", "challenged") for f in findings)

    return canonical.model_copy(
        update={
            "confidence": confidence,
            "corroborated_by": perspectives,
            "sections_involved": sections or canonical.sections_involved,
            "source": "challenged" if any_llm else "deterministic",
        }
    )


def rollup_deterministic(
    findings: list[Finding],
    *,
    min_count: int = ROLLUP_MIN_COUNT,
) -> list[Finding]:
    """Collapse high-volume deterministic style flags into per-section summaries.

    Groups deterministic findings by ``(category, section)``. A group with
    ≥ ``min_count`` instances becomes ONE summary finding anchored at its
    first instance, its rationale listing every flagged snippet so nothing
    is lost. Groups below the threshold pass through unchanged, as do
    findings with no category or no quote (nothing to group on).

    Order is preserved: each group's summary (or its members) appears where
    the group's first member was.
    """
    # Bucket indices by (category, section); ungroupable findings get a
    # unique singleton key so they pass through untouched in place.
    buckets: dict[object, list[int]] = defaultdict(list)
    for i, f in enumerate(findings):
        span = _first_span(f)
        if f.category and span is not None:
            buckets[(f.category, span[0])].append(i)
        else:
            buckets[("__ungrouped__", i)].append(i)

    # Emit in order of each bucket's first index.
    out: list[Finding] = []
    for key in sorted(buckets, key=lambda k: buckets[k][0]):
        idxs = buckets[key]
        members = [findings[i] for i in idxs]
        if len(members) < min_count:
            out.extend(members)
            continue
        out.append(_rollup_summary(members))
    return out


def _rollup_summary(members: list[Finding]) -> Finding:
    """One summary finding standing in for many same-category flags."""
    count = len(members)
    canonical = max(
        members,
        key=lambda f: (
            _SEVERITY_RANK.get(f.severity, 0),
            _CONFIDENCE_RANK.get(f.confidence, 0),
        ),
    )
    snippets = []
    for f in members:
        if f.quotes:
            q = f.quotes[0].text.replace("\n", " ").strip()
            if len(q) > 100:
                q = q[:97] + "…"
            snippets.append(f"  • {q}")
    instances_block = "\n".join(snippets)
    rationale = (
        f"{count} instances in this section. {canonical.rationale}".rstrip()
    )
    if instances_block:
        rationale = f"{rationale}\n\nFlagged spans:\n{instances_block}"
    return canonical.model_copy(
        update={
            "title": f"{count}× {canonical.title}",
            "rationale": rationale,
        }
    )
