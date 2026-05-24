"""Layer-1 tools for v3 criterion-review agents.

Two pure-Python tools, universally available to every criterion (no
opt-in via ``Criterion.tools`` needed — those are reserved for layer 2
and beyond):

- ``read_section(section_id)`` — return the full text of a section the
  digest only summarised.
- ``search_paper(query, max_results=5, regex=False)`` — substring or
  regex search across the source.

Both are pure functions over the ``DocumentModel`` held in ``DocDeps``.
No LLM calls, no network, no external deps beyond stdlib ``re``.

Error-signalling discipline: when something goes wrong (unknown
section id, malformed regex, query too long, regex timeout) the tools
``raise ModelRetry(...)``. Pydantic-AI synthesises a ``RetryPromptPart``
from the exception message and sends it back to the model on its next
turn, just like it does for argument-schema mismatches. This routes
through the per-tool retry counter (``retries=N`` on the agent); a
model that loops on bad ids will surface as ``UnexpectedModelBehavior``
after N misses, which ``run_criteria`` catches and degrades gracefully.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from pydantic_ai import ModelRetry, RunContext

from .model import DocumentModel, Section

logger = logging.getLogger("andamentum.whetstone.v3")

# Width of the snippet returned per match — chars on each side of the
# hit, so the total snippet is ~200 chars. Small enough that dozens of
# matches fit in one tool response; large enough to give the agent
# useful context for each hit.
_SNIPPET_PADDING = 100

# Regex-mode safety knobs. Patterns longer than this almost always come
# from a confused LLM trying to encode logic in regex that doesn't
# belong there; reject at the door.
_MAX_REGEX_LENGTH = 200

# Wall-clock guard on regex execution. Python's stdlib ``re`` has no
# native timeout parameter and ``signal.alarm`` isn't async-safe; the
# cleanest stdlib-only option is to run finditer in a thread via
# ``asyncio.to_thread`` and bound it with ``asyncio.wait_for``.
#
# Honest limitation: a single catastrophic-backtracking iteration that
# never returns can't be cancelled (Python threads aren't cancellable);
# ``wait_for`` returns and the agent gets a timeout error, but the
# thread leaks until the regex naturally completes. If this becomes a
# practical problem we'd swap stdlib ``re`` for the third-party
# ``regex`` module — single-line change, it has a real ``timeout=``.
_REGEX_TIMEOUT_S = 2.0


@dataclass
class DocDeps:
    """Typed dependency object handed to v3 review tools through
    pydantic-ai's ``RunContext``.

    Layer 2 will extend this with novelty-search counters and a
    deep_research handle; layer 3 may add a code-interpreter handle.
    For now it carries only the ``DocumentModel`` the tools read from.
    """

    document_model: DocumentModel


async def read_section(ctx: RunContext[DocDeps], section_id: str) -> str:
    """Return the full text of the section identified by ``section_id``.

    Valid ids appear in the SECTIONS block of your prompt (for example
    ``"4.2"``, ``"abstract"``, or ``"sec_004"`` — whatever the digest
    listed). If ``section_id`` doesn't match any section, raises
    ``ModelRetry`` so pydantic-ai sends the error back to the model and
    lets it correct on its next turn.

    Use this when the digest's one-sentence gist isn't enough to answer
    a criterion question — for example, when prior-stage findings draw
    attention to a section and you want to read it in full.
    """
    section = ctx.deps.document_model.section_by_id(section_id)
    if section is None:
        logger.info("[v3.tool] read_section(%r) → no such section", section_id)
        raise ModelRetry(
            f"no section with id {section_id!r}; valid ids are listed in the "
            f"SECTIONS block of your prompt. Re-issue read_section with one "
            f"of them."
        )
    logger.info(
        "[v3.tool] read_section(%r) → %d chars", section_id, len(section.text)
    )
    return section.text


async def search_paper(
    ctx: RunContext[DocDeps],
    query: str,
    *,
    max_results: int = 5,
    regex: bool = False,
) -> list[dict]:
    """Search across the paper for ``query``.

    With ``regex=False`` (default) does a case-insensitive substring
    search — always succeeds, no compile step, no failure modes. With
    ``regex=True`` treats ``query`` as a Python regex with
    ``re.IGNORECASE``; useful for alternation
    (``(limitation|caveat|weakness)``), character classes
    (``Theorem [0-9]+``), and word boundaries (``\\bAdam\\b``).

    Returns a list of matches, each
    ``{section_id, snippet, position}``, capped at ``max_results``. An
    empty list means the term doesn't appear in the paper — that's a
    real signal a reviewer often wants (confirming absence).

    Use this to verify whether the paper mentions a concept the digest
    didn't surface, to find every cross-reference to a table or
    theorem, or to confirm that something you're about to flag as
    missing genuinely is missing.

    On regex-mode problems (invalid pattern, pattern too long, runaway
    backtracking) raises ``ModelRetry`` so pydantic-ai sends the error
    back to the model and lets it correct on its next turn.
    """
    source = ctx.deps.document_model.source
    sections = ctx.deps.document_model.sections

    mode = "regex" if regex else "substring"
    if regex:
        if len(query) > _MAX_REGEX_LENGTH:
            logger.info("[v3.tool] search_paper(%r, regex) → too long", query[:40])
            raise ModelRetry(
                f"regex pattern too long ({len(query)} chars; max "
                f"{_MAX_REGEX_LENGTH}). Use a simpler pattern, or call "
                f"search_paper with regex=False for plain substring search."
            )
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error as e:
            logger.info(
                "[v3.tool] search_paper(%r, regex) → compile error: %s", query, e
            )
            raise ModelRetry(
                f"invalid regex {query!r}: {e}. Try a simpler pattern, or "
                f"call search_paper with regex=False for plain substring search."
            )
        try:
            positions = await asyncio.wait_for(
                asyncio.to_thread(_regex_positions, pattern, source, max_results),
                timeout=_REGEX_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.info("[v3.tool] search_paper(%r, regex) → timed out", query)
            raise ModelRetry(
                f"regex {query!r} timed out (>{_REGEX_TIMEOUT_S}s) — "
                f"likely catastrophic backtracking. Use a simpler pattern, "
                f"or call search_paper with regex=False for plain substring search."
            )
    else:
        positions = _substring_positions(query, source, max_results)

    matches = [_build_match(start, end, source, sections) for start, end in positions]
    logger.info(
        "[v3.tool] search_paper(%r, %s) → %d match(es)", query, mode, len(matches)
    )
    return matches


# ── pure helpers ────────────────────────────────────────────────────


def _substring_positions(
    needle: str, source: str, max_results: int
) -> list[tuple[int, int]]:
    """All (start, end) char-ranges where ``needle`` appears in
    ``source``, case-insensitive, up to ``max_results`` non-overlapping
    hits. Empty needle returns no hits (otherwise every position would
    match, which is meaningless)."""
    if not needle:
        return []
    needle_lower = needle.lower()
    source_lower = source.lower()
    positions: list[tuple[int, int]] = []
    cursor = 0
    while True:
        idx = source_lower.find(needle_lower, cursor)
        if idx < 0:
            break
        positions.append((idx, idx + len(needle)))
        if len(positions) >= max_results:
            break
        cursor = idx + len(needle)
    return positions


def _regex_positions(
    pattern: re.Pattern[str], source: str, max_results: int
) -> list[tuple[int, int]]:
    """Synchronous regex finditer, called from a thread via
    ``asyncio.to_thread`` so it can be bounded by ``wait_for``."""
    positions: list[tuple[int, int]] = []
    for m in pattern.finditer(source):
        positions.append((m.start(), m.end()))
        if len(positions) >= max_results:
            break
    return positions


def _build_match(
    start: int, end: int, source: str, sections: list[Section]
) -> dict:
    """Build the dict representation of a single match for the agent."""
    snippet_start = max(0, start - _SNIPPET_PADDING)
    snippet_end = min(len(source), end + _SNIPPET_PADDING)
    return {
        "section_id": _section_id_at(start, sections),
        "snippet": source[snippet_start:snippet_end].strip(),
        "position": start,
    }


def _section_id_at(position: int, sections: list[Section]) -> str:
    """Return the id of the section whose char-range contains
    ``position``, or ``"?"`` for positions that fall outside any
    section (e.g. whitespace between sections — rare but possible)."""
    for s in sections:
        if s.start <= position < s.end:
            return s.id
    return "?"
