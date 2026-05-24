"""Tiered anchor matching for the editor / benchmark.

Used by the editor's `/api/match-anchor` endpoint and by the benchmark
runner to resolve truth-file anchors against source. The structural-first
chunker doesn't need anchor matching for its main path (it knows positions
directly), but exposing this is the contract benchmark cases rely on.

This module is now a thin shim over ``andamentum.core.text_match.find_span``
— the single canonical answer to "is this verbatim in the source." The
tier-named ``AnchorMatch.method`` ("exact" / "whitespace_normalised" /
"fuzzy") is preserved as the public contract so existing callers
(``whetstone/anchoring.py``, ``whetstone/lenses/strunk/nodes/*``,
``whetstone/nodes/edit_sections.py``) don't have to change.

Tiers (in priority order):
  1. exact                  — byte-identical substring (case-sensitive,
                              preserves the chunker's load-bearing
                              byte-identical contract for FTS5).
  2. whitespace_normalised  — exact in normalised space (markdown
                              stripped, smart quotes stripped, case
                              folded, whitespace collapsed). Picks up
                              markdown + quote stripping from the
                              shared canonical contract; before
                              consolidation this tier was case-fold +
                              whitespace-collapse only.
  3. fuzzy                  — rapidfuzz token_set_ratio ≥ 0.85 over a
                              sliding window of [0.7×, 1.5×] anchor
                              length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from andamentum.core.text_match import find_span

_FUZZY_MIN_SCORE = 0.85  # rapidfuzz token_set_ratio threshold (0.0-1.0)


@dataclass
class AnchorMatch:
    """A successful anchor location in the source."""

    start: int
    end: int
    method: Literal["exact", "whitespace_normalised", "fuzzy", "best_effort"]


def find_anchor(
    anchor: str,
    text: str,
    *,
    search_from: int,
) -> AnchorMatch | None:
    """Find ``anchor`` in ``text`` starting at ``search_from``. Tiered match.

    Returns the FIRST match after ``search_from`` using the highest-priority
    tier that succeeds. The method label in the returned ``AnchorMatch``
    names which tier produced the result.
    """
    if not anchor:
        return None

    match = find_span(
        anchor,
        text,
        within=(search_from, len(text)),
        fuzzy="rapidfuzz",
        fuzzy_threshold=_FUZZY_MIN_SCORE,
    )
    if match is None:
        return None

    # Translate the unified API's method label into the chunker-specific
    # public vocabulary that existing callers depend on.
    method: Literal["exact", "whitespace_normalised", "fuzzy", "best_effort"]
    if match.method == "exact":
        method = "exact"
    elif match.method == "normalized":
        method = "whitespace_normalised"
    else:
        method = "fuzzy"

    return AnchorMatch(start=match.start, end=match.end, method=method)


# `make_validator` (the LLM ModelRetry validator) was removed when the
# chunker switched from agentic-windowing to structural-first extraction.
# `find_anchor` above is now the only thing this module exports.
