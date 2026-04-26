"""Tiered anchor matching + ModelRetry-raising validator factory.

Tiers (in priority order):
  1. exact            — substring match, case-sensitive
  2. whitespace_normalised — collapse all whitespace, lowercase
  3. fuzzy            — rapidfuzz token-set ratio > threshold

If none match, find_anchor returns None and the caller can either
raise ModelRetry (during validation) or escalate (during refinement).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# rapidfuzz is widely available; fall back to no fuzzy matching if absent.
try:
    from rapidfuzz import fuzz as _fuzz

    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover
    _fuzz = None  # type: ignore[assignment]
    _HAS_RAPIDFUZZ = False

_WS_RE = re.compile(r"\s+")
_FUZZY_MIN_SCORE = 85  # 0-100; rapidfuzz token_set_ratio


@dataclass
class AnchorMatch:
    """A successful anchor location in the source."""

    start: int
    end: int
    method: Literal["exact", "whitespace_normalised", "fuzzy", "best_effort"]


def _normalise(s: str) -> str:
    return _WS_RE.sub(" ", s).strip().lower()


def find_anchor(
    anchor: str,
    text: str,
    *,
    search_from: int,
) -> AnchorMatch | None:
    """Find `anchor` in `text` starting at `search_from`. Tiered match.

    Returns the FIRST match after `search_from` using the highest-priority
    tier that succeeds.
    """
    if not anchor:
        return None

    # Tier 1: exact substring
    pos = text.find(anchor, search_from)
    if pos != -1:
        return AnchorMatch(start=pos, end=pos + len(anchor), method="exact")

    # Tier 2: whitespace-normalised match (slide a window of equivalent length)
    # We compare normalised anchor to normalised candidates of similar size.
    norm_anchor = _normalise(anchor)
    if norm_anchor:
        # Heuristic: scan windows of [len(anchor) * 0.8, len(anchor) * 1.6]
        for window_len in range(int(len(anchor) * 0.8), int(len(anchor) * 1.6) + 1):
            for start in range(search_from, len(text) - window_len + 1):
                candidate = text[start : start + window_len]
                if _normalise(candidate) == norm_anchor:
                    return AnchorMatch(
                        start=start,
                        end=start + window_len,
                        method="whitespace_normalised",
                    )

    # Tier 3: fuzzy match
    if _HAS_RAPIDFUZZ and _fuzz is not None:
        for window_len in range(int(len(anchor) * 0.7), int(len(anchor) * 1.5) + 1):
            best_score = 0
            best_start = -1
            for start in range(search_from, len(text) - window_len + 1):
                candidate = text[start : start + window_len]
                score = _fuzz.token_set_ratio(anchor, candidate)
                if score > best_score:
                    best_score = score
                    best_start = start
            if best_score >= _FUZZY_MIN_SCORE and best_start >= 0:
                return AnchorMatch(
                    start=best_start,
                    end=best_start + window_len,
                    method="fuzzy",
                )

    return None
