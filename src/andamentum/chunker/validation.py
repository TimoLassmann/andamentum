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
from typing import Callable, Literal

from pydantic_ai import ModelRetry

from .types import NextUnitResult
from .windowing import Window

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


def make_validator(window: Window) -> Callable[[NextUnitResult], NextUnitResult]:
    """Build an output_validator closed over the current window.

    The validator raises ``pydantic_ai.ModelRetry`` with deterministic,
    actionable feedback on any of:
      - found=True but anchors empty
      - found=False but skip_to empty
      - start_anchor not findable in the visible text
      - end_anchor not findable AFTER start_anchor
      - end_anchor lands before start_anchor (inverted range)
    """

    def validate(output: NextUnitResult) -> NextUnitResult:
        problems: list[str] = []

        if output.found:
            if not output.start_anchor or not output.end_anchor:
                problems.append(
                    "found=True requires both start_anchor and end_anchor "
                    "to be non-empty (verbatim from the visible text)."
                )
            else:
                start = find_anchor(output.start_anchor, window.text, search_from=0)
                if start is None:
                    problems.append(
                        f"start_anchor {output.start_anchor!r} was not found "
                        f"anywhere in the visible text. Copy it VERBATIM from "
                        f"the source — exact wording, exact spelling."
                    )

                end = (
                    find_anchor(output.end_anchor, window.text, search_from=start.end)
                    if start is not None
                    else None
                )
                if start is not None and end is None:
                    problems.append(
                        f"end_anchor {output.end_anchor!r} was not found AFTER "
                        f"start_anchor in the visible text. Make sure end_anchor "
                        f"comes after start_anchor and is copied VERBATIM."
                    )
                if start is not None and end is not None and end.start <= start.end:
                    problems.append(
                        "end_anchor must land STRICTLY AFTER start_anchor — "
                        "they appear to overlap or be reversed."
                    )
        else:
            if not output.skip_to:
                problems.append(
                    "found=False requires skip_to to be set to a short verbatim "
                    "phrase from near the end of the visible text, so the "
                    "system can advance the cursor past the junk region."
                )

        if problems:
            raise ModelRetry("\n".join(problems))
        return output

    return validate
