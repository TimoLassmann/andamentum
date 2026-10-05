"""Changes that flip a statistical conclusion, read from the text around a number.

Two patterns, both on the paragraph text after the number is bound:

- a p-value: the number is preceded by ``p =``, ``p <`` or ``P >`` and its old
  and new values lie on different sides of .05 (significant means below .05);
- a confidence interval: ``95% CI [lo, hi]`` or ``(lo to hi)`` where at least
  one limit is a changed number; the interval contains the null value (0, or 1
  when the words before it name a ratio: OR, HR, RR, "ratio") before and not
  after, or the other way round. A limit that did not change is read from the
  text.

Anything else (an interval written without brackets, a p-value in a table, a
threshold other than .05) is not examined. That is a coverage limit: a missing
flag means nothing was recognised, not that the conclusion holds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ALPHA = 0.05
_P_BEFORE = re.compile(r"(?:^|[\s(,;\[])[pP]\s*[=<>≤≥]\s*$")
_NUM = r"[-−]?\d+(?:\.\d+)?"
_CI = re.compile(
    rf"CI[^\[\(\d]{{0,25}}[\[\(]\s*(?P<lo>{_NUM})\s*(?:,|;|\bto\b|–|—)\s*(?P<hi>{_NUM})\s*[\]\)]"
)
_RATIO = re.compile(r"\b(?:OR|HR|RR|IRR|[Rr]atio)\b")


@dataclass(frozen=True)
class Change:
    """A bound number whose shown value changed (paragraph-local offsets)."""

    start: int
    end: int
    old: float
    new: float
    old_text: str
    new_text: str


@dataclass(frozen=True)
class Flip:
    kind: str  # "p" or "ci"
    at: int  # paragraph-local offset of the number (p) or of the interval (ci)
    message: str


def _float(text: str) -> float:
    return float(text.replace("−", "-"))


def find_flips(text: str, changes: list[Change]) -> list[Flip]:
    """The significance flips among *changes*, which are on *text*."""
    flips: list[Flip] = []
    for change in changes:
        if _P_BEFORE.search(text[: change.start]) and (change.old < ALPHA) != (
            change.new < ALPHA
        ):
            flips.append(
                Flip(
                    "p",
                    change.start,
                    f"p changed {change.old_text} -> {change.new_text} and now lies "
                    f"on the other side of {ALPHA:g}",
                )
            )
    by_span = {(c.start, c.end): c for c in changes}
    for match in _CI.finditer(text):
        limits = []
        touched = False
        for name in ("lo", "hi"):
            span = match.span(name)
            change = by_span.get(span)
            touched = touched or change is not None
            value = _float(match[name])
            limits.append((change.old, change.new) if change else (value, value))
        if not touched:
            continue
        before = text[max(0, match.start() - 60) : match.start()]
        null = 1.0 if _RATIO.search(before) else 0.0

        def contains(low: float, high: float) -> bool:
            return min(low, high) <= null <= max(low, high)

        was = contains(limits[0][0], limits[1][0])
        now = contains(limits[0][1], limits[1][1])
        if was != now:
            flips.append(
                Flip(
                    "ci",
                    match.start(),
                    f"the interval changed to [{limits[0][1]:g}, {limits[1][1]:g}] "
                    f"and {'now contains' if now else 'no longer contains'} {null:g}",
                )
            )
    return flips
