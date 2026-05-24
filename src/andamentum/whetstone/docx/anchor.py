"""Normalised, document-level text anchoring for comment placement.

The quote a finding refers to comes from harvested markdown and carries
markup (``##`` headings, ``[links]``, ``*emphasis*``) and markdown
whitespace (``\\n\\n``). The Word body is plain text. To locate the
quote robustly we normalise BOTH sides identically — lowercase, drop a
common set of markdown markers, collapse every run of whitespace to a
single space — and search the normalised forms.

After normalisation the match is **exact** — no fuzzy thresholds, no
dynamic-programming alignment. Anything that still doesn't match is a
genuine miss and is reported (with the closest region, for diagnosis).

The normalisation rules are shared with ``andamentum.core.text_match``
(``MARKDOWN_MARKERS`` constant + ``normalize_for_match``) so the rules
cannot drift between the flat string matcher used by whetstone v3 and
the multi-segment indexing here. ``DocIndex`` keeps its own per-run
loop because the cross-paragraph synthetic-separator behaviour
(inserting a normalised space between paragraphs so a quote can match
across a ``## Heading\\n\\nBody`` boundary) is docx-specific and not
appropriate for the flat matcher.

This module is pure and framework-free: it operates on ``(key, text)``
segments, where for a .docx the key identifies a run. It has no docx or
lxml dependency, so it is unit-testable in isolation.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Optional

from andamentum.core.text_match import MARKDOWN_MARKERS, normalize_for_match

# Internal alias preserved for the DocIndex inline loop below — sharing
# the exact frozenset from core.text_match guarantees the character
# class can't drift.
_MARKDOWN_MARKERS = MARKDOWN_MARKERS


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalise *text*; return ``(normalized, index_map)``.

    Thin shim over ``core.text_match.normalize_for_match`` with the
    canonical defaults (drop markdown markers, strip edge quotes,
    lowercase, collapse whitespace). The signature stays narrow so
    existing callers (``DocIndex``, ``locate``) don't have to know
    about kwargs.

    ``index_map[i]`` is the index in the ORIGINAL *text* of the character
    that produced normalised character ``i`` — so a match found in
    normalised space can be mapped back to original offsets.
    """
    return normalize_for_match(text)


@dataclass(frozen=True)
class AnchorSpan:
    """Where a target resolved, in segment-local coordinates.

    ``start_key`` / ``end_key`` identify the segments (runs) the match
    starts and ends in; ``start_char`` / ``end_char`` are character
    offsets WITHIN those segments. ``end_char`` is inclusive of the last
    matched character (half-open ``end_char + 1`` when slicing).
    """

    start_key: Any
    start_char: int
    end_key: Any
    end_char: int


class DocIndex:
    """Normalised, document-level searchable index over paragraphs of runs.

    Build it from an ordered list of paragraphs, each an ordered list of
    ``(run_key, run_text)``. Runs WITHIN a paragraph concatenate directly
    (Word runs carry no implied space); a single normalised space is
    inserted BETWEEN paragraphs — mirroring how markdown ``\\n\\n`` and a
    Word paragraph break both normalise to one space. This lets a quote
    that spans a heading→body boundary (``## Heading\\n\\nBody``) match
    across the two paragraphs.

    Per normalised character it records which run it came from and the
    char offset within that run; ``find`` maps a match back to an
    :class:`AnchorSpan`.
    """

    def __init__(self, paragraphs: list[list[tuple[Any, str]]]):
        self._keys: list[Any] = []  # per normalised char → run key (None = synthetic separator)
        self._chars: list[int] = []  # per normalised char → char index in run
        parts: list[str] = []
        prev_space = True
        for p_i, runs in enumerate(paragraphs):
            if p_i > 0 and not prev_space:
                # synthetic inter-paragraph separator → one normalised space
                parts.append(" ")
                self._keys.append(None)
                self._chars.append(-1)
                prev_space = True
            for key, text in runs:
                for i, ch in enumerate(text):
                    if ch in _MARKDOWN_MARKERS:
                        continue
                    if ch.isspace():
                        if prev_space:
                            continue
                        parts.append(" ")
                        self._keys.append(key)
                        self._chars.append(i)
                        prev_space = True
                    else:
                        parts.append(ch.lower())
                        self._keys.append(key)
                        self._chars.append(i)
                        prev_space = False
        self._norm = "".join(parts)

    @property
    def normalized(self) -> str:
        return self._norm

    def find(self, target: str) -> Optional[AnchorSpan]:
        """Locate *target* (normalised) in the document. None if absent
        or if a match endpoint falls on a synthetic paragraph separator
        (which would have no real run to anchor to)."""
        norm_target, _ = normalize_with_map(target)
        norm_target = norm_target.strip()
        if not norm_target:
            return None
        pos = self._norm.find(norm_target)
        if pos < 0:
            return None
        end = pos + len(norm_target) - 1
        # Endpoints must land on real runs, not the synthetic separator.
        if self._keys[pos] is None or self._keys[end] is None:
            return None
        return AnchorSpan(
            start_key=self._keys[pos],
            start_char=self._chars[pos],
            end_key=self._keys[end],
            end_char=self._chars[end],
        )

    def closest(self, target: str, window_pad: int = 30) -> tuple[float, str]:
        """Best-effort closest match, for diagnostics when ``find`` fails.

        Returns ``(similarity, snippet)`` where *snippet* is the region of
        the original document text most similar to the normalised target.
        Uses :class:`difflib.SequenceMatcher` — for reporting only, never
        for placement.
        """
        norm_target_full, _ = normalize_with_map(target)
        norm_target = norm_target_full.strip()
        if not norm_target or not self._norm:
            return 0.0, ""
        sm = difflib.SequenceMatcher(None, norm_target, self._norm, autojunk=False)
        match = sm.find_longest_match(0, len(norm_target), 0, len(self._norm))
        if match.size == 0:
            return 0.0, ""
        # Score the BEST LOCAL window, not the whole document: align a
        # target-length window of the document to the longest common block
        # (so match.a inside the target lines up with match.b in the doc),
        # then compare. A whole-document ratio is meaninglessly tiny for a
        # long target even when its region matches well.
        tlen = len(norm_target)
        win_start = max(0, match.b - match.a)
        win_end = min(len(self._norm), win_start + tlen)
        window = self._norm[win_start:win_end]
        score = difflib.SequenceMatcher(
            None, norm_target, window, autojunk=False
        ).ratio()
        # Snippet for display: padded view around the matched region.
        lo = max(0, match.b - window_pad)
        hi = min(len(self._norm), match.b + match.size + window_pad)
        snippet = self._norm[lo:hi]
        return score, snippet
