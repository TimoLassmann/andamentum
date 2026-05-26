"""Unified string-matching: locate a target string as a span in a source.

The single canonical answer to "is this target verbatim in this source, and
if so, where?" The previous codebase had FIVE different implementations of
this with materially different normalisation rules — see
docs/.internal/plans/2026-05-24-string-match-consolidation.md for the inventory.

The contract:

- ``normalize_for_match(text, ...)`` returns ``(normalized, idx_map)`` where
  ``idx_map[i]`` is the offset in the ORIGINAL text of the character that
  produced ``normalized[i]``. This lets matches found in normalised space
  be mapped back to original source coordinates without ambiguity.
- ``find_span(target, source, ...)`` tries three tiers in order: byte-exact
  substring, normalised-exact substring, optional fuzzy fallback. Returns
  a ``Match`` with the source coordinates and the tier that succeeded, or
  ``None`` on miss.

The canonical default — markdown stripping + smart-quote stripping +
case folding + whitespace collapsing, fuzzy off — is the system's
answer to "is this verbatim." Any callsite that diverges from these
defaults must do so explicitly via kwargs, making the divergence
visible.

This module has no andamentum dependencies and is unit-testable in
isolation. ``rapidfuzz`` is imported lazily so the module loads even
when it isn't installed (fuzzy="rapidfuzz" then errors at call time).
"""

from __future__ import annotations

import difflib
from typing import Literal, NamedTuple

# Markdown markers dropped entirely during normalisation. Aggressive on
# purpose: these never carry meaning for *locating* prose, and dropping
# them lets a markdown-flavoured quote match plain text. Exposed as a
# public constant so callers with their own multi-segment indexing
# (e.g. whetstone/docx/anchor.py's DocIndex) can share the exact same
# character class without re-declaring it — the "rules cannot drift"
# guarantee.
MARKDOWN_MARKERS: frozenset[str] = frozenset("#*_`[]~")
_MARKDOWN_MARKERS = MARKDOWN_MARKERS  # internal alias for backward compat

# Quote-mark characters stripped from the *edges* of the target when
# ``strip_quotes=True``. ASCII " and ' plus the four Unicode smart-quote
# variants. End-strip only (not stripped from the middle) — LLMs wrap
# verbatim excerpts in these but the underlying source doesn't.
_QUOTE_CHARS = "\"'“”‘’"

FuzzyBackend = Literal["off", "rapidfuzz", "sequence"]


class Match(NamedTuple):
    """A successful match. ``start`` / ``end`` are half-open offsets in
    the ORIGINAL source (not the normalised form). ``method`` names the
    tier that succeeded; ``score`` is 1.0 for exact and normalised
    matches, the actual fuzzy ratio (0.0-1.0) for fuzzy matches."""

    start: int
    end: int
    method: Literal["exact", "normalized", "fuzzy"]
    score: float


def normalize_for_match(
    text: str,
    *,
    strip_markdown: bool = True,
    strip_quotes: bool = True,
    fold_case: bool = True,
    collapse_whitespace: bool = True,
) -> tuple[str, list[int]]:
    """Normalise *text* for matching; return ``(normalized, idx_map)``.

    ``idx_map[i]`` is the index in the ORIGINAL ``text`` of the character
    that produced ``normalized[i]``. A match found in normalised space
    can be mapped back to original coordinates via this array.

    Normalisation steps (each toggleable):

    - ``strip_quotes``: strip quote characters from the *edges* of the
      text. Only edge-strip — quotes in the middle of the text are
      preserved (they may carry semantic content there).
    - ``strip_markdown``: drop markdown markers (``# * _ ` [ ] ~``)
      everywhere. Aggressive on purpose.
    - ``fold_case``: lowercase non-space characters.
    - ``collapse_whitespace``: collapse every run of whitespace
      (including newlines) to a single space; strip leading/trailing
      whitespace.

    All four default-on, which is the canonical contract.
    """
    if strip_quotes:
        text = text.strip(_QUOTE_CHARS)

    out: list[str] = []
    idx_map: list[int] = []
    # Track whether the previous emitted character was a space so we can
    # collapse runs. Start as True so leading whitespace is dropped when
    # collapse_whitespace is on.
    prev_space = collapse_whitespace

    for i, ch in enumerate(text):
        if strip_markdown and ch in _MARKDOWN_MARKERS:
            continue
        if ch.isspace() and collapse_whitespace:
            if prev_space:
                continue
            out.append(" ")
            idx_map.append(i)
            prev_space = True
        else:
            out.append(ch.lower() if fold_case else ch)
            idx_map.append(i)
            prev_space = False

    if collapse_whitespace:
        while out and out[-1] == " ":
            out.pop()
            idx_map.pop()

    return "".join(out), idx_map


def find_span(
    target: str,
    source: str,
    *,
    within: tuple[int, int] | None = None,
    try_exact: bool = True,
    try_normalized: bool = True,
    fuzzy: FuzzyBackend = "off",
    fuzzy_threshold: float = 0.85,
    fuzzy_window_min_ratio: float = 0.7,
    fuzzy_window_max_ratio: float = 1.5,
    strip_markdown: bool = True,
    strip_quotes: bool = True,
    fold_case: bool = True,
    collapse_whitespace: bool = True,
) -> Match | None:
    """Locate *target* in *source*; return a ``Match`` or ``None``.

    Tries up to three tiers in order:

    1. **exact** (``try_exact=True``): byte-identical substring. Useful
       when the caller needs to preserve a verbatim contract (the
       chunker's byte-identical units, for example).
    2. **normalized** (``try_normalized=True``): exact substring in the
       normalised forms. Normalisation toggles control what's folded.
    3. **fuzzy** (``fuzzy != "off"``): sliding-window fuzzy match using
       the named backend (``"rapidfuzz"`` for token_set_ratio,
       ``"sequence"`` for ``difflib.SequenceMatcher``). Accepts if score
       ≥ ``fuzzy_threshold``.

    ``within`` scopes the search to ``[start, end)`` of *source*;
    offsets in the returned ``Match`` are still in whole-source
    coordinates.

    The fuzzy window slides over lengths
    ``[fuzzy_window_min_ratio * len(target), fuzzy_window_max_ratio * len(target)]``
    — matches the chunker's pre-consolidation behaviour.
    """
    if not target:
        return None

    base = 0
    scoped_source = source
    if within is not None:
        base = max(0, within[0])
        scoped_source = source[base : within[1]]

    # Tier 1: byte-exact substring.
    if try_exact:
        pos = scoped_source.find(target)
        if pos >= 0:
            return Match(
                start=base + pos,
                end=base + pos + len(target),
                method="exact",
                score=1.0,
            )

    # Tier 2: normalised-exact substring.
    norm_kwargs = dict(
        strip_markdown=strip_markdown,
        strip_quotes=strip_quotes,
        fold_case=fold_case,
        collapse_whitespace=collapse_whitespace,
    )
    if try_normalized:
        norm_target, _ = normalize_for_match(target, **norm_kwargs)
        if not norm_target:
            return None
        norm_source, idx_map = normalize_for_match(scoped_source, **norm_kwargs)
        pos = norm_source.find(norm_target)
        if pos >= 0:
            end_idx = pos + len(norm_target) - 1
            return Match(
                start=base + idx_map[pos],
                end=base + idx_map[end_idx] + 1,
                method="normalized",
                score=1.0,
            )

    # Tier 3: fuzzy fallback.
    if fuzzy != "off":
        result = _fuzzy_match(
            target,
            scoped_source,
            backend=fuzzy,
            threshold=fuzzy_threshold,
            window_min_ratio=fuzzy_window_min_ratio,
            window_max_ratio=fuzzy_window_max_ratio,
        )
        if result is not None:
            start, end, score = result
            return Match(
                start=base + start,
                end=base + end,
                method="fuzzy",
                score=score,
            )

    return None


def is_present(target: str, source: str, **kwargs) -> bool:
    """True if *target* can be located in *source* via ``find_span``."""
    return find_span(target, source, **kwargs) is not None


# ── Fuzzy backends ─────────────────────────────────────────────────────


def _fuzzy_match(
    target: str,
    source: str,
    *,
    backend: FuzzyBackend,
    threshold: float,
    window_min_ratio: float,
    window_max_ratio: float,
) -> tuple[int, int, float] | None:
    """Sliding-window fuzzy match; returns (start, end, score) or None."""
    tlen = len(target)
    if tlen == 0 or len(source) == 0:
        return None

    window_min = max(1, int(tlen * window_min_ratio))
    window_max = min(len(source), int(tlen * window_max_ratio) + 1)
    if window_min > window_max:
        window_min = window_max

    if backend == "rapidfuzz":
        scorer = _rapidfuzz_token_set_ratio
    elif backend == "sequence":
        scorer = _sequence_ratio
    else:
        return None

    best_score = 0.0
    best_start = -1
    best_window_len = 0
    for window_len in range(window_min, window_max + 1):
        for start in range(0, len(source) - window_len + 1):
            candidate = source[start : start + window_len]
            score = scorer(target, candidate)
            if score > best_score:
                best_score = score
                best_start = start
                best_window_len = window_len
    if best_start >= 0 and best_score >= threshold:
        return best_start, best_start + best_window_len, best_score
    return None


def _rapidfuzz_token_set_ratio(a: str, b: str) -> float:
    """Lazy-import wrapper around rapidfuzz.fuzz.token_set_ratio; returns
    a 0.0-1.0 score (rapidfuzz returns 0-100 natively)."""
    try:
        from rapidfuzz import fuzz
    except ImportError as e:
        raise RuntimeError(
            "fuzzy='rapidfuzz' requested but rapidfuzz is not installed"
        ) from e
    return fuzz.token_set_ratio(a, b) / 100.0


def _sequence_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
