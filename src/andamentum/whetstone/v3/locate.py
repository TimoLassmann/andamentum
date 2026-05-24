"""The reliability spine: locate an LLM-emitted span in the source text.

Every claim and every finding the LLM produces is verified by string-matching
it against the source. A span that can't be located is a hallucination (or a
paraphrase) and is rejected.

This module is now a thin shim over ``andamentum.core.text_match.find_span``
— the single canonical answer to "is this verbatim in the source." Before
consolidation there were five separate implementations across the codebase
with materially different normalisation rules; see
docs/plans/2026-05-24-string-match-consolidation.md.

Behaviour is preserved (markdown stripping + case folding + whitespace
collapsing) and gains smart-quote stripping from the canonical defaults —
the smoke logs show LLMs occasionally wrap quotes in curly “…” that would
have silently failed the gate before. The ≤2-attempt agent re-quote on a
miss lives in the verify *nodes*, not here — this module only ever answers
"is it there, and where?".
"""

from __future__ import annotations

from andamentum.core.text_match import find_span


def locate(
    quote: str, source: str, *, within: tuple[int, int] | None = None
) -> tuple[int, int] | None:
    """Locate *quote* in *source*; return its ``[start, end)`` char span in
    ORIGINAL source coordinates, or ``None`` if not present.

    Tries byte-exact substring first, then exact substring in normalised
    space (markdown markers dropped, smart quotes stripped from edges,
    case folded, whitespace collapsed). No fuzzy matching here — a miss
    is a real miss and the caller's agent retry loop handles it.

    When *within* ``(start, end)`` is given, only that slice of the source
    is searched; offsets are still returned in whole-source coordinates.
    """
    match = find_span(quote, source, within=within, fuzzy="off")
    if match is None:
        return None
    return match.start, match.end


def is_present(
    quote: str, source: str, *, within: tuple[int, int] | None = None
) -> bool:
    """True if *quote* can be located in *source* (the hallucination gate)."""
    return locate(quote, source, within=within) is not None
