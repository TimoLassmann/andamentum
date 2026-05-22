"""The reliability spine: locate an LLM-emitted span in the source text.

Every claim and every finding the LLM produces is verified by string-matching
it against the source. A span that can't be located is a hallucination (or a
paraphrase) and is rejected. This is the same principle the docx renderer uses
to anchor comments — here generalised to plain source text.

Matching is **normalised** (markdown markers stripped, lowercased, whitespace
collapsed) so a markdown-flavoured quote matches plain prose, then **exact** in
the normalised space — no fuzzy thresholds. Matching is **origin-section
scoped** when the caller knows where the span came from (extraction is
per-section), which removes occurrence ambiguity for short quotes.

Pure and deterministic. The ≤3-attempt agent re-quote on a miss lives in the
verify *nodes*, not here — this module only ever answers "is it there, and
where?".
"""

from __future__ import annotations

from ..docx.anchor import normalize_with_map


def locate(
    quote: str, source: str, *, within: tuple[int, int] | None = None
) -> tuple[int, int] | None:
    """Locate *quote* in *source*; return its ``[start, end)`` char span in the
    ORIGINAL source coordinates, or ``None`` if not present.

    Normalises both sides and matches exactly in normalised space. When
    *within* ``(start, end)`` is given, only that slice of the source is
    searched (and offsets are returned in whole-source coordinates) — pass the
    origin section's range to disambiguate short quotes.
    """
    norm_q, _ = normalize_with_map(quote)
    norm_q = norm_q.strip()
    if not norm_q:
        return None

    base = 0
    seg = source
    if within is not None:
        base = max(0, within[0])
        seg = source[base : within[1]]

    norm_seg, idx_map = normalize_with_map(seg)
    pos = norm_seg.find(norm_q)
    if pos < 0:
        return None
    end = pos + len(norm_q) - 1
    return base + idx_map[pos], base + idx_map[end] + 1


def is_present(
    quote: str, source: str, *, within: tuple[int, int] | None = None
) -> bool:
    """True if *quote* can be located in *source* (the hallucination gate)."""
    return locate(quote, source, within=within) is not None
