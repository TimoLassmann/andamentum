"""The reliability spine: locate an LLM-emitted span in the source text.

Every claim and every finding the LLM produces is verified by string-matching
it against the source. A span that can't be located is a hallucination (or a
paraphrase) and is rejected.

This module is now a thin shim over ``andamentum.core.text_match.find_span``
— the single canonical answer to "is this verbatim in the source." Before
consolidation there were five separate implementations across the codebase
with materially different normalisation rules; see
internal plans/2026-05-24-string-match-consolidation.md.

Behaviour is preserved (markdown stripping + case folding + whitespace
collapsing) and gains smart-quote stripping from the canonical defaults —
the smoke logs show LLMs occasionally wrap quotes in curly “…” that would
have silently failed the gate before. The ≤2-attempt agent re-quote on a
miss lives in the verify *nodes*, not here — this module only ever answers
"is it there, and where?".

``anchor_span`` answers that same question in the ``DocumentModel``'s own
terms (offsets plus the section they fall in), which is the form every
finding-producing mode needs. It is here rather than in each mode's verify
step because "where is this text" is precisely this module's job, and having
it in one place keeps the section-resolution rule identical across modes.
"""

from __future__ import annotations

from andamentum.core.text_match import find_span

from .model import DocumentModel, Span


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


def anchor_span(quote: str, model: DocumentModel) -> Span | None:
    """Locate *quote* in the document and return its ``Span``, or ``None``.

    The one step every finding-producing mode takes before it can keep a
    finding: locate the quote, then work out which section the offset landed
    in. It is searched against the WHOLE source rather than any one section
    because an agent reports what it read via ``read_section`` / ``search_paper``
    and does not reliably track which section id that came from — so the
    section is recovered from the offset instead of trusted from the model.

    ``None`` means the quote is not in the source, i.e. the finding attached to
    it is unanchorable. What to do about that is the caller's policy (every
    caller today drops it and logs); this function only answers where the text
    is.

    ``section_id`` falls back to ``"?"`` when the offset lands outside every
    section — possible for text in the gap before the first heading. The span
    is still correct and still usable; only the label is unknown.
    """
    location = locate(quote, model.source)
    if location is None:
        return None
    start, end = location
    section = next((s for s in model.sections if s.start <= start < s.end), None)
    return Span(section_id=section.id if section else "?", start=start, end=end)
