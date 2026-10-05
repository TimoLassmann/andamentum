"""Finding a quoted passage in a document: how every model claim is verified.

A model that says "this passage" must quote it, and the quote must occur in
the text (whitespace-insensitive, one paragraph). A quote that does not occur
is never acted on.
"""

from __future__ import annotations

import re

from ..kernel import DocumentView, Paragraph

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalise(text: str) -> str:
    """Whitespace collapsed and typographic quotes straightened."""
    return " ".join(text.translate(_QUOTES).split())


def _pattern(quote: str) -> re.Pattern[str] | None:
    words = normalise(quote).split(" ")
    if not words or words == [""]:
        return None
    return re.compile(r"\s+".join(re.escape(w) for w in words))


def find_in_paragraph(paragraph: Paragraph, quote: str) -> tuple[int, int] | None:
    """Offsets of *quote* inside the paragraph text, or None."""
    pattern = _pattern(quote)
    if pattern is None:
        return None
    found = pattern.search(paragraph.text.translate(_QUOTES))
    return None if found is None else (found.start(), found.end())


def find_in_view(view: DocumentView, quote: str) -> tuple[Paragraph, int, int] | None:
    """The first paragraph holding *quote* with its offsets, or None."""
    for paragraph in view.paragraphs:
        span = find_in_paragraph(paragraph, quote)
        if span is not None:
            return paragraph, span[0], span[1]
    return None


def occurs_in(text: str, quote: str) -> bool:
    """Whether *quote* occurs in *text* (whitespace-insensitive)."""
    pattern = _pattern(quote)
    return pattern is not None and pattern.search(text.translate(_QUOTES)) is not None
