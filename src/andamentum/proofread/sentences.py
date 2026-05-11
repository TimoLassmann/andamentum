"""Cheap regex sentence segmentation that preserves char offsets.

Not as accurate as NLTK's Punkt, but deterministic, dependency-free, and good
enough for prose linting. We split on sentence-ending punctuation followed
by whitespace and a capital letter or end-of-string. Common abbreviations
("Dr.", "e.g.", "etc.") are guarded against to avoid spurious splits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc", "e.g", "i.e",
    "fig", "no", "vol", "pp", "p", "ca", "approx", "cf", "al",
    "u.s", "u.k", "ph.d", "m.d", "b.sc", "m.sc",
}

# Sentence terminator immediately followed by whitespace, then anything.
# We refine by checking the token preceding the terminator against
# _ABBREVIATIONS so "Dr. Smith" doesn't split.
_SPLIT = re.compile(r"([.!?])(\s+)")


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int  # inclusive
    end: int    # exclusive


def segment(text: str) -> list[Sentence]:
    """Return the sentences of ``text`` with char offsets into ``text``."""
    if not text:
        return []

    sentences: list[Sentence] = []
    cursor = 0
    for m in _SPLIT.finditer(text):
        # Token immediately before the terminator (for abbreviation check).
        preceding = text[cursor : m.start()]
        last_word_match = re.search(r"(\w+)\s*$", preceding)
        last_word = last_word_match.group(1).lower() if last_word_match else ""
        if last_word in _ABBREVIATIONS:
            continue

        end = m.end(1)  # include the punctuation, drop the whitespace
        snippet = text[cursor:end].strip()
        if snippet:
            # Re-locate the snippet inside [cursor:end] to preserve offsets
            stripped_start = cursor + (len(text[cursor:end]) - len(text[cursor:end].lstrip()))
            sentences.append(Sentence(text=snippet, start=stripped_start, end=end))
        cursor = m.end()

    # Trailing fragment (no terminator).
    if cursor < len(text):
        tail = text[cursor:].strip()
        if tail:
            stripped_start = cursor + (len(text[cursor:]) - len(text[cursor:].lstrip()))
            sentences.append(
                Sentence(text=tail, start=stripped_start, end=len(text))
            )

    return sentences


def index_for_offset(sentences: list[Sentence], offset: int) -> int:
    """Return the 0-based sentence index containing ``offset``.

    Falls back to the nearest sentence if the offset lies in inter-sentence
    whitespace. Returns -1 only if ``sentences`` is empty.
    """
    if not sentences:
        return -1
    for i, s in enumerate(sentences):
        if s.start <= offset < s.end:
            return i
    # Offset is in whitespace between sentences (or past the end). Snap to
    # the last sentence whose start is <= offset.
    last = 0
    for i, s in enumerate(sentences):
        if s.start <= offset:
            last = i
    return last
