"""Phrase matching over normalised text, shared by the loss attribution.

``normalise_for_match`` folds case, accents and punctuation and reads ``1,000``
as ``1000``; ``contains_phrase`` asks whether a phrase occurs as whole words.
"""

from __future__ import annotations

import re
import unicodedata

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NON_WORD = re.compile(r"[^0-9a-z]+")


def normalise_for_match(text: str) -> str:
    """Case-, accent- and punctuation-folded words joined by single spaces."""
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in folded if not unicodedata.combining(c)).casefold()
    folded = _THOUSANDS.sub("", folded)
    return " ".join(_NON_WORD.sub(" ", folded).split())


def contains_phrase(text: str, phrase: str) -> bool:
    """Does *phrase* occur in *text* as whole words (both normalised)?"""
    wanted = normalise_for_match(phrase)
    if not wanted:
        raise ValueError(f"the phrase {phrase!r} has no letters or digits")
    return f" {wanted} " in f" {normalise_for_match(text)} "


__all__ = ["contains_phrase", "normalise_for_match"]
