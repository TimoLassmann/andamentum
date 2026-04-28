"""Extract acronym definitions and usages from sections.

Phase 1 focuses on **acronym definitions** because they're unusually
regular: ``Some Long Phrase (SLP)`` is a near-universal academic
convention. Detection heuristic:

  • Find a parenthesised UPPERCASE+digit token of length 2-8.
  • Walk backwards from the opening paren and capture as many capitalised
    or hyphenated words as the acronym has letters (heuristically, this
    is the "expansion").
  • Discard if the candidate expansion's initials don't loosely match the
    acronym's letters.

Then collect **usages**: every standalone occurrence of any defined
acronym. The downstream synthesiser uses (definitions, usages) to flag
redefinitions and missing definitions.

Defined-twice-with-different-expansion is a high-value finding; missing
definitions are a stylistic finding (lower severity).
"""

from __future__ import annotations

import re

from .types import SectionRef, TermDefinition, TermGlossary, TermUsage

# An acronym candidate inside parens, optionally with hyphens/digits.
# Match e.g. (MCC), (POET), (3D-CNN), (E.S.).
_ACRONYM_PAREN_RE = re.compile(
    r"\(\s*([A-Z][A-Za-z0-9.\-]{1,15})\s*\)"
)
# What "looks like an expansion word" — capitalised, possibly hyphenated.
_EXPANSION_WORD_RE = re.compile(
    r"\b([A-Z][A-Za-z\-']*)\b"
)


def extract_term_glossary(sections: list[SectionRef]) -> TermGlossary:
    """Extract acronym definitions and usages across the document."""
    glossary = TermGlossary()
    for section in sections:
        glossary.definitions.extend(_find_definitions(section))

    # Now collect usages of every defined acronym.
    defined_terms = sorted({d.term for d in glossary.definitions}, key=len, reverse=True)
    if defined_terms:
        # Build one regex that matches any defined term as a standalone word.
        pattern = r"\b(" + "|".join(re.escape(t) for t in defined_terms) + r")\b"
        usage_re = re.compile(pattern)
        for section in sections:
            for m in usage_re.finditer(section.text):
                glossary.usages.append(
                    TermUsage(
                        term=m.group(1),
                        section_id=section.id,
                        char_start=m.start(),
                        char_end=m.end(),
                    )
                )

    return glossary


def _find_definitions(section: SectionRef) -> list[TermDefinition]:
    """Find ``Long Phrase (LP)`` patterns in one section."""
    out: list[TermDefinition] = []
    text = section.text
    for m in _ACRONYM_PAREN_RE.finditer(text):
        acronym = m.group(1)
        # Skip parens that are clearly NOT acronym defs — short numbers,
        # citation lists, etc. (Heuristic: at least 2 alpha chars.)
        alpha_only = re.sub(r"[^A-Za-z]", "", acronym)
        if len(alpha_only) < 2:
            continue
        expansion = _capture_expansion(text, m.start(), len(alpha_only))
        if expansion is None:
            continue
        if not _initials_match(expansion, alpha_only):
            continue
        out.append(
            TermDefinition(
                term=acronym,
                expansion=expansion,
                section_id=section.id,
                char_start=m.start(),
                char_end=m.end(),
            )
        )
    return out


def _capture_expansion(text: str, paren_start: int, n_letters: int) -> str | None:
    """Walk backwards from `paren_start`, capture up to `n_letters * 2` words.

    We allow a generous window because expansion words can include short
    connectives (``of``, ``the``) that don't contribute initials.
    """
    # Look at up to ~80 chars before the paren — that's enough room for any
    # reasonable expansion.
    window_start = max(0, paren_start - 80)
    window = text[window_start:paren_start].rstrip()
    # Capture all capitalised words in the window.
    words = _EXPANSION_WORD_RE.findall(window)
    if len(words) < n_letters:
        return None
    # Keep the LAST n_letters * 2 words (allowing room for connectives).
    candidate_words = words[-(n_letters * 2) :]
    if not candidate_words:
        return None
    # Find where the first candidate word starts in the window (so we can
    # return a clean substring of the original text).
    first = candidate_words[0]
    first_idx = window.rfind(first)
    if first_idx == -1:
        return " ".join(candidate_words)
    return window[first_idx:].strip()


def _initials_match(expansion: str, acronym: str) -> bool:
    """Loose check: do the capitalised words' initials cover the acronym?"""
    initials = "".join(w[0] for w in _EXPANSION_WORD_RE.findall(expansion))
    if not initials:
        return False
    # The acronym should appear (in order, but not contiguous) inside the
    # initials. This tolerates connectives like "of" / "the" that the user
    # may have skipped in the acronym.
    i = 0
    target = acronym.upper()
    for ch in initials.upper():
        if i < len(target) and ch == target[i]:
            i += 1
    return i == len(target)
