"""Extract citations and reference entries from a chunked document.

Two citation styles handled in v1:
  • numeric: ``[12]``, ``[1, 2, 3]``, ``[1-3]``  (academic standard)
  • pandoc:  ``[@smith2020]``, ``[@a2020; @b2021]``  (markdown convention)

Author-year citations like ``(Smith et al., 2020)`` are intentionally
deferred — they're harder to disambiguate from regular parenthetical
prose and the bulk of the user's papers use one of the two supported
styles.

Reference entries are discovered by finding a section whose title matches
``References | Bibliography | Works Cited`` and parsing the entries
within. For numeric style, an entry is a paragraph starting with ``[N]``
or ``N.``. For pandoc style, references are usually rendered separately
and we treat any line beginning with ``[@key]`` as defining ``key``.
"""

from __future__ import annotations

import re

from .types import (
    CitationGraph,
    CitationKey,
    CitationOccurrence,
    SectionRef,
)

# Numeric in-text: "[12]", "[1, 2, 3]", "[1-3]"
_NUMERIC_RE = re.compile(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]")
# Pandoc in-text: "[@key1; @key2]"  — we capture each @key individually
_PANDOC_RE = re.compile(r"@([A-Za-z][\w:.\-]+)")
# Reference entry (numeric): "[N] author..." or "N. author..." or "- [N] author..."
# The optional leading bullet "-" is what docling produces for
# bibliographies extracted from PDFs (each entry rendered as a list item).
_NUMERIC_REF_LINE_RE = re.compile(
    r"^\s*-?\s*(?:\[(\d+)\]|(\d+)\.)\s+(.+)$",
    re.MULTILINE,
)
# Pandoc-style reference line: "[@key]: ..." (Pandoc citation list style)
_PANDOC_REF_LINE_RE = re.compile(
    r"^\s*\[@([A-Za-z][\w:.\-]+)\]:\s*(.+)$",
    re.MULTILINE,
)
# Sections we treat as the references list.
_REFERENCES_TITLE_RE = re.compile(
    r"^\s*(references|bibliography|works\s+cited|literature\s+cited)\s*$",
    re.IGNORECASE,
)


def extract_citations(sections: list[SectionRef]) -> CitationGraph:
    """Walk every section, collect citation occurrences and the references list."""
    graph = CitationGraph()
    references_sections = _find_references_sections(sections)
    graph.references_section_ids = [s.id for s in references_sections]
    references_id_set = set(graph.references_section_ids)

    for section in sections:
        # Skip in-text citation extraction inside ANY references section —
        # the [N] markers there are entry headers, not in-text cites.
        if section.id in references_id_set:
            continue
        graph.occurrences.extend(_find_in_text_citations(section))

    for ref_section in references_sections:
        graph.references_defined.update(_extract_reference_entries(ref_section))

    return graph


# ── References-section detection ────────────────────────────────────────


# A line that LOOKS like a reference entry header. Tolerates leading
# bullet markers ("- [N]") and either bracketed-or-dotted numerics
# ("[N]" or "N.").
_REF_ENTRY_LINE_RE = re.compile(r"^\s*-?\s*(?:\[\d+\]|\d+\.)\s+\S", re.MULTILINE)
# Same for pandoc-style entries ("[@key]: ...")
_PANDOC_REF_LINE_FAST_RE = re.compile(r"^\s*\[@[A-Za-z][\w:.\-]+\]:", re.MULTILINE)


def _find_references_sections(sections: list[SectionRef]) -> list[SectionRef]:
    """Find every section that constitutes the references list.

    Real papers have bibliographies that the chunker may have split across
    multiple section-level units. This walks forward from the first section
    titled ``References | Bibliography | …`` and includes every subsequent
    section whose body looks like reference continuations, stopping at the
    first non-references section.
    """
    out: list[SectionRef] = []
    in_references = False
    for section in sections:
        if not in_references:
            if _REFERENCES_TITLE_RE.match(section.title or ""):
                out.append(section)
                in_references = True
        else:
            if _looks_like_reference_continuation(section):
                out.append(section)
            else:
                # Some other section (Appendix, Supplemental, etc.) — stop.
                break
    return out


def _looks_like_reference_continuation(section: SectionRef) -> bool:
    """True iff the section's body is dominated by reference-entry lines.

    Heuristic: at least 2 reference-style lines AND those lines make up
    more than 30% of the section's non-empty lines. Tuned to catch real
    bibliography continuations (which are nearly 100% reference lines)
    without false-positive on prose sections that happen to contain the
    occasional ``[12]`` citation. The density check carries most of the
    load — 30% of lines being reference-headers is a strong signal.
    """
    text = section.text
    ref_line_count = (
        len(_REF_ENTRY_LINE_RE.findall(text))
        + len(_PANDOC_REF_LINE_FAST_RE.findall(text))
    )
    if ref_line_count < 2:
        return False
    non_empty_lines = sum(1 for line in text.splitlines() if line.strip())
    if non_empty_lines == 0:
        return False
    return ref_line_count / non_empty_lines > 0.3


def _find_in_text_citations(section: SectionRef) -> list[CitationOccurrence]:
    """Find ``[N]`` / ``[N, M]`` / ``[N-M]`` and ``[@key]`` occurrences."""
    out: list[CitationOccurrence] = []
    text = section.text

    # Numeric: expand "[1, 2-4]" into individual citation occurrences.
    for m in _NUMERIC_RE.finditer(text):
        body = m.group(1)
        for key_str in _expand_numeric_range(body):
            out.append(
                CitationOccurrence(
                    key=CitationKey(raw=m.group(0), key=key_str, style="numeric"),
                    section_id=section.id,
                    char_start=m.start(),
                    char_end=m.end(),
                )
            )

    # Pandoc: every "@key" inside square brackets in an in-text context.
    # We require the @ to be preceded by [ or ; (not just bare @ in text)
    # to avoid false positives like email addresses.
    for m in re.finditer(r"\[([^\]]*)\]", text):
        bracket_body = m.group(1)
        bracket_start = m.start()
        for km in _PANDOC_RE.finditer(bracket_body):
            key_str = km.group(1)
            # Skip if the @ is not the first char OR not preceded by ; / space
            prev = bracket_body[km.start() - 1] if km.start() > 0 else "["
            if prev not in (" ", ";", "[", ","):
                continue
            absolute_start = bracket_start + km.start() - 1  # include the @
            out.append(
                CitationOccurrence(
                    key=CitationKey(
                        raw=f"@{key_str}", key=key_str, style="pandoc"
                    ),
                    section_id=section.id,
                    char_start=absolute_start,
                    char_end=bracket_start + km.end(),
                )
            )
    return out


def _expand_numeric_range(body: str) -> list[str]:
    """Turn ``"1, 2-4, 7"`` into ``["1", "2", "3", "4", "7"]``.

    Tolerates en-dash and regular hyphen as range separator. Skips garbage
    silently — this is an extractor, not a validator.
    """
    out: list[str] = []
    for piece in re.split(r"\s*,\s*", body):
        piece = piece.strip()
        if not piece:
            continue
        m = re.match(r"^(\d+)\s*[\-–]\s*(\d+)$", piece)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if hi - lo > 100:  # sanity guard against absurd ranges
                continue
            for n in range(lo, hi + 1):
                out.append(str(n))
        elif piece.isdigit():
            out.append(piece)
    return out


def _extract_reference_entries(section: SectionRef) -> dict[str, str]:
    """Pull entries out of the references section.

    Tries numeric format first (``[N] ...`` or ``N. ...``); falls back to
    pandoc-style (``[@key]: ...``). Both styles can coexist in the same
    section but in practice only one is used per document.
    """
    out: dict[str, str] = {}
    for m in _NUMERIC_REF_LINE_RE.finditer(section.text):
        key = m.group(1) or m.group(2)
        body = m.group(3).strip()
        out[key] = _normalise_whitespace(body)
    for m in _PANDOC_REF_LINE_RE.finditer(section.text):
        out[m.group(1)] = _normalise_whitespace(m.group(2))
    return out


def _normalise_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()
