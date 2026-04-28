"""Extract cross-references (``Figure 3``, ``Table 2``, etc.) from sections.

Phase 1 covers four anchor types — the ones that show up in essentially
every academic paper:
  • figure   → ``Figure 3``, ``Fig. 3``
  • table    → ``Table 2``, ``Tab. 2``
  • section  → ``Section 4.1``, ``Sec. 4.1``, ``§4.1``
  • equation → ``Equation 5``, ``Eq. 5``, ``Eqn 5``

The downstream synthesiser cross-checks every reference against the
markdown for an actual anchor — a heading line, a figure caption, an
equation marker. Broken cross-references are a high-confidence finding.
"""

from __future__ import annotations

import re

from .types import CrossReference, SectionRef

# Each pattern's group 1 is the target identifier (number or section path).
_REF_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "figure",
        re.compile(r"\b(?:Figure|Figures|Fig\.?|Figs\.?)\s+(\d+(?:\.\d+)*)\b"),
    ),
    (
        "table",
        re.compile(r"\b(?:Table|Tables|Tab\.?|Tabs\.?)\s+(\d+(?:\.\d+)*)\b"),
    ),
    (
        "section",
        re.compile(r"(?:\b(?:Section|Sections|Sec\.?|Secs\.?)\s+|§\s*)(\d+(?:\.\d+)*)\b"),
    ),
    (
        "equation",
        re.compile(r"\b(?:Equation|Equations|Eq\.?|Eqs?\.?|Eqn\.?|Eqns?\.?)\s+(\d+(?:\.\d+)*)\b"),
    ),
]


def extract_cross_references(sections: list[SectionRef]) -> list[CrossReference]:
    """Walk every section, extract every recognised cross-reference."""
    out: list[CrossReference] = []
    for section in sections:
        for kind, pattern in _REF_PATTERNS:
            for m in pattern.finditer(section.text):
                out.append(
                    CrossReference(
                        raw=m.group(0),
                        kind=kind,  # type: ignore[arg-type]
                        target=m.group(1),
                        section_id=section.id,
                        char_start=m.start(),
                        char_end=m.end(),
                    )
                )
    return out


# ── Anchor detection (used by deterministic_findings) ────────────────────


# A figure / table caption is conventionally a line that BEGINS with the
# label, e.g. "Figure 3: Caption text" or "Figure 3. Caption text".
_FIGURE_ANCHOR_RE = re.compile(r"^\s*Figure\s+(\d+(?:\.\d+)*)[:.]", re.MULTILINE)
_TABLE_ANCHOR_RE = re.compile(r"^\s*Table\s+(\d+(?:\.\d+)*)[:.]", re.MULTILINE)
# An equation anchor is rendered as a label like "(3)" at end of an equation
# block, or — in markdown — sometimes just the number on its own line.
_EQUATION_ANCHOR_RE = re.compile(r"\(\s*(\d+(?:\.\d+)*)\s*\)\s*$", re.MULTILINE)


def find_figure_anchors(sections: list[SectionRef]) -> set[str]:
    """Return the set of figure numbers anchored anywhere in the document."""
    out: set[str] = set()
    for section in sections:
        out.update(m.group(1) for m in _FIGURE_ANCHOR_RE.finditer(section.text))
    return out


def find_table_anchors(sections: list[SectionRef]) -> set[str]:
    """Return the set of table numbers anchored anywhere in the document."""
    out: set[str] = set()
    for section in sections:
        out.update(m.group(1) for m in _TABLE_ANCHOR_RE.finditer(section.text))
    return out


def find_section_anchors(sections: list[SectionRef]) -> set[str]:
    """Return the set of section numbers (e.g. "2", "2.1") declared in headings.

    Looks for headings that start with a digit (e.g. "## 2 Background",
    "## 2.1 Foo"). Not perfect — some authors don't number their sections.
    """
    out: set[str] = set()
    heading_with_number = re.compile(r"^\s*#{1,6}\s+(\d+(?:\.\d+)*)\b")
    for section in sections:
        for line in section.text.splitlines():
            m = heading_with_number.match(line)
            if m:
                out.add(m.group(1))
    return out


def find_equation_anchors(sections: list[SectionRef]) -> set[str]:
    """Return the set of equation numbers anchored anywhere.

    This is intentionally loose — equation anchors are the hardest to
    detect reliably, so the synthesiser treats missing equation anchors
    as a low-severity finding.
    """
    out: set[str] = set()
    for section in sections:
        out.update(m.group(1) for m in _EQUATION_ANCHOR_RE.finditer(section.text))
    return out
