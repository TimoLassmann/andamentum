"""Structural split: parse markdown headers, build a section tree.

This is stage 1 of the structural-first chunker. We rely on markdown
heading lines (`# `, `## `, etc.) to identify section boundaries — for
academic papers and clean web articles this gets us 80%+ of the way
without any LLM call.

Outputs are dataclasses with absolute character spans into the source.
The orchestrator turns these into final ``Unit`` objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ATX-style markdown headers — leading whitespace allowed but rare in well-
# formed markdown. Setext (underline) headers are not handled here because
# trafilatura/docling output is consistently ATX.
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)


@dataclass
class Heading:
    """One markdown heading occurrence in source."""

    start: int  # absolute char offset of the `#` character
    line_end: int  # offset of the newline after the heading
    level: int  # 1–6
    title: str  # heading text without the `#` prefix


@dataclass
class Section:
    """A semantic section: a heading and the body that follows it.

    Sections nest: a level-2 section contains all level-3+ headings up to
    the next level-2-or-shallower heading. ``children`` lists those nested
    sections; the section's body INCLUDES its children's text (we do not
    pre-split — the orchestrator decides).
    """

    start: (
        int  # offset of the heading's `#` character (so includes the `## title` line)
    )
    end: int  # exclusive end offset (= start of next sibling/parent, or len(source))
    level: int
    title: str
    children: list["Section"] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.end - self.start


def find_headings(source: str) -> list[Heading]:
    """Find all ATX markdown headings in `source`."""
    out: list[Heading] = []
    for m in _HEADING_RE.finditer(source):
        line_end = source.find("\n", m.end())
        if line_end == -1:
            line_end = len(source)
        out.append(
            Heading(
                start=m.start(),
                line_end=line_end,
                level=len(m.group(1)),
                title=m.group(2).strip(),
            )
        )
    return out


def build_section_tree(source: str, headings: list[Heading]) -> list[Section]:
    """Build a nested section tree from a flat list of headings.

    A section's ``end`` is the start of the next heading at the same or
    shallower level (or len(source) if none). Children are sections whose
    level is strictly deeper, until that next sibling/parent.

    If the source has content BEFORE the first heading, we don't manufacture
    a fake section for it — the orchestrator handles preamble separately.
    """
    if not headings:
        return []

    sections: list[Section] = []
    n = len(headings)
    for i, h in enumerate(headings):
        # Find the next heading at this level or shallower (closes this section)
        end = len(source)
        for j in range(i + 1, n):
            if headings[j].level <= h.level:
                end = headings[j].start
                break
        sections.append(Section(start=h.start, end=end, level=h.level, title=h.title))

    # Nest: walk the flat list and attach deeper-level sections as children
    # of the most recent shallower-level section.
    return _nest(sections)


def _nest(flat: list[Section]) -> list[Section]:
    """Convert flat list of Section spans into a forest by level."""
    roots: list[Section] = []
    stack: list[Section] = []  # currently-open section parents
    for sec in flat:
        # Pop any stack entries that don't contain this section (level >= sec.level)
        while stack and stack[-1].level >= sec.level:
            stack.pop()
        if stack:
            stack[-1].children.append(sec)
        else:
            roots.append(sec)
        stack.append(sec)
    return roots


def section_iter_leaves(section: Section) -> list[Section]:
    """Recursive: return all leaf (no-children) sections under `section`."""
    if not section.children:
        return [section]
    out: list[Section] = []
    for child in section.children:
        out.extend(section_iter_leaves(child))
    return out


def split_section_recursively(section: Section, target_max: int) -> list[Section]:
    """Try to split a section into pieces ≤ target_max chars by recursing.

    If the section is small enough, returns [section]. Otherwise descends
    into its children — emitting each child as its own piece (recursively)
    plus a synthetic "header + intro" piece for the part of `section`
    BEFORE its first child (if any text exists there).
    """
    if section.length <= target_max:
        return [section]

    if not section.children:
        # No structural breakdown possible — caller (orchestrator) must use
        # the embedding-based split as a fallback.
        return [section]

    pieces: list[Section] = []

    # Synthesize an "intro" piece for content between this section's heading
    # and its first child (if non-trivial).
    first_child_start = section.children[0].start
    intro_end = first_child_start
    if intro_end > section.start:
        intro_text_len = intro_end - section.start
        if intro_text_len > 0:
            pieces.append(
                Section(
                    start=section.start,
                    end=intro_end,
                    level=section.level,
                    title=section.title,  # inherit
                )
            )

    for child in section.children:
        pieces.extend(split_section_recursively(child, target_max))

    return pieces
