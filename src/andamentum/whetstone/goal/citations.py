"""Deterministic reference-list consistency check. No LLM.

Checks that every in-text citation marker has a matching reference-list
entry, and that every reference-list entry is cited somewhere in the body —
structural cross-checking only. It never asks whether the cited work
actually supports the sentence; that needs a real bibliographic lookup
(PubMed, OpenAlex, a web search), which is separate, future work, not this
function.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from ..v3.digest import _CITATION_RES
from ..v3.model import Section

# Numeric-bracket and Pandoc [@key] only — see check_reference_consistency's
# docstring for why author-year is out of scope. Reusing _CITATION_RES rather
# than re-deriving the patterns keeps one source of truth for what a marker
# looks like.
_NUMERIC_RE = _CITATION_RES[0]
_PANDOC_RE = _CITATION_RES[1]

_REFERENCE_TITLES = frozenset(
    {
        "references",
        "reference list",
        "bibliography",
        "works cited",
        "literature cited",
    }
)

# A reference-list entry's leading marker: "1. ...", "[1] ...", "12) ...".
_REF_ENTRY_NUMERIC = re.compile(r"^\s*\[?(\d+)\]?[.)]?\s")
# "[@smith2020] ..." — the same key shape a Pandoc source repeats in its own
# bibliography when it has not been resolved to a rendered list.
_REF_ENTRY_KEY = re.compile(r"^\s*\[@([\w:.\-]+)\]")

ReferenceStyle = Literal[
    "numeric_bracket", "pandoc_key", "none_detected", "no_reference_section_found"
]


class ReferenceMismatch(BaseModel):
    """One marker that appears on only one side of the cited/listed divide."""

    marker: str
    kind: Literal["cited_not_listed", "listed_not_cited"]


class ReferenceConsistencyReport(BaseModel):
    style: ReferenceStyle
    mismatches: list[ReferenceMismatch] = Field(default_factory=list)
    reference_section_id: str | None = None

    @property
    def satisfied(self) -> bool:
        return self.style in ("numeric_bracket", "pandoc_key") and not self.mismatches


def _find_reference_section(sections: list[Section]) -> Section | None:
    for section in sections:
        if section.title.strip().casefold() in _REFERENCE_TITLES:
            return section
    return None


def _body_without_heading(section: Section) -> str:
    """Section text with its own heading line removed, if it repeats one.

    The same shape as ``v3.digest``'s heading strip: a markdown source
    repeats the heading as a leading ``#`` line; a Word source repeats it as
    a plain line equal to the section title. Either way it is not a
    reference entry and must not be parsed as one.
    """
    body = section.text.lstrip()
    first, newline, rest = body.partition("\n")
    if first.startswith("#") or first.strip() == section.title:
        return rest if newline else ""
    return body


def _expand_numeric_marker(marker: str) -> set[int]:
    """Every number a numeric-bracket marker names, ranges expanded.

    ``"[3-5]"`` -> ``{3, 4, 5}``; ``"[12, 13]"`` -> ``{12, 13}``;
    ``"[3-5, 8]"`` -> ``{3, 4, 5, 8}``.
    """
    tokens = re.findall(r"\d+|[,–-]", marker)
    numbers: set[int] = set()
    previous: int | None = None
    pending_range = False
    for token in tokens:
        if token.isdigit():
            value = int(token)
            if pending_range and previous is not None:
                lo, hi = sorted((previous, value))
                numbers.update(range(lo, hi + 1))
            else:
                numbers.add(value)
            previous = value
            pending_range = False
        else:
            pending_range = token in ("-", "–")
    return numbers


def _numeric_defined(body: str) -> dict[int, str]:
    defined: dict[int, str] = {}
    for line in body.split("\n"):
        match = _REF_ENTRY_NUMERIC.match(line)
        if match:
            defined.setdefault(int(match.group(1)), line.strip())
    return defined


def _key_defined(body: str) -> dict[str, str]:
    defined: dict[str, str] = {}
    for line in body.split("\n"):
        match = _REF_ENTRY_KEY.match(line)
        if match:
            defined.setdefault(match.group(1), line.strip())
    return defined


def check_reference_consistency(
    text: str, sections: list[Section]
) -> ReferenceConsistencyReport:
    """Cross-check in-text citation markers against the reference list.

    Structural only: a marker exists and is listed, or it does not, or it
    is listed but never cited. Never whether the cited work says what the
    sentence claims — that needs a real lookup against the source, which is
    separate, future work, not this function.

    Numeric-bracket and Pandoc ``[@key]`` markers only. Author-year markers
    (``(Smith et al., 2020)``) cannot be matched to a reference-list entry
    without fuzzy name/year matching, which risks the same claim-level
    judgement this check exists to avoid — a document using only that style
    reports ``"none_detected"``, not a false pass.

    When a manuscript mixes both marker styles (rare — usually a stray false
    match, such as a footnote number in brackets), the more frequent style is
    treated as the document's real convention and the other is not diffed.
    """
    reference_section = _find_reference_section(sections)
    if reference_section is None:
        return ReferenceConsistencyReport(style="no_reference_section_found")

    body_before = text[: reference_section.start]
    numeric_markers = _NUMERIC_RE.findall(body_before)
    pandoc_markers = _PANDOC_RE.findall(body_before)

    if not numeric_markers and not pandoc_markers:
        return ReferenceConsistencyReport(
            style="none_detected", reference_section_id=reference_section.id
        )

    reference_body = _body_without_heading(reference_section)
    mismatches: list[ReferenceMismatch] = []

    if len(numeric_markers) >= len(pandoc_markers):
        style: ReferenceStyle = "numeric_bracket"
        cited: set[int] = set()
        for marker in numeric_markers:
            cited |= _expand_numeric_marker(marker)
        defined = _numeric_defined(reference_body)
        for number in sorted(cited - defined.keys()):
            mismatches.append(
                ReferenceMismatch(marker=f"[{number}]", kind="cited_not_listed")
            )
        for number in sorted(defined.keys() - cited):
            mismatches.append(
                ReferenceMismatch(marker=f"[{number}]", kind="listed_not_cited")
            )
    else:
        style = "pandoc_key"
        cited_keys = {marker[2:-1] for marker in pandoc_markers}
        defined_keys = _key_defined(reference_body)
        for key in sorted(cited_keys - defined_keys.keys()):
            mismatches.append(
                ReferenceMismatch(marker=f"[@{key}]", kind="cited_not_listed")
            )
        for key in sorted(defined_keys.keys() - cited_keys):
            mismatches.append(
                ReferenceMismatch(marker=f"[@{key}]", kind="listed_not_cited")
            )

    return ReferenceConsistencyReport(
        style=style,
        mismatches=mismatches,
        reference_section_id=reference_section.id,
    )
