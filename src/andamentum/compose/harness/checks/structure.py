"""Limits and required sections from a project's ``checks.yaml`` (spec section 4, Tier 0).

No configuration, no diagnostics. The configuration is a mapping, passed to
``StructureCheck(config=...)`` or read from the project's ``checks.yaml``
(``<stem>.harness/checks.yaml``, key ``structure``; the same file that holds
the ``models:`` section). Keys, all optional:

    structure:
      max_words: 6000            # whole document, headings excluded
      max_figures: 8             # figure captions
      max_tables: 5              # table captions
      required_sections: [Abstract, Methods]    # heading text, case-insensitive
      sections:
        Abstract: {max_words: 250, min_words: 100}

Word counts split on whitespace over body paragraphs (non-heading). A section
is the paragraphs under one heading, up to the next heading of any level, so a
section with sub-headings counts only its own lead paragraphs. Figures and
tables are counted by caption (style containing "caption", or a paragraph that
starts "Figure N." / "Table N."); a figure inserted without a caption is not
counted. A limit key for a section that is absent is not reported (list it in
``required_sections`` for that). A malformed ``checks.yaml`` raises ValueError
rather than silently passing.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from ..kernel import (
    CheckContext,
    CheckSpec,
    Diagnostic,
    DocumentView,
    register,
)

_CAPTION_TEXT = re.compile(r"^\s*(Figure|Fig\.|Table)\s+S?\d+\s*[.:–—-]")
_KNOWN = {"max_words", "max_figures", "max_tables", "required_sections", "sections"}


def _words(text: str) -> int:
    return len(text.split())


def _from_project(ctx: CheckContext) -> Mapping[str, Any] | None:
    """The ``structure:`` section of the project's ``checks.yaml``, if any."""
    if ctx.store is None:
        return None
    section = ctx.store.config().get("structure")
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ValueError("checks.yaml: 'structure' must be a mapping")
    return section


class StructureCheck:
    spec = CheckSpec(id="structure", tier=0, scope="document", needs=[])
    version = "1"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        if config is not None:
            unknown = set(config) - _KNOWN
            if unknown:
                raise ValueError(f"unknown structure keys: {sorted(unknown)}")
        self.config = config

    def run(self, view: DocumentView, ctx: CheckContext) -> list[Diagnostic]:
        config = self.config
        if config is None:
            config = _from_project(ctx)
            if config is None:
                return []
            unknown = set(config) - _KNOWN
            if unknown:
                raise ValueError(f"unknown structure keys: {sorted(unknown)}")
        out: list[Diagnostic] = []
        by_id = {p.id: p for p in view.paragraphs}
        body = [p for p in view.paragraphs if p.heading_level is None]

        max_words = config.get("max_words")
        if max_words is not None:
            total = sum(_words(p.text) for p in body)
            if total > max_words:
                out.append(
                    self._diag(
                        f"Document has {total} words; the limit is {max_words}.",
                        evidence=[f"{total} > {max_words}"],
                    )
                )

        for key, label in (("max_figures", "Figure"), ("max_tables", "Table")):
            limit = config.get(key)
            if limit is None:
                continue
            numbers = set()
            for p in view.paragraphs:
                m = _CAPTION_TEXT.match(p.text)
                if m and m[1].startswith(label[:3]):
                    numbers.add(re.search(r"S?\d+", p.text)[0])  # type: ignore[index]
                elif "caption" in p.style.lower():
                    m2 = re.search(rf"({label}|Fig\.)\s+(S?\d+)", p.text)
                    if m2:
                        numbers.add(m2[2])
            if len(numbers) > limit:
                out.append(
                    self._diag(
                        f"Document has {len(numbers)} {label.lower()}s; the limit is {limit}.",
                        evidence=[f"{len(numbers)} > {limit}"],
                    )
                )

        sections = view.sections()
        headings = {h.strip().lower(): ids for h, ids in sections if h}
        heading_paragraph = {
            p.text.strip().lower(): p
            for p in view.paragraphs
            if p.heading_level is not None
        }
        for name in config.get("required_sections", []):
            if name.strip().lower() not in headings:
                out.append(
                    self._diag(
                        f"Required section {name!r} is missing.", evidence=[name]
                    )
                )
        for name, limits in config.get("sections", {}).items():
            key = name.strip().lower()
            if key not in headings:
                continue
            count = sum(_words(by_id[i].text) for i in headings[key])
            heading = heading_paragraph[key]
            for limit_key, breached in (
                ("max_words", lambda n, lim: n > lim),
                ("min_words", lambda n, lim: n < lim),
            ):
                lim = limits.get(limit_key)
                if lim is not None and breached(count, lim):
                    out.append(
                        self._diag(
                            f"Section {name!r} has {count} words; {limit_key} is {lim}.",
                            paragraph_id=heading.id,
                            start=0,
                            end=len(heading.text),
                            evidence=[f"{count} words"],
                        )
                    )
        return out

    def _diag(
        self,
        message: str,
        *,
        evidence: list[str],
        paragraph_id: str | None = None,
        start: int | None = None,
        end: int | None = None,
    ) -> Diagnostic:
        return Diagnostic(
            check=self.spec.id,
            tier=self.spec.tier,
            severity="error",
            paragraph_id=paragraph_id,
            start=start,
            end=end,
            message=message,
            evidence=evidence,
        )


register(StructureCheck())
