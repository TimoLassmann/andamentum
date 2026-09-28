"""Deterministic parts of the document model — no LLM.

Section gists, the citation summary (read by :mod:`andamentum.references`,
the package's one citation reader), and assembling the verified claims +
deterministic facets into a `DocumentModel`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .model import CitationSummary, Claim, DocumentModel, Section, SectionGist

if TYPE_CHECKING:
    from andamentum.references import Hint

async def read_citations(
    source: str, sections: list[Section], *, hints: "list[Hint] | None" = None
) -> CitationSummary:
    """How *source* cites, read by the reference checker with no model and no
    lookup: deterministic, no model call, nothing sent anywhere.

    *hints* are what the Word file knows and its text does not — superscript
    runs, reference-manager fields, footnote marks (see
    :func:`~..docx.citation_hints.citation_hints`). Without them, a superscript
    citation is still read when the text kept it glued to its word.
    """
    from andamentum.references import check_references

    report = await check_references(
        source,
        sections=[(s.title, s.start, s.end) for s in sections],
        hints=hints,
        model=None,
        lookup=False,
    )
    return CitationSummary(
        found_list=report.found_list,
        entries=len(report.entries),
        style=report.style.style if report.style is not None else None,
        citations=len(report.citations),
    )


def citations_line(summary: CitationSummary) -> str:
    """The summary as one prompt line, in the words the reference report uses."""
    from ..references_check import style_name

    if summary.style is None:
        return "CITATIONS: none found, and the document has no reference list."
    if summary.style == "undecided":
        return (
            f"CITATIONS: the document has a reference list of {summary.entries} "
            f"entries, but how the text cites it could not be decided."
        )
    target = (
        f"a reference list of {summary.entries} entries"
        if summary.found_list
        else "no separate reference list"
    )
    return (
        f"CITATIONS: {summary.citations} in the text, as "
        f"{style_name(summary.style)}; {target}."
    )


def _heading_stripped(section: Section) -> str:
    """Section body with its leading heading line removed.

    Two shapes, because a section has two possible origins. Markdown text
    opens with ``## Methods``; a Word document opens with the bare word
    ``Methods``, because the heading is carried in the paragraph's style and
    the projection holds only what a reader sees. Stripping only the first
    made every gist on the .docx path read back as the section's own title.
    """
    body = section.text.lstrip()
    first, newline, rest = body.partition("\n")
    if first.startswith("#") or first.strip() == section.title:
        body = rest if newline else ""
    return body.strip()


def gist_for(section: Section, *, max_chars: int = 160) -> str:
    """Deterministic one-line gist: the first sentence of the section body."""
    body = " ".join(_heading_stripped(section).split())
    if not body:
        return ""
    # First sentence: up to a sentence-ending punctuation, else a char cap.
    m = re.search(r"[.!?](?:\s|$)", body)
    sentence = body[: m.end()].strip() if m else body
    if len(sentence) > max_chars:
        sentence = sentence[: max_chars - 1].rstrip() + "…"
    return sentence


def build_document_model(
    source: str,
    sections: list[Section],
    claims: list[Claim],
    *,
    citations: CitationSummary | None = None,
) -> DocumentModel:
    """Assemble verified claims + deterministic facets into the model.

    Claims arrive already located (Phase-1 verify). *citations* is
    :func:`read_citations`'s summary, read by the caller because it is async.
    """
    gists = [
        SectionGist(section_id=s.id, title=s.title, gist=gist_for(s)) for s in sections
    ]
    return DocumentModel(
        source=source,
        sections=sections,
        claims=claims,
        gists=gists,
        citations=citations,
    )
