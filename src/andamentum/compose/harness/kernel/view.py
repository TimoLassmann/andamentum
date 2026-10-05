"""DocumentView: the format-neutral reading of a document that checks run against."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

AnchorKind = Literal["claim", "number", "figure", "span"]


class Anchor(BaseModel):
    """A bound range (offsets are within the paragraph text)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: AnchorKind
    paragraph_id: str
    start: int
    end: int


class FieldToken(BaseModel):
    """A field in the text: citation, bibliography, cross-reference or other."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["citation", "bibliography", "crossref", "other"]
    manager: Literal["zotero", "endnote", "mendeley", "word", "none"]
    paragraph_id: str
    start: int
    end: int
    display: str
    data: dict[str, object]  # CSL items / EndNote record / ref target


class Comment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    author: str
    date: str | None
    text: str
    paragraph_id: str
    start: int
    end: int
    parent_id: str | None
    resolved: bool


class Paragraph(BaseModel):
    """One paragraph. ``text`` is what Word shows with tracked changes accepted
    (insertions in, deletions out); fields appear as their display text."""

    model_config = ConfigDict(extra="forbid")

    id: str
    index: int
    style: str
    heading_level: int | None
    text: str


class DocumentView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path
    project_id: str | None
    paragraphs: list[Paragraph]
    anchors: list[Anchor]
    fields: list[FieldToken]
    comments: list[Comment]

    def text(self) -> str:
        """Paragraph texts joined by a single newline."""
        return "\n".join(p.text for p in self.paragraphs)

    def locate(self, offset: int) -> tuple[str, int]:
        """Map an offset in ``text()`` to ``(paragraph_id, local offset)``.

        Valid offsets are ``0 <= offset <= len(text())``. An offset on the
        newline that separates two paragraphs belongs to the paragraph before
        it (its end), so a range end never lands in the next paragraph.
        Anything outside the text raises ``ValueError``.
        """
        if offset < 0:
            raise ValueError(f"offset {offset} is negative")
        start = 0
        for paragraph in self.paragraphs:
            end = start + len(paragraph.text)
            if offset <= end:
                return paragraph.id, offset - start
            start = end + 1  # skip the "\n" separator
        raise ValueError(
            f"offset {offset} is beyond the end of the text ({len(self.text())})"
        )

    def sections(self) -> list[tuple[str, list[str]]]:
        """``(heading text, body paragraph ids)`` by heading styles.

        Each heading of any level starts a new section; the heading paragraph
        itself is not in the list. Paragraphs before the first heading form a
        section with the empty heading ``""`` (only present when non-empty).
        """
        result: list[tuple[str, list[str]]] = []
        current: tuple[str, list[str]] | None = None
        for paragraph in self.paragraphs:
            if paragraph.heading_level is not None:
                current = (paragraph.text, [])
                result.append(current)
            else:
                if current is None:
                    current = ("", [])
                    result.append(current)
                current[1].append(paragraph.id)
        return result
