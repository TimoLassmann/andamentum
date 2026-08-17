"""Where the document's sections begin, read from the document.

A Word file already says which paragraphs are headings — it says it in the
paragraph's style, and Word's own navigation pane, table of contents and
outline view all read it from there. Nothing needs to be inferred from the
text.

The alternative, and what whetstone did before, is to convert the document to
markdown and look for ``#``. That answer is only as good as the converter:
Docling emits a heading for a bold standalone line that is not one, misses a
``Heading2`` whose text happens to end in a colon, and — measured on a real
manuscript — writes ``aim 3,4 .`` where Word holds ``aim3,4.``. Once
the text has been rewritten, the offsets a reviewer quotes are offsets into the
rewrite, and placing them back in the document becomes a matching problem. Read
the style instead and there is nothing to match: a section boundary is a
paragraph, and a paragraph is at a known index in the atom stream.

Level resolution, in order (the first answer wins):

1. The paragraph's own ``w:outlineLvl`` — direct formatting outranks its style.
2. Its ``w:pStyle``, resolved through ``styles.xml``: the style's own
   ``w:outlineLvl``, then its ``w:name`` against ``heading N``, then its
   ``w:basedOn`` parent.
3. Failing all that, the styleId itself, for a document that references a style
   it does not define.

Name beats styleId because the styleId is not the name: a German Word file
holds ``styleId="berschrift1"`` with ``w:name="heading 1"``. Built-in styles
always carry the canonical English name, whatever the UI shows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from .atoms import Atom, AtomKind, projection, qn
from .reading import has_ancestor

_TABLE_TAG = qn("tbl")
_STYLE_TAG = qn("style")
_VAL_ATTR = qn("val")

# ElementPath, not XPath: python-docx's element subclasses override ``xpath``
# with a signature that takes no namespace map, so the same call would work on
# a tree parsed from a zip and raise on one handed over by python-docx.
# A ``<w:pPr>`` hangs off a paragraph and off a style alike, so one path serves.
_OUTLINE_PATH = f"{qn('pPr')}/{qn('outlineLvl')}"
_STYLE_PATH = f"{qn('pPr')}/{qn('pStyle')}"
_NAME_PATH = qn("name")
_BASED_ON_PATH = qn("basedOn")

# ``heading 1`` … ``heading 9`` — the canonical built-in names. The separator is
# permissive so a styleId fallback (``Heading_2``, ``Heading3``) resolves too.
_HEADING_NAME_RE = re.compile(r"^heading[\s_-]*([1-9])$", re.IGNORECASE)

_TITLE_NAME = "title"

# Word stores outline level 0-based, and reserves 9 for "body text". A style
# can therefore say *explicitly* that it is not a heading — ``TOCHeading`` does
# exactly this — so 9 must read as "not a heading", never as level 10.
_BODY_TEXT_OUTLINE_LEVEL = 9

# A cycle in ``w:basedOn`` is malformed but cheap to survive; so is a chain long
# enough to suggest one.
_MAX_BASED_ON_DEPTH = 16

# The document title sits above ``Heading1`` so that everything before the first
# ``Heading1`` lands inside the title's section rather than in a preamble.
_TITLE_LEVEL = 0


@dataclass(frozen=True)
class ParagraphSpan:
    """Where one paragraph's text sits in the projection: ``[start, end)``.

    ``end == start`` for an empty paragraph. It still occupies a position —
    the paragraph break after it is a character — so it cannot be skipped
    without moving every offset that follows.
    """

    paragraph: Any
    start: int
    end: int


@dataclass(frozen=True)
class DocxHeading:
    """A heading paragraph, addressed in projection coordinates."""

    start: int
    end: int
    level: int
    title: str


def paragraph_spans(
    stream: Sequence[Atom], paragraphs: Sequence[Any]
) -> list[ParagraphSpan]:
    """Map each paragraph to its range in ``projection(stream)``.

    Derived by *consuming* the stream — a paragraph runs until the next
    ``PARAGRAPH_BREAK`` atom — rather than by re-deriving where
    :func:`~andamentum.whetstone.docx.atoms.document_atoms` puts its breaks. A
    second copy of that rule is exactly the drift ``atoms`` was written to kill.

    Grouping by ``atom.paragraph`` identity instead would look simpler and be
    wrong: an empty paragraph contributes no atoms, and a real manuscript is
    full of them (54 in the test document).
    """
    total = len(stream)
    if not paragraphs:
        if total:
            raise ValueError("atom stream is non-empty but no paragraphs were given")
        return []

    spans: list[ParagraphSpan] = []
    cursor = 0
    for paragraph in paragraphs:
        start = cursor
        while cursor < total and stream[cursor].kind is not AtomKind.PARAGRAPH_BREAK:
            cursor += 1
        end = cursor
        if end > start and stream[start].paragraph is not paragraph:
            raise ValueError(
                "atom stream and paragraph list disagree — the stream was built "
                "from a different paragraph list"
            )
        spans.append(ParagraphSpan(paragraph=paragraph, start=start, end=end))
        cursor += 1  # step over the break closing this paragraph

    # N paragraphs means N-1 breaks, so a walk that consumed the whole stream
    # ends one past its end. Anything else means the two inputs describe
    # different documents, and every offset derived here would be wrong.
    if cursor != total + 1:
        raise ValueError(
            f"atom stream and paragraph list disagree: {len(paragraphs)} paragraphs "
            f"consumed {cursor - 1} of {total} atoms"
        )
    return spans


class StyleLevels:
    """The heading level of every paragraph style, resolved once.

    Built from ``styles.xml`` up front so a 200-paragraph document makes one
    pass over the style table rather than 200.
    """

    def __init__(self, styles_root: Any | None) -> None:
        self._outline: dict[str, int] = {}
        self._name: dict[str, str] = {}
        self._based_on: dict[str, str] = {}
        self._resolved: dict[str, int | None] = {}

        if styles_root is None:
            return
        for style in styles_root.iter(_STYLE_TAG):
            style_id = style.get(qn("styleId"))
            if not style_id:
                continue
            outline = _first_val(style, _OUTLINE_PATH)
            if outline is not None:
                self._outline[style_id] = outline
            name = _first_attr(style, _NAME_PATH)
            if name:
                self._name[style_id] = name
            based_on = _first_attr(style, _BASED_ON_PATH)
            if based_on:
                self._based_on[style_id] = based_on

    def level_of(self, style_id: str) -> int | None:
        """The heading level *style_id* denotes, or ``None`` for body text."""
        if style_id in self._resolved:
            return self._resolved[style_id]

        level = self._resolve(style_id)
        self._resolved[style_id] = level
        return level

    def _resolve(self, style_id: str) -> int | None:
        seen: set[str] = set()
        current = style_id
        for _ in range(_MAX_BASED_ON_DEPTH):
            if current in seen:
                break
            seen.add(current)

            known = (
                current in self._outline
                or current in self._name
                or current in self._based_on
            )
            if not known:
                # A dangling reference: the document uses a style it never
                # defines. Its id is the only evidence left.
                return _level_from_name(current)

            outline = self._outline.get(current)
            if outline is not None:
                # An explicit outline level is the style's own answer, including
                # when the answer is "body text".
                return _level_from_outline(outline)

            named = _level_from_name(self._name.get(current, ""))
            if named is not None:
                return named

            parent = self._based_on.get(current)
            if parent is None:
                return None
            current = parent
        return None


def heading_level(p_elem: Any, *, styles: StyleLevels) -> int | None:
    """The heading level of one paragraph, or ``None`` if it is body text."""
    outline = _first_val(p_elem, _OUTLINE_PATH)
    if outline is not None:
        return _level_from_outline(outline)

    style_id = _first_attr(p_elem, _STYLE_PATH)
    if not style_id:
        return None
    return styles.level_of(style_id)


def document_headings(
    paragraphs: Sequence[Any], *, stream: Sequence[Atom], styles_root: Any | None
) -> list[DocxHeading]:
    """Every heading in the document, in reading order, with its offsets."""
    styles = StyleLevels(styles_root)
    text = projection(stream)
    headings: list[DocxHeading] = []

    for span in paragraph_spans(stream, paragraphs):
        if span.end == span.start:
            continue  # an empty paragraph is a break, never a heading
        if has_ancestor(span.paragraph, _TABLE_TAG):
            # A Heading-styled table cell is a cell, not a section boundary —
            # a table of contents built as a table would otherwise cut the
            # document into one section per row.
            continue
        level = heading_level(span.paragraph, styles=styles)
        if level is None:
            continue
        title = text[span.start : span.end].strip()
        if not title:
            continue
        headings.append(
            DocxHeading(start=span.start, end=span.end, level=level, title=title)
        )
    return headings


def _first_attr(element: Any, path: str) -> str | None:
    """The ``w:val`` of the first element at *path*, as a string."""
    found = element.find(path)
    if found is None:
        return None
    return found.get(_VAL_ATTR)


def _first_val(element: Any, path: str) -> int | None:
    """The ``w:val`` of the first element at *path*, as an int."""
    raw = _first_attr(element, path)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _level_from_outline(outline: int) -> int | None:
    if 0 <= outline < _BODY_TEXT_OUTLINE_LEVEL:
        return outline + 1
    return None


def _level_from_name(name: str) -> int | None:
    stripped = name.strip()
    if stripped.lower() == _TITLE_NAME:
        return _TITLE_LEVEL
    match = _HEADING_NAME_RE.match(stripped)
    return int(match.group(1)) if match else None
