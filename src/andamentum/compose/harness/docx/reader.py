"""Read a ``.docx`` into a :class:`~andamentum.compose.harness.kernel.DocumentView`.

Text comes from whetstone's atom stream (its public ``paragraph_atoms``): the
document as Word shows it with tracked changes accepted, field code left out
and field results kept. Offsets in every anchor, field and comment are indices
into that paragraph text, so a range the reader reports is a range the writer
can address.

Paragraph ids
    ``p<index>`` where *index* is the paragraph's place among the body
    paragraphs (``whetstone.body_paragraphs``: tables included, a prepended
    review report excluded), followed by ``-<w14:paraId>`` when Word recorded
    one. The rule is a pure function of the file, so ids are identical across
    re-opens of one file; an edit that adds or removes a paragraph renumbers
    those after it, so an id is only valid against the file it was read from.

A field that spans paragraphs (the bibliography) is reported on the paragraph
where it starts, ``end`` is that paragraph's length and ``display`` holds the
whole result with paragraphs joined by a newline. A comment whose anchor is
missing from the body gets ``paragraph_id == ""`` and a zero range.
"""

from __future__ import annotations

import re
import zipfile
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from andamentum.references import Hint, scan
from andamentum.docx_edit import (
    Atom,
    body_paragraphs,
    main_document_part,
    paragraph_atoms,
)

from ..kernel.view import Anchor, Comment, DocumentView, FieldToken, Paragraph
from ..errors import WordInputError
from .anchors import read_project_id, sdt_anchor
from .fields import classify_field
from .ooxml import (
    W15_NS,
    parse_xml,
    related_part,
    w,
    w14,
    w15,
)

_PACKAGE_RELS = "_rels/.rels"
_DEFAULT_DOCUMENT_PART = "word/document.xml"

_DELETED_TAGS = frozenset({w("del"), w("moveFrom")})
_HEADING_NAME = re.compile(r"^heading\s*(\d)$", re.IGNORECASE)

# Plain-text candidates from the references scanner that read as citations on
# their own; "(3)" and glued digits are too often something else to report as
# a field without the whole-document decision the checks make later.
_PLAIN_CITATION_SOURCES = frozenset(
    {"brackets", "parenthetical", "narrative", "key", "superscript"}
)


@dataclass
class ParsedDocx:
    """The XML behind a view, for the writer: elements line up with ``view.paragraphs``."""

    path: Path
    document_part: str
    root: Any
    elements: list[Any]
    atoms: list[list[Atom]]
    view: DocumentView


@dataclass
class _Style:
    name: str
    outline_level: int | None
    based_on: str | None


@dataclass
class _OpenField:
    seq: int
    start: tuple[int, int]
    instruction: list[str]


@dataclass
class _Located:
    paragraph: int
    start: int
    end: int


class DocxReadError(WordInputError, ValueError):
    """The file is not a readable ``.docx``."""


def open_docx(path: Path) -> DocumentView:
    """Read *path* into a view. Never modifies the file."""
    return parse_docx(path).view


def parse_docx(path: Path) -> ParsedDocx:
    """Read *path*, keeping the parsed XML next to the view."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            document_part = _DEFAULT_DOCUMENT_PART
            if _PACKAGE_RELS in names:
                document_part = main_document_part(archive.read(_PACKAGE_RELS))
            if document_part not in names:
                raise DocxReadError(
                    f"{path}: no main document part ({document_part!r})"
                )
            root = parse_xml(archive.read(document_part))
            styles = _read_styles(archive, document_part)
            project_id = read_project_id(archive)
            comment_parts = _read_comment_parts(archive, document_part)
    except zipfile.BadZipFile as exc:
        raise DocxReadError(f"{path}: not a readable .docx archive ({exc})") from exc

    elements = body_paragraphs(root)
    atoms = [list(paragraph_atoms(p)) for p in elements]
    texts = ["".join(a.char for a in group) for group in atoms]
    ids = [_paragraph_id(index, p) for index, p in enumerate(elements)]
    paragraphs = [
        _paragraph(index, p, ids[index], texts[index], styles)
        for index, p in enumerate(elements)
    ]
    index_of = {p: index for index, p in enumerate(elements)}

    scanned = _scan_elements(elements, atoms)
    anchors = _block_anchors(root, index_of, texts, ids) + [
        Anchor(
            id=a.id,
            kind=a.kind,
            paragraph_id=ids[a.paragraph],
            start=a.start,
            end=a.end,
        )
        for a in scanned.anchors
    ]
    ordered = sorted(scanned.fields, key=lambda f: (f.start, f.seq))
    bookmarks = {
        name
        for mark in root.iter(w("bookmarkStart"))
        if (name := mark.get(w("name"))) is not None
    }
    fields = [_field_token(f, ids, texts, bookmarks) for f in ordered]
    comments = _comments(comment_parts, scanned.comment_ranges, ids, texts)

    view = DocumentView(
        path=path,
        project_id=project_id,
        paragraphs=paragraphs,
        anchors=anchors,
        fields=fields + _plain_citations(paragraphs, fields, scanned.superscripts),
        comments=comments,
    )
    return ParsedDocx(
        path=path,
        document_part=document_part,
        root=root,
        elements=elements,
        atoms=atoms,
        view=view,
    )


# ---------------------------------------------------------------- paragraphs


def _paragraph_id(index: int, element: Any) -> str:
    para_id = element.get(w14("paraId"))
    return f"p{index}-{para_id}" if para_id else f"p{index}"


def _paragraph(
    index: int, element: Any, para_id: str, text: str, styles: dict[str, _Style]
) -> Paragraph:
    style_id = None
    properties = element.find(w("pPr"))
    if properties is not None:
        pstyle = properties.find(w("pStyle"))
        if pstyle is not None:
            style_id = pstyle.get(w("val"))
    style = styles.get(style_id) if style_id else None
    name = style.name if style else (style_id or "Normal")
    return Paragraph(
        id=para_id,
        index=index,
        style=name,
        heading_level=_heading_level(properties, style_id, styles),
        text=text,
    )


def _heading_level(
    properties: Any, style_id: str | None, styles: dict[str, _Style]
) -> int | None:
    """Heading level 1..9 from the style name, an outline level, or direct formatting."""
    if properties is not None:
        direct = properties.find(w("outlineLvl"))
        if direct is not None:
            return _outline(direct.get(w("val")))
    seen: set[str] = set()
    while style_id and style_id not in seen and style_id in styles:
        seen.add(style_id)
        style = styles[style_id]
        match = _HEADING_NAME.match(style.name)
        if match:
            return int(match.group(1))
        if style.outline_level is not None:
            return style.outline_level
        style_id = style.based_on
    return None


def _outline(value: str | None) -> int | None:
    # w:outlineLvl is 0-based, with 9 meaning body text.
    if value is not None and value.isdigit() and int(value) < 9:
        return int(value) + 1
    return None


def _read_styles(archive: zipfile.ZipFile, document_part: str) -> dict[str, _Style]:
    folder = document_part.rpartition("/")[0]
    conventional = f"{folder}/styles.xml" if folder else "styles.xml"
    part = related_part(
        archive, document_part, type_suffix="/styles", fallback=conventional
    )
    if part is None:
        return {}
    root = parse_xml(archive.read(part))
    styles: dict[str, _Style] = {}
    for style in root.iter(w("style")):
        style_id = style.get(w("styleId"))
        if not style_id:
            continue
        name = style.find(w("name"))
        based = style.find(w("basedOn"))
        outline = style.find(f"{w('pPr')}/{w('outlineLvl')}")
        styles[style_id] = _Style(
            name=(name.get(w("val")) if name is not None else None) or style_id,
            outline_level=_outline(
                outline.get(w("val")) if outline is not None else None
            ),
            based_on=based.get(w("val")) if based is not None else None,
        )
    return styles


# ------------------------------------------------------- one pass over the XML


@dataclass
class _Found:
    id: str
    kind: Any
    paragraph: int
    start: int
    end: int


@dataclass
class _RawField:
    seq: int  # order in which the field began; outer fields begin before inner ones
    instruction: str
    start: tuple[int, int]
    end: tuple[int, int]
    simple: bool


@dataclass
class _Scan:
    anchors: list[_Found]
    fields: list[_RawField]
    comment_ranges: dict[str, dict[str, tuple[int, int]]]
    superscripts: list[tuple[int, int, int]]  # (paragraph, start, end)


def _in_deleted(element: Any) -> bool:
    parent = element.getparent()
    while parent is not None:
        if parent.tag in _DELETED_TAGS:
            return True
        parent = parent.getparent()
    return False


def _scan_elements(elements: list[Any], atoms: list[list[Atom]]) -> _Scan:
    result = _Scan(anchors=[], fields=[], comment_ranges={}, superscripts=[])
    stack: list[_OpenField] = []  # complex fields open across paragraphs
    began = 0

    for index, paragraph in enumerate(elements):
        order = {el: i for i, el in enumerate(paragraph.iter())}
        atom_orders = [order[a.element] for a in atoms[index]]

        def before(element: Any) -> tuple[int, int]:
            return index, bisect_left(atom_orders, order[element])

        def after(element: Any) -> tuple[int, int]:
            last = element
            for last in element.iter():
                pass
            return index, bisect_right(atom_orders, order[last])

        for element in paragraph.iter():
            tag = element.tag
            if not isinstance(tag, str):
                continue
            if tag == w("fldChar"):
                if _in_deleted(element):
                    continue
                kind = element.get(w("fldCharType"))
                if kind == "begin":
                    began += 1
                    stack.append(
                        _OpenField(seq=began, start=before(element), instruction=[])
                    )
                elif kind == "end" and stack:
                    opened = stack.pop()
                    result.fields.append(
                        _RawField(
                            seq=opened.seq,
                            instruction="".join(opened.instruction),
                            start=opened.start,
                            end=before(element),
                            simple=False,
                        )
                    )
            elif tag == w("instrText"):
                if stack and not _in_deleted(element):
                    stack[-1].instruction.append(element.text or "")
            elif tag == w("fldSimple"):
                began += 1
                result.fields.append(
                    _RawField(
                        seq=began,
                        instruction=element.get(w("instr")) or "",
                        start=before(element),
                        end=after(element),
                        simple=True,
                    )
                )
            elif tag == w("sdt"):
                found = sdt_anchor(element)
                if found is not None:
                    start, end = before(element), after(element)
                    result.anchors.append(
                        _Found(
                            id=found[0],
                            kind=found[1],
                            paragraph=index,
                            start=start[1],
                            end=end[1],
                        )
                    )
            elif tag == w("commentRangeStart"):
                result.comment_ranges.setdefault(element.get(w("id")) or "", {})[
                    "start"
                ] = before(element)
            elif tag == w("commentRangeEnd"):
                result.comment_ranges.setdefault(element.get(w("id")) or "", {})[
                    "end"
                ] = before(element)
            elif tag == w("commentReference"):
                result.comment_ranges.setdefault(element.get(w("id")) or "", {})[
                    "point"
                ] = before(element)
            elif tag == w("vertAlign") and element.get(w("val")) == "superscript":
                run = (
                    element.getparent().getparent()
                    if element.getparent() is not None
                    else None
                )
                if run is not None and run.tag == w("r") and run in order:
                    start, end = before(run), after(run)
                    if end[1] > start[1]:
                        result.superscripts.append((index, start[1], end[1]))
    return result


def _block_anchors(
    root: Any, index_of: dict[Any, int], texts: list[str], ids: list[str]
) -> list[Anchor]:
    """Anchors that wrap whole paragraphs: reported on the first paragraph they hold."""
    anchors: list[Anchor] = []
    for sdt in root.iter(w("sdt")):
        found = sdt_anchor(sdt)
        if found is None or _has_paragraph_ancestor(sdt):
            continue
        for inner in sdt.iter(w("p")):
            if inner in index_of:
                index = index_of[inner]
                anchors.append(
                    Anchor(
                        id=found[0],
                        kind=found[1],
                        paragraph_id=ids[index],
                        start=0,
                        end=len(texts[index]),
                    )
                )
                break
    return anchors


def _has_paragraph_ancestor(element: Any) -> bool:
    parent = element.getparent()
    while parent is not None:
        if parent.tag == w("p"):
            return True
        parent = parent.getparent()
    return False


# -------------------------------------------------------------------- fields


_BOOKMARK_FIELDS = frozenset({"REF", "PAGEREF", "NOTEREF"})


def _field_token(
    raw: _RawField, ids: list[str], texts: list[str], bookmarks: set[str]
) -> FieldToken:
    kind, manager, data = classify_field(raw.instruction)
    if kind == "crossref" and data.get("field") in _BOOKMARK_FIELDS:
        # The check reads this flag: the view carries no bookmark list.
        data["target_exists"] = data.get("target") in bookmarks
    first, start = raw.start
    last, end = raw.end
    if last == first:
        display = texts[first][start:end]
    else:
        pieces = [texts[first][start:], *texts[first + 1 : last], texts[last][:end]]
        display = "\n".join(pieces)
        end = len(texts[first])
    return FieldToken(
        kind=kind,
        manager=manager,
        paragraph_id=ids[first],
        start=start,
        end=end,
        display=display,
        data=data,
    )


def _plain_citations(
    paragraphs: list[Paragraph],
    fields: list[FieldToken],
    superscripts: list[tuple[int, int, int]],
) -> list[FieldToken]:
    """Plain-text citations, found by ``andamentum.references`` scanning.

    Offsets are mapped back from the joined text to ``(paragraph, local)``;
    a candidate spanning two paragraphs is not a citation and is skipped.
    """
    text = "\n".join(p.text for p in paragraphs)
    starts: list[int] = []
    offset = 0
    for paragraph in paragraphs:
        starts.append(offset)
        offset += len(paragraph.text) + 1
    index_by_id = {p.id: i for i, p in enumerate(paragraphs)}

    hints = [
        Hint(kind="superscript", start=starts[i] + a, end=starts[i] + b)
        for i, a, b in superscripts
    ]
    covered: list[tuple[int, int]] = []
    for field in fields:
        if field.kind != "citation":
            continue
        base = starts[index_by_id[field.paragraph_id]]
        hints.append(
            Hint(kind="field_citation", start=base + field.start, end=base + field.end)
        )
        covered.append((base + field.start, base + field.end))

    tokens: list[FieldToken] = []
    for candidate in scan(text, start=0, end=len(text), hints=hints):
        if candidate.source not in _PLAIN_CITATION_SOURCES:
            continue
        if any(a <= candidate.start < b for a, b in covered):
            continue
        index = _paragraph_at(starts, paragraphs, candidate.start)
        if candidate.end > starts[index] + len(paragraphs[index].text):
            continue
        tokens.append(
            FieldToken(
                kind="citation",
                manager="none",
                paragraph_id=paragraphs[index].id,
                start=candidate.start - starts[index],
                end=candidate.end - starts[index],
                display=candidate.text,
                data={
                    "source": candidate.source,
                    "numbers": candidate.numbers,
                    "works": [work.model_dump() for work in candidate.works],
                    "keys": candidate.keys,
                },
            )
        )
    return tokens


def _paragraph_at(starts: list[int], paragraphs: list[Paragraph], offset: int) -> int:
    index = bisect_right(starts, offset) - 1
    return max(index, 0)


# ------------------------------------------------------------------ comments


@dataclass
class _RawComment:
    id: str
    author: str
    date: str | None
    text: str
    para_ids: list[str]


@dataclass
class _CommentParts:
    comments: list[_RawComment]
    parent_of: dict[str, str]  # w14:paraId -> parent w14:paraId
    done: dict[str, bool]  # w14:paraId -> resolved


def _read_comment_parts(archive: zipfile.ZipFile, document_part: str) -> _CommentParts:
    folder = document_part.rpartition("/")[0]
    prefix = f"{folder}/" if folder else ""
    comments_part = related_part(
        archive,
        document_part,
        type_suffix="/comments",
        fallback=f"{prefix}comments.xml",
    )
    raw: list[_RawComment] = []
    if comments_part is not None:
        root = parse_xml(archive.read(comments_part))
        for comment in root.iter(w("comment")):
            paragraphs = body_paragraphs(comment)
            raw.append(
                _RawComment(
                    id=comment.get(w("id")) or "",
                    author=comment.get(w("author")) or "",
                    date=comment.get(w("date")),
                    text="\n".join(
                        "".join(a.char for a in paragraph_atoms(p)) for p in paragraphs
                    ),
                    para_ids=[
                        p.get(w14("paraId")) for p in paragraphs if p.get(w14("paraId"))
                    ],
                )
            )
    parent_of: dict[str, str] = {}
    done: dict[str, bool] = {}
    extended = related_part(
        archive,
        document_part,
        type_suffix="/commentsExtended",
        fallback=f"{prefix}commentsExtended.xml",
    )
    if extended is not None:
        for ex in parse_xml(archive.read(extended)).iter(f"{{{W15_NS}}}commentEx"):
            para_id = ex.get(w15("paraId"))
            if not para_id:
                continue
            parent = ex.get(w15("paraIdParent"))
            if parent:
                parent_of[para_id] = parent
            done[para_id] = ex.get(w15("done")) in ("1", "true")
    return _CommentParts(comments=raw, parent_of=parent_of, done=done)


def _comments(
    parts: _CommentParts,
    ranges: dict[str, dict[str, tuple[int, int]]],
    ids: list[str],
    texts: list[str],
) -> list[Comment]:
    id_by_para = {pid: c.id for c in parts.comments for pid in c.para_ids}
    comments: list[Comment] = []
    for raw in parts.comments:
        located = _locate_comment(ranges.get(raw.id, {}), texts)
        parent = next(
            (
                id_by_para[parts.parent_of[pid]]
                for pid in raw.para_ids
                if pid in parts.parent_of and parts.parent_of[pid] in id_by_para
            ),
            None,
        )
        comments.append(
            Comment(
                id=raw.id,
                author=raw.author,
                date=raw.date,
                text=raw.text,
                paragraph_id=ids[located.paragraph] if located else "",
                start=located.start if located else 0,
                end=located.end if located else 0,
                parent_id=parent,
                resolved=any(parts.done.get(pid, False) for pid in raw.para_ids),
            )
        )
    return comments


def _locate_comment(
    marks: dict[str, tuple[int, int]], texts: list[str]
) -> _Located | None:
    if "start" in marks:
        first, start = marks["start"]
        last, end = marks.get("end", marks["start"])
        return _Located(first, start, end if last == first else len(texts[first]))
    if "point" in marks:
        index, offset = marks["point"]
        return _Located(index, offset, offset)
    return None
