"""Reading a ``.docx`` — the parts you need when you are not going to write.

:class:`~andamentum.whetstone.docx.low_level.DocxEditor` unzips a document into
a temporary directory because it is going to zip the archive back up again.
Sectionising only reads, so it parses the two parts it needs straight out of
the zip and leaves nothing on disk.

The two rules that decide *what* gets read live here rather than in either
caller: which part is the main document, and which ``<w:p>`` elements are body
paragraphs. ``DocxEditor`` delegates to both, so a paragraph list built for
sectionising and one built for editing cannot disagree — and a span resolved
against the first is a valid address in the second.

The second rule has a third job, and it is what makes a review of a reviewed
document possible: **a prepended review report is not part of the document.**
Round one writes its report into the top of the file, so without this round two
would read its own predecessor's findings as if the author had written them,
and re-flagging a passage round one already flagged would make that passage's
text ambiguous (the report restates every quote). The marker is a bookmark
written into the file rather than a flag on an object, because round two is a
different process opening a file.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lxml import etree  # type: ignore

from .atoms import paragraph_atoms, projection, qn
from .exceptions import ConversionError

_PARAGRAPH_TAG = qn("p")
_BOOKMARK_START_TAG = qn("bookmarkStart")
_BOOKMARK_END_TAG = qn("bookmarkEnd")
_ID_ATTR = qn("id")
_NAME_ATTR = qn("name")

# Bookmark names bracketing a prepended review report. A bookmark rather than a
# paragraph style, because the report's paragraphs already carry the styles that
# make it readable (Heading1, ListBullet, Quote) and a marker must not compete
# for that slot; and because it is invisible, so nothing about the document a
# reader sees changes. Each round takes the next free number, so two reports
# never share a name — Word renames a duplicate bookmark on save, and a renamed
# marker is an unmarked report.
REVIEW_REPORT_BOOKMARK_PREFIX = "andamentum_review_report_"

# Parts that hold reader-visible text but are not the body. The reviewer never
# sees them — every offset is an address in the body projection — but the
# confidentiality tripwire has to, because "CONFIDENTIAL — FOR PEER REVIEW
# ONLY" is most often a running header rather than a paragraph of the draft.
_AUXILIARY_PART_PREFIXES: tuple[str, ...] = ("word/header", "word/footer")
_AUXILIARY_PARTS: frozenset[str] = frozenset(
    {"word/footnotes.xml", "word/endnotes.xml"}
)

_RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_RELATIONSHIP_TAG = f"{{{_RELS_NS}}}Relationship"

_PACKAGE_RELS = "_rels/.rels"
_DEFAULT_DOCUMENT_PART = "word/document.xml"

# Relationship type suffixes. Matched on the suffix because the namespace
# prefix of the type URI differs between the package-level and part-level
# relationship vocabularies.
_OFFICE_DOCUMENT_REL = "/officeDocument"
_STYLES_REL = "/styles"


def has_ancestor(element: Any, tag: str) -> bool:
    """Is *element* nested inside a *tag* element?"""
    parent = element.getparent()
    while parent is not None:
        if parent.tag == tag:
            return True
        parent = parent.getparent()
    return False


def body_paragraphs(root: Any) -> list[Any]:
    """The outermost ``<w:p>`` elements under *root*, in reading order.

    Paragraphs nested inside another paragraph are excluded. A text box holds
    its content in a ``<w:p>`` inside the anchoring paragraph's run, and the
    atom walker descends into it already — returning both would read that text
    twice, once in place and once as a paragraph of its own. Table cells are
    unaffected: a ``<w:tbl>`` is a sibling of body paragraphs, not a descendant
    of one.

    Paragraphs inside a review-report bookmark are excluded too: a report an
    earlier round prepended is this pipeline's own output, not the author's
    draft (see the module docstring).

    Empty paragraphs are kept. Each one is a real break a reader sees, and
    dropping them would move every offset after it.

    Navigation is ``iter`` / ``getparent`` rather than XPath because
    python-docx's element subclasses override ``xpath`` with a signature that
    takes no namespace map — the same call would work on a tree parsed here and
    raise on one handed over by python-docx.
    """
    every: list[Any] = []
    outside_report: list[Any] = []
    open_report: set[str | None] = set()

    for element in root.iter():
        tag = element.tag
        if not isinstance(tag, str):  # XML comment / processing instruction
            continue
        if tag == _BOOKMARK_START_TAG:
            if (element.get(_NAME_ATTR) or "").startswith(
                REVIEW_REPORT_BOOKMARK_PREFIX
            ):
                open_report.add(element.get(_ID_ATTR))
        elif tag == _BOOKMARK_END_TAG:
            open_report.discard(element.get(_ID_ATTR))
        elif tag == _PARAGRAPH_TAG:
            if element is root or has_ancestor(element, _PARAGRAPH_TAG):
                continue
            every.append(element)
            if not open_report:
                outside_report.append(element)

    if open_report:
        # A start with no end: someone edited the report and took the closing
        # marker with it. Excluding everything from there to the end of the file
        # would hide the manuscript, so a damaged marker is no marker.
        return every
    return outside_report


def _relationship_target(rels_xml: bytes, *, type_suffix: str) -> str | None:
    """The ``Target`` of the first relationship whose type ends with *suffix*."""
    try:
        root = etree.fromstring(rels_xml)
    except etree.XMLSyntaxError:
        return None
    for relationship in root.iter(_RELATIONSHIP_TAG):
        if relationship.get("Type", "").endswith(type_suffix):
            target = relationship.get("Target", "")
            return target.lstrip("/") or None
    return None


def main_document_part(rels_xml: bytes) -> str:
    """The zip path of the main document part, per ``_rels/.rels``.

    Recent Word versions sometimes save the body as ``word/document2.xml`` when
    extended features (threaded comments, @mentions) are in play, so the part
    name is read from the package relationships rather than assumed.
    """
    return _relationship_target(rels_xml, type_suffix=_OFFICE_DOCUMENT_REL) or (
        _DEFAULT_DOCUMENT_PART
    )


@dataclass(frozen=True)
class DocxParts:
    """What sectionising reads out of a document: its body and its style table."""

    paragraphs: list[Any]
    styles_root: Any | None


def read_docx(path: str | Path) -> DocxParts:
    """Parse *path* into body paragraphs plus the style table, without unzipping.

    ``styles_root`` is ``None`` for a document that ships no ``styles.xml``;
    heading detection then falls back to whatever direct formatting the
    paragraphs carry.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())

            document_part = _DEFAULT_DOCUMENT_PART
            if _PACKAGE_RELS in names:
                document_part = main_document_part(archive.read(_PACKAGE_RELS))
            if document_part not in names:
                raise ConversionError(
                    f"{path}: no main document part (looked for {document_part!r})"
                )
            document_root = etree.fromstring(archive.read(document_part))

            styles_part = _styles_part(archive, names, document_part=document_part)
            styles_root = (
                etree.fromstring(archive.read(styles_part)) if styles_part else None
            )
    except zipfile.BadZipFile as exc:
        raise ConversionError(f"{path}: not a readable .docx archive ({exc})") from exc

    return DocxParts(paragraphs=body_paragraphs(document_root), styles_root=styles_root)


def auxiliary_text(path: str | Path) -> str:
    """Reader-visible text OUTSIDE the body: headers, footers, notes.

    Returned as one blob for scanning, not for reviewing — it carries no
    offsets and nothing is ever anchored into it. The body projection is the
    document as far as every coordinate in whetstone is concerned; this exists
    so a marker Word keeps in a header cannot walk past the confidentiality
    gate just because the gate now reads the body directly.

    A part that will not parse is skipped rather than raised on: a malformed
    header must not stop a review of a well-formed body.
    """
    blocks: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if not (
                    name.startswith(_AUXILIARY_PART_PREFIXES)
                    or name in _AUXILIARY_PARTS
                ):
                    continue
                try:
                    root = etree.fromstring(archive.read(name))
                except etree.XMLSyntaxError:
                    continue
                for paragraph in body_paragraphs(root):
                    text = projection(list(paragraph_atoms(paragraph))).strip()
                    if text:
                        blocks.append(text)
    except zipfile.BadZipFile as exc:
        raise ConversionError(f"{path}: not a readable .docx archive ({exc})") from exc
    return "\n".join(blocks)


def _styles_part(
    archive: zipfile.ZipFile, names: set[str], *, document_part: str
) -> str | None:
    """Locate ``styles.xml`` via the document's relationships, then by convention."""
    folder = document_part.rpartition("/")[0]
    rels_name = f"{folder}/_rels/{document_part.rpartition('/')[2]}.rels"
    if rels_name in names:
        target = _relationship_target(archive.read(rels_name), type_suffix=_STYLES_REL)
        # A part-level Target is relative to the part's own folder.
        if target:
            candidate = f"{folder}/{target}" if folder else target
            if candidate in names:
                return candidate
    conventional = f"{folder}/styles.xml" if folder else "styles.xml"
    return conventional if conventional in names else None
