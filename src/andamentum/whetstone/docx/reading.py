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

from andamentum.docx_edit.atoms import paragraph_atoms, projection
from andamentum.docx_edit.body import (
    _DEFAULT_DOCUMENT_PART,
    _PACKAGE_RELS,
    body_paragraphs,
    main_document_part,
    relationship_target,
)
from .exceptions import ConversionError


# Bookmark names bracketing a prepended review report. A bookmark rather than a
# paragraph style, because the report's paragraphs already carry the styles that
# make it readable (Heading1, ListBullet, Quote) and a marker must not compete
# for that slot; and because it is invisible, so nothing about the document a
# reader sees changes. Each round takes the next free number, so two reports
# never share a name — Word renames a duplicate bookmark on save, and a renamed
# marker is an unmarked report.

# Parts that hold reader-visible text but are not the body. The reviewer never
# sees them — every offset is an address in the body projection — but the
# confidentiality tripwire has to, because "CONFIDENTIAL — FOR PEER REVIEW
# ONLY" is most often a running header rather than a paragraph of the draft.
_AUXILIARY_PART_PREFIXES: tuple[str, ...] = ("word/header", "word/footer")
_AUXILIARY_PARTS: frozenset[str] = frozenset(
    {"word/footnotes.xml", "word/endnotes.xml"}
)



# Relationship type suffixes. Matched on the suffix because the namespace
# prefix of the type URI differs between the package-level and part-level
# relationship vocabularies.
_STYLES_REL = "/styles"


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
        target = relationship_target(archive.read(rels_name), type_suffix=_STYLES_REL)
        # A part-level Target is relative to the part's own folder.
        if target:
            candidate = f"{folder}/{target}" if folder else target
            if candidate in names:
                return candidate
    conventional = f"{folder}/styles.xml" if folder else "styles.xml"
    return conventional if conventional in names else None
