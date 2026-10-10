"""Body and relationship helpers for a .docx: which paragraphs are the document body.

Shared by whetstone (sectionising, review) and compose (the Word writer and reader),
so both read the same body the same way.
"""

from __future__ import annotations

from typing import Any

from lxml import etree  # type: ignore

from .atoms import qn

_PARAGRAPH_TAG = qn("p")
_BOOKMARK_START_TAG = qn("bookmarkStart")
_BOOKMARK_END_TAG = qn("bookmarkEnd")
_ID_ATTR = qn("id")
_NAME_ATTR = qn("name")
REVIEW_REPORT_BOOKMARK_PREFIX = "andamentum_review_report_"
_RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_RELATIONSHIP_TAG = f"{{{_RELS_NS}}}Relationship"
_PACKAGE_RELS = "_rels/.rels"
_DEFAULT_DOCUMENT_PART = "word/document.xml"
_OFFICE_DOCUMENT_REL = "/officeDocument"


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


def relationship_target(rels_xml: bytes, *, type_suffix: str) -> str | None:
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
    return relationship_target(rels_xml, type_suffix=_OFFICE_DOCUMENT_REL) or (
        _DEFAULT_DOCUMENT_PART
    )
