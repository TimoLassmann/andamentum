"""Content-control anchors and the project id.

An anchor is a ``w:sdt`` whose ``w:tag`` is ``am:<id>``. The anchor kind is
stored in the control's ``w:alias`` (``claim``, ``number``, ``figure`` or
``span``); an absent or unknown alias reads as ``span``. The writer sets the
tag, the alias and ``w15:appearance w15:val="hidden"``; the reader needs only
the first two.

The project id lives in a custom XML part whose root element is in the
``urn:andamentum:harness`` namespace and carries the id in its ``projectId``
attribute (a ``<projectId>`` child element is read too).
"""

from __future__ import annotations

import zipfile
from typing import Any, get_args

from lxml import etree  # type: ignore

from ..kernel.view import AnchorKind
from .ooxml import parse_xml, w

HARNESS_NS = "urn:andamentum:harness"
ANCHOR_TAG_PREFIX = "am:"
PROJECT_ID_ATTR = "projectId"

_KINDS: tuple[str, ...] = get_args(AnchorKind)


def anchor_tag(anchor_id: str) -> str:
    """The ``w:tag`` value that marks a control as anchor *anchor_id*."""
    return f"{ANCHOR_TAG_PREFIX}{anchor_id}"


def parse_anchor_tag(tag: str | None) -> str | None:
    """The anchor id in a ``w:tag`` value, or None when it is not ours."""
    if tag and tag.startswith(ANCHOR_TAG_PREFIX) and len(tag) > len(ANCHOR_TAG_PREFIX):
        return tag[len(ANCHOR_TAG_PREFIX) :]
    return None


def sdt_anchor(sdt: Any) -> tuple[str, AnchorKind] | None:
    """``(anchor id, kind)`` of a ``w:sdt`` element, or None if it is not an anchor."""
    properties = sdt.find(w("sdtPr"))
    if properties is None:
        return None
    tag = properties.find(w("tag"))
    anchor_id = parse_anchor_tag(tag.get(w("val")) if tag is not None else None)
    if anchor_id is None:
        return None
    alias = properties.find(w("alias"))
    kind = alias.get(w("val")) if alias is not None else None
    return anchor_id, kind if kind in _KINDS else "span"  # type: ignore[return-value]


def project_id_part_xml(project_id: str) -> bytes:
    """The custom XML part that records *project_id* in a document."""
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<harness xmlns="{HARNESS_NS}" {PROJECT_ID_ATTR}="{project_id}"/>'
    ).encode()


def read_project_id(archive: zipfile.ZipFile) -> str | None:
    """The project id stored in the document's custom XML, or None."""
    for name in sorted(archive.namelist()):
        if not (name.startswith("customXml/") and name.endswith(".xml")):
            continue
        if "/_rels/" in name or name.rpartition("/")[2].startswith("itemProps"):
            continue
        try:
            root = parse_xml(archive.read(name))
        except etree.XMLSyntaxError:  # a custom part that is not XML is not ours
            continue
        if not isinstance(root.tag, str) or not root.tag.startswith(
            f"{{{HARNESS_NS}}}"
        ):
            continue
        value = root.get(PROJECT_ID_ATTR)
        if value is None:
            child = root.find(f"{{{HARNESS_NS}}}{PROJECT_ID_ATTR}")
            value = child.text if child is not None else None
        if value and value.strip():
            return value.strip()
    return None
