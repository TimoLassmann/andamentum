"""WordprocessingML names and small XML helpers shared by the docx reader."""

from __future__ import annotations

import posixpath
import zipfile
from typing import Any

from lxml import etree  # type: ignore

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W14_NS = "http://schemas.microsoft.com/office/word/2010/wordml"
W15_NS = "http://schemas.microsoft.com/office/word/2012/wordml"
W16CID_NS = "http://schemas.microsoft.com/office/word/2016/wordml/cid"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
XML_NS = "http://www.w3.org/XML/1998/namespace"


def w(tag: str) -> str:
    """``"t"`` -> the fully qualified ``<w:t>`` name."""
    return f"{{{W_NS}}}{tag}"


def w14(tag: str) -> str:
    return f"{{{W14_NS}}}{tag}"


def w15(tag: str) -> str:
    return f"{{{W15_NS}}}{tag}"


def parse_xml(data: bytes) -> Any:
    """Parse a part. No entity expansion and no network, whatever the file says."""
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
    return etree.fromstring(data, parser)


def relationships(archive: zipfile.ZipFile, part: str) -> list[tuple[str, str]]:
    """``(type, resolved zip path)`` for every relationship of *part*.

    ``part`` is a zip path such as ``word/document.xml``; the empty string is
    the package itself. External targets are skipped.
    """
    folder, _, name = part.rpartition("/")
    rels_name = f"{folder}/_rels/{name}.rels" if part else "_rels/.rels"
    rels_name = rels_name.lstrip("/")
    if rels_name not in archive.namelist():
        return []
    root = parse_xml(archive.read(rels_name))
    found: list[tuple[str, str]] = []
    for rel in root.iter(f"{{{REL_NS}}}Relationship"):
        if rel.get("TargetMode") == "External":
            continue
        target = rel.get("Target", "")
        if target.startswith("/"):
            resolved = target.lstrip("/")
        else:
            resolved = posixpath.normpath(posixpath.join(folder, target))
        found.append((rel.get("Type", ""), resolved))
    return found


def related_part(
    archive: zipfile.ZipFile, part: str, *, type_suffix: str, fallback: str
) -> str | None:
    """The part *part* relates to by a type ending in *type_suffix*, else the
    conventional *fallback* name when that exists, else None."""
    names = set(archive.namelist())
    for rel_type, target in relationships(archive, part):
        if rel_type.endswith(type_suffix) and target in names:
            return target
    return fallback if fallback in names else None
