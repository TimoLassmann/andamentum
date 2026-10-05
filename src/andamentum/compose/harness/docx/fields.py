"""Field instructions: which reference manager wrote a field, and what it says.

Pure functions over the instruction text of a Word field (the concatenated
``w:instrText`` of a complex field, or the ``w:instr`` of a simple one). The
reader walks the XML and finds the fields; this module classifies them.

A payload that does not parse is not dropped: the token is still produced and
``data`` holds ``{"parse_error": <reason>, "raw": <instruction>}`` so the
cross-reference check can report it.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from typing import Literal

from lxml import etree  # type: ignore

from .ooxml import parse_xml

FieldKind = Literal["citation", "bibliography", "crossref", "other"]
Manager = Literal["zotero", "endnote", "mendeley", "word", "none"]

_REFERENCE_FIELDS = frozenset({"REF", "PAGEREF", "NOTEREF"})


def classify_field(instruction: str) -> tuple[FieldKind, Manager, dict[str, object]]:
    """``(kind, manager, data)`` for a field instruction."""
    text = instruction.strip()
    parts = text.split(None, 1)
    if not parts:
        return "other", "none", {"instruction": ""}
    name = parts[0].upper()
    rest = parts[1] if len(parts) > 1 else ""

    if name == "ADDIN":
        return _addin(rest.strip(), instruction)
    if name in _REFERENCE_FIELDS:
        return "crossref", "word", _reference_data(name, rest)
    if name == "SEQ":
        return "crossref", "word", _reference_data(name, rest)
    if name == "CITATION":
        tokens = _tokens(rest)
        return "citation", "word", {"field": name, "tag": tokens[0] if tokens else ""}
    if name == "BIBLIOGRAPHY":
        return "bibliography", "word", {"field": name}
    return "other", "word", {"field": name, "instruction": text}


def _addin(rest: str, instruction: str) -> tuple[FieldKind, Manager, dict[str, object]]:
    if rest.startswith("ZOTERO_ITEM"):
        return (
            "citation",
            "zotero",
            _json_payload(rest, instruction, prefix="ZOTERO_ITEM"),
        )
    if rest.startswith("ZOTERO_BIBL"):
        return (
            "bibliography",
            "zotero",
            _json_payload(rest, instruction, prefix="ZOTERO_BIBL"),
        )
    if rest.startswith("CSL_CITATION"):
        return (
            "citation",
            "mendeley",
            _json_payload(rest, instruction, prefix="CSL_CITATION"),
        )
    if rest.startswith("Mendeley Bibliography"):
        return (
            "bibliography",
            "mendeley",
            _json_payload(rest, instruction, prefix="Mendeley Bibliography"),
        )
    if rest.startswith("EN.CITE.DATA"):
        return (
            "other",
            "endnote",
            _endnote_data(rest[len("EN.CITE.DATA") :], instruction),
        )
    if rest.startswith("EN.CITE"):
        return "citation", "endnote", _endnote_cite(rest[len("EN.CITE") :], instruction)
    if rest.startswith("EN.REFLIST"):
        return "bibliography", "endnote", {}
    return "other", "none", {"instruction": instruction.strip()}


def _json_payload(rest: str, instruction: str, *, prefix: str) -> dict[str, object]:
    """The JSON object after *prefix*; a ``CSL_CITATION`` lead-in and a trailing
    ``CSL_BIBLIOGRAPHY`` marker are ignored."""
    body = rest[len(prefix) :].strip()
    body = re.sub(r"^CSL_CITATION\b\s*", "", body)
    body = re.sub(r"\s*CSL_BIBLIOGRAPHY\s*$", "", body)
    if not body:
        return {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        return {
            "parse_error": f"invalid JSON: {exc.msg} (char {exc.pos})",
            "raw": instruction.strip(),
        }
    if not isinstance(parsed, dict):
        return {
            "parse_error": "JSON payload is not an object",
            "raw": instruction.strip(),
        }
    return parsed


def _endnote_cite(body: str, instruction: str) -> dict[str, object]:
    body = body.strip()
    if not body:
        return {
            "parse_error": "EN.CITE has no <EndNote> payload",
            "raw": instruction.strip(),
        }
    try:
        root = parse_xml(body.encode())
    except etree.XMLSyntaxError as exc:
        return {
            "parse_error": f"invalid EndNote XML: {exc}",
            "raw": instruction.strip(),
        }
    return {"cites": _endnote_cites(root)}


def _endnote_data(body: str, instruction: str) -> dict[str, object]:
    """EN.CITE.DATA carries the record base64-encoded, when it carries it at all."""
    body = body.strip()
    if not body:
        return {"encoded": False}
    try:
        xml = base64.b64decode(body, validate=True)
        root = parse_xml(xml)
    except (binascii.Error, etree.XMLSyntaxError) as exc:
        return {
            "parse_error": f"undecodable EN.CITE.DATA: {exc}",
            "raw": instruction.strip(),
        }
    return {"encoded": True, "cites": _endnote_cites(root)}


def _endnote_cites(root: object) -> list[dict[str, object]]:
    cites: list[dict[str, object]] = []
    for cite in root.iter("Cite"):  # type: ignore[attr-defined]
        record = cite.find("record")
        entry: dict[str, object] = {
            "author": _text(cite, "Author"),
            "year": _text(cite, "Year"),
            "rec_number": _text(cite, "RecNum"),
            "display_text": _text(cite, "DisplayText"),
        }
        if record is not None:
            entry["title"] = _text(record, "titles/title")
            entry["authors"] = [
                "".join(a.itertext()).strip()
                for a in record.iterfind("contributors/authors/author")
            ]
            entry["record_year"] = _text(record, "dates/year")
            entry["doi"] = _text(record, "electronic-resource-num")
        cites.append(entry)
    return cites


def _text(element: object, path: str) -> str:
    found = element.find(path)  # type: ignore[attr-defined]
    return "".join(found.itertext()).strip() if found is not None else ""


def _tokens(rest: str) -> list[str]:
    """Non-switch tokens of a field's argument text; a quoted token stays whole."""
    tokens = re.findall(r'"([^"]*)"|(\\\S+)|(\S+)', rest)
    return [quoted or bare for quoted, switch, bare in tokens if not switch]


def _switches(rest: str) -> list[str]:
    return re.findall(r"\\\S+", rest)


def _reference_data(name: str, rest: str) -> dict[str, object]:
    tokens = _tokens(rest)
    return {
        "field": name,
        "target": tokens[0] if tokens else "",
        "switches": _switches(rest),
    }
