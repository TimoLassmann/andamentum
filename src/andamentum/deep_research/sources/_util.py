"""Small readers shared by the adapters: whitespace, years, ids out of API fields."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime

from lxml import etree  # type: ignore[attr-defined]  # lxml ships no stubs
from pydantic import JsonValue

from andamentum.references import Work

from .base import SourceError

SNIPPET_CHARS = 500
_SPACE = re.compile(r"\s+")


def squash(text: object) -> str:
    """*text* with every run of whitespace as one space, trimmed; non-strings read as empty."""
    return _SPACE.sub(" ", text).strip() if isinstance(text, str) else ""


def as_mapping(value: object) -> Mapping[str, JsonValue]:
    return value if isinstance(value, dict) else {}


def as_list(value: object) -> list[JsonValue]:
    return value if isinstance(value, list) else []


def as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def year_of_date(value: object) -> int | None:
    """The year of an ISO date or of a bare year; None when there is none."""
    text = (
        str(value)
        if isinstance(value, int) and not isinstance(value, bool)
        else squash(value)
    )
    if len(text) >= 4 and text[:4].isdigit():
        year = int(text[:4])
        if 1000 <= year <= datetime.now().year + 2:
            return year
    return None


def snippet_of(work: Work, fallback: str = "") -> str:
    return (work.abstract or fallback)[:SNIPPET_CHARS]


def expect_mapping(
    source: str, value: JsonValue, *, what: str
) -> Mapping[str, JsonValue]:
    """A JSON object, or a ``parse`` error naming what was expected."""
    if not isinstance(value, dict):
        raise SourceError(source, f"expected a JSON object for {what}", kind="parse")
    return value


_XML_PARSER = etree.XMLParser(
    resolve_entities=False,
    no_network=True,
    load_dtd=False,
    dtd_validation=False,
    huge_tree=False,
    recover=False,
)


def parse_xml(data: bytes) -> etree._Element:
    """The root of *data*. No entity is resolved and no DTD or network is read, so a
    hostile document cannot reach a file or a host. Raises ``etree.XMLSyntaxError``."""
    return etree.fromstring(data, _XML_PARSER)
