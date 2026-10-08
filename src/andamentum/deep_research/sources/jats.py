"""JATS XML (what Europe PMC serves as full text) to markdown text.

Reads the title, the abstract and the body's sections and paragraphs. Left out
because they carry no sentence a quote could be taken from: formulas, tables,
figures (captions too) and the reference list. The parser (``_util.parse_xml``)
resolves no entity and fetches no DTD.
"""

from __future__ import annotations

from lxml import etree  # type: ignore[attr-defined]  # lxml ships no stubs

from ._util import parse_xml, squash

# Subtrees dropped from running text.
_SKIP = frozenset(
    {
        "math",
        "tex-math",
        "disp-formula",
        "inline-formula",
        "table-wrap",
        "table-wrap-foot",
        "fig",
        "ref-list",
        "supplementary-material",
        "object-id",
    }
)
_MAX_HEADING_LEVEL = 4


class JatsError(ValueError):
    """The document is not JATS, or not well-formed XML."""


def jats_to_text(xml: bytes) -> str:
    """Markdown for the article in *xml*: title, abstract, then the body's sections."""
    try:
        root = parse_xml(xml)
    except etree.XMLSyntaxError as exc:
        raise JatsError(f"not well-formed XML: {exc}") from exc
    article = root if _name(root) == "article" else _first(root, "article")
    if article is None:
        raise JatsError("no <article> element")
    blocks: list[str] = []
    title = _first(article, "article-title")
    if title is not None and (text := _inline(title)):
        blocks.append(f"# {text}")
    for abstract in article.iterfind(".//front//abstract"):
        if abstract.get("abstract-type") in {"graphical", "toc"}:
            continue
        blocks.append("## Abstract")
        blocks += _blocks(abstract, level=3)
    body = _first(article, "body")
    if body is not None:
        blocks += _blocks(body, level=2)
    return "\n\n".join(b for b in blocks if b)


def _blocks(parent: etree._Element, *, level: int) -> list[str]:
    out: list[str] = []
    for child in parent:
        name = _name(child)
        if name == "p":
            if text := _inline(child):
                out.append(text)
        elif name == "sec":
            heading = _first_child(child, "title")
            if heading is not None and (text := _inline(heading)):
                out.append(f"{'#' * min(level, _MAX_HEADING_LEVEL)} {text}")
            out += _blocks(child, level=level + 1)
        elif name == "list":
            out += [t for item in child if (t := _inline(item))]
        elif name in {"boxed-text", "disp-quote"}:
            out += _blocks(child, level=level)
    return out


def _inline(element: etree._Element) -> str:
    """The running text of *element*, skipped subtrees removed, whitespace squashed."""
    parts: list[str] = [element.text or ""]
    for child in element:
        name = _name(child)
        if name is not None and name not in _SKIP:
            parts.append(_inline(child))
        parts.append(child.tail or "")
    return squash("".join(parts))


def _name(element: etree._Element) -> str | None:
    """The tag without its namespace; None for a comment or processing instruction."""
    tag = element.tag
    return etree.QName(tag).localname if isinstance(tag, str) else None


def _first(element: etree._Element, name: str) -> etree._Element | None:
    for found in element.iter():
        if _name(found) == name:
            return found
    return None


def _first_child(element: etree._Element, name: str) -> etree._Element | None:
    for child in element:
        if _name(child) == name:
            return child
    return None
