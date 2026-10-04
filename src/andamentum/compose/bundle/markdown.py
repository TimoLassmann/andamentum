"""The markdown subset a section body may use, parsed into blocks.

Supported: paragraphs, ``- `` bullet lists, ``1. `` numbered lists, pipe
tables (first row is the header) and whole-line images
``![caption](path){#fig:label}``. Blank lines separate blocks. Anything else is
a plain paragraph — in particular a ``#`` heading line is never a heading
(``Bundle.validate`` reports it as an error). Inline runs (bold, italic, code)
are handled by :mod:`andamentum.compose.bundle.parser`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_BULLET_RE = re.compile(r"^[-] +(?P<text>.*\S)\s*$")
_NUMBERED_RE = re.compile(r"^\d+[.)] +(?P<text>.*\S)\s*$")
_IMAGE_RE = re.compile(
    r"^!\[(?P<caption>.*)\]\((?P<path>[^)\s]+)\)(?:\{#(?:fig:)?(?P<label>[^}\s]+)\})?\s*$"
)
_TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
HEADING_RE = re.compile(r"^ {0,3}#{1,6}(\s|$)")


@dataclass
class ParagraphBlock:
    text: str


@dataclass
class ListBlock:
    ordered: bool
    items: list[str] = field(default_factory=list)


@dataclass
class TableBlock:
    rows: list[list[str]]


@dataclass
class ImageBlock:
    caption: str
    path: str
    label: str


Block = ParagraphBlock | ListBlock | TableBlock | ImageBlock


def _table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def parse_blocks(body: str) -> list[Block]:
    """Split *body* into blocks of the supported subset."""
    blocks: list[Block] = []
    paragraph: list[str] = []
    current_list: ListBlock | None = None
    table: list[str] = []

    def flush() -> None:
        nonlocal paragraph, current_list, table
        if paragraph:
            blocks.append(ParagraphBlock(" ".join(paragraph)))
        if current_list is not None:
            blocks.append(current_list)
        if table:
            rows = [
                _table_cells(line)
                for line in table
                if not _TABLE_SEPARATOR_RE.match(line.strip())
            ]
            if rows:
                width = max(len(r) for r in rows)
                blocks.append(TableBlock([r + [""] * (width - len(r)) for r in rows]))
        paragraph, current_list, table = [], None, []

    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        image = _IMAGE_RE.match(line)
        if image:
            flush()
            blocks.append(
                ImageBlock(
                    caption=image.group("caption"),
                    path=image.group("path"),
                    label=image.group("label") or "",
                )
            )
            continue
        if line.startswith("|"):
            if paragraph or current_list is not None:
                flush()
            table.append(line)
            continue
        bullet = _BULLET_RE.match(line)
        numbered = None if bullet else _NUMBERED_RE.match(line)
        if bullet or numbered:
            match = bullet or numbered
            assert match is not None
            ordered = numbered is not None
            if (
                table
                or paragraph
                or (current_list is not None and current_list.ordered != ordered)
            ):
                flush()
            if current_list is None:
                current_list = ListBlock(ordered=ordered)
            current_list.items.append(match.group("text"))
            continue
        if table or current_list is not None:
            flush()
        paragraph.append(line)
    flush()
    return blocks


def blocks_to_markdown(blocks: list[Block]) -> str:
    """Re-emit *blocks* as markdown (used by the html and txt renderers)."""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, ParagraphBlock):
            parts.append(block.text)
        elif isinstance(block, ListBlock):
            parts.append(
                "\n".join(
                    f"{i}. {item}" if block.ordered else f"- {item}"
                    for i, item in enumerate(block.items, start=1)
                )
            )
        elif isinstance(block, TableBlock):
            head, *rest = block.rows
            lines = ["| " + " | ".join(head) + " |"]
            lines.append("| " + " | ".join(["---"] * len(head)) + " |")
            lines.extend("| " + " | ".join(r) + " |" for r in rest)
            parts.append("\n".join(lines))
        else:
            label = f"{{#fig:{block.label}}}" if block.label else ""
            parts.append(f"![{block.caption}]({block.path}){label}")
    return "\n\n".join(parts)
