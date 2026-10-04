"""Map a bundle to typeset atom dicts (HTML output).

The title becomes the ``heading`` atom; each section becomes one ``prose`` atom
that opens with a ``##`` heading, split around images, which become ``card``
atoms. References follow as ``reference`` atoms.
"""

from __future__ import annotations

import html
from typing import Any, Callable

from .markdown import ImageBlock, blocks_to_markdown, parse_blocks
from .models import Reference, Section


def to_typeset_atoms(
    *,
    title: str,
    sections: list[Section],
    references: list[Reference],
    text_filter: Callable[[str], str] | None,
    image_path: Callable[[str], str],
) -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = [{"kind": "heading", "content": title}]
    for section in sections:
        body = text_filter(section.body) if text_filter else section.body
        pending = [f"## {section.title}"]
        buffer: list = []

        def flush_text() -> None:
            if buffer:
                pending.append(blocks_to_markdown(buffer))
                buffer.clear()
            if pending:
                atoms.append({"kind": "prose", "content": "\n\n".join(pending)})
                pending.clear()

        for block in parse_blocks(body):
            if isinstance(block, ImageBlock):
                flush_text()
                src = html.escape(image_path(block.path), quote=True)
                alt = html.escape(block.label or block.caption, quote=True)
                content = f'<img src="{src}" alt="{alt}" />'
                if block.caption:
                    content += f"\n\n**{block.caption}**"
                atoms.append({"kind": "card", "content": content})
            else:
                buffer.append(block)
        flush_text()
    if references:
        atoms.append({"kind": "prose", "content": "## References"})
        for ref in references:
            atom: dict[str, Any] = {
                "kind": "reference",
                "content": f"[{ref.key}] {ref.text}",
            }
            if ref.url:
                atom["source"] = ref.url
            atoms.append(atom)
    return atoms
