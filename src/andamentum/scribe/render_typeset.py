"""Map scribe blocks to typeset atom dicts.

Typeset is the package's display layer (HTML+PDF). Scribe owns the
authoring schema; this module is the thin adapter that lets typeset
render scribe documents without typeset learning anything new.

For v1 we map:
  paragraph -> prose
  heading   -> heading
  figure    -> card (with embedded <img> + caption)
  table     -> prose with a markdown table body (typeset already loads
               python-markdown's tables extension)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .api import Document


def _table_to_markdown(rows: list[list[str]], header_row: bool) -> str:
    if not rows:
        return ""
    if header_row:
        head, *body = rows
    else:
        head = [""] * len(rows[0])
        body = rows
    out = ["| " + " | ".join(head) + " |"]
    out.append("| " + " | ".join(["---"] * len(head)) + " |")
    for r in body:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def to_typeset_atoms(doc: "Document") -> list[dict[str, Any]]:
    """Convert the document's blocks to typeset atom dicts."""
    atoms: list[dict[str, Any]] = []
    for blk in doc.query():
        if blk.type == "paragraph":
            atoms.append({"kind": "prose", "content": blk.content})
        elif blk.type == "heading":
            atoms.append(
                {
                    "kind": "heading",
                    "content": blk.content,
                    "level": int(blk.metadata.get("level", 1)),
                }
            )
        elif blk.type == "figure":
            path = blk.metadata.get("path", "")
            caption = blk.metadata.get("caption", "")
            label = blk.metadata.get("label", "")
            body = (
                f'<img src="{path}" alt="{label}" />\n\n**{caption}**'
                if caption
                else f'<img src="{path}" />'
            )
            atoms.append({"kind": "card", "content": body})
        elif blk.type == "table":
            md_table = _table_to_markdown(
                blk.metadata.get("rows", []),
                bool(blk.metadata.get("header_row", True)),
            )
            caption = blk.metadata.get("caption", "")
            content = md_table if not caption else f"{md_table}\n\n*{caption}*"
            atoms.append({"kind": "prose", "content": content})
        else:  # pragma: no cover — schema enforces the type literal
            raise ValueError(f"Unknown block type for typeset render: {blk.type!r}")
    return atoms
