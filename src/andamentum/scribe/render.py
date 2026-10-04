"""Render dispatch: docx, md, html, txt."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

from .bundle import _atomic_write
from .markdown import ImageBlock, ListBlock, ParagraphBlock, TableBlock, parse_blocks
from .models import Reference, ScribeError, Section
from .parser import inline_runs

if TYPE_CHECKING:  # pragma: no cover
    from .bundle import Bundle

_FORMATS = ("docx", "md", "html", "txt")


def _plain(text: str) -> str:
    return "".join(chunk for chunk, _styles in inline_runs(text))


def _relative_image(bundle_root: Path, output_dir: Path) -> Callable[[str], str]:
    """Image paths as seen from *output_dir* (bundle bodies are bundle-relative)."""

    def convert(path: str) -> str:
        if Path(path).is_absolute() or "://" in path:
            return path
        return os.path.relpath((bundle_root / path).resolve(), output_dir.resolve())

    return convert


def _markdown(
    title: str,
    sections: list[Section],
    references: list[Reference],
    text_filter: Callable[[str], str] | None,
    image_path: Callable[[str], str],
) -> str:
    from .markdown import blocks_to_markdown

    parts = [f"# {title}"]
    for section in sections:
        body = text_filter(section.body) if text_filter else section.body
        blocks = parse_blocks(body)
        for block in blocks:
            if isinstance(block, ImageBlock):
                block.path = image_path(block.path)
        parts.append(f"## {section.title}")
        if blocks:
            parts.append(blocks_to_markdown(blocks))
    if references:
        parts.append("## References")
        parts.append(
            "\n".join(
                f"- [{r.key}] {r.text}" + (f" <{r.url}>" if r.url else "")
                for r in references
            )
        )
    return "\n\n".join(parts) + "\n"


def _text(
    title: str,
    sections: list[Section],
    text_filter: Callable[[str], str] | None,
) -> str:
    parts = [title]
    for section in sections:
        body = text_filter(section.body) if text_filter else section.body
        parts.append(section.title)
        for block in parse_blocks(body):
            if isinstance(block, ParagraphBlock):
                parts.append(_plain(block.text))
            elif isinstance(block, ListBlock):
                parts.append(
                    "\n".join(
                        f"{i}. {_plain(item)}" if block.ordered else f"- {_plain(item)}"
                        for i, item in enumerate(block.items, start=1)
                    )
                )
            elif isinstance(block, TableBlock):
                parts.append(
                    "\n".join(" | ".join(_plain(c) for c in row) for row in block.rows)
                )
            elif isinstance(block, ImageBlock):
                parts.append(
                    f"[Figure: {_plain(block.caption)}]"
                    if block.caption
                    else "[Figure]"
                )
    return "\n\n".join(parts) + "\n"


def render_bundle(
    bundle: "Bundle",
    output_path: Path,
    *,
    format: Literal["docx", "md", "html", "txt"],
    text_filter: Callable[[str], str] | None,
    image_root: Path | None = None,
) -> Path:
    """Render *bundle* to *output_path* in *format*.

    Bundle-relative image paths resolve against *image_root* (default: the bundle).
    """
    root = image_root if image_root is not None else bundle.path
    if format not in _FORMATS:
        raise ScribeError(f"unknown format {format!r}; valid formats: {list(_FORMATS)}")
    sections = bundle.sections()
    references = bundle.references()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "docx":
        from .render_docx import render_to_docx

        render_to_docx(
            title=bundle.title,
            sections=sections,
            references=references,
            root=root,
            output_path=output_path,
            text_filter=text_filter,
        )
        return output_path

    image_path = _relative_image(root, output_path.parent)
    if format == "md":
        text = _markdown(bundle.title, sections, references, text_filter, image_path)
    elif format == "txt":
        text = _text(bundle.title, sections, text_filter)
    else:
        from andamentum.typeset import render as typeset_render

        from .render_typeset import to_typeset_atoms

        atoms = to_typeset_atoms(
            title=bundle.title,
            sections=sections,
            references=references,
            text_filter=text_filter,
            image_path=image_path,
        )
        text = typeset_render(atoms, style="article", title=bundle.title)

    _atomic_write(output_path, text)
    return output_path
