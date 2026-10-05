"""Shared builders for check tests (synthetic views only)."""

from __future__ import annotations

from pathlib import Path

from andamentum.compose.harness.kernel import (
    CheckContext,
    DocumentView,
    FieldToken,
    Paragraph,
)

CTX = CheckContext(store=None, offline=True)


def make_view(
    *paragraphs: tuple[str, int | None] | tuple[str, int | None, str],
    fields: list[FieldToken] | None = None,
    path: str = "x.docx",
) -> DocumentView:
    """Paragraphs as (text, heading_level[, style]); ids are p0, p1, ..."""
    built = []
    for i, item in enumerate(paragraphs):
        text, level = item[0], item[1]
        style = item[2] if len(item) == 3 else ("Heading 1" if level else "Normal")  # type: ignore[misc]
        built.append(
            Paragraph(id=f"p{i}", index=i, style=style, heading_level=level, text=text)
        )
    return DocumentView(
        path=Path(path),
        project_id=None,
        paragraphs=built,
        anchors=[],
        fields=fields or [],
        comments=[],
    )


def field(
    kind: str,
    manager: str = "none",
    *,
    data: dict | None = None,
    pid: str = "p0",
    display: str = "[1]",
) -> FieldToken:
    return FieldToken(
        kind=kind,  # type: ignore[arg-type]
        manager=manager,  # type: ignore[arg-type]
        paragraph_id=pid,
        start=0,
        end=len(display),
        display=display,
        data=data or {},
    )
