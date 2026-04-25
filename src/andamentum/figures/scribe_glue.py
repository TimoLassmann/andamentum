"""Glue between :mod:`andamentum.figures` and :mod:`andamentum.scribe`.

The figures module renders standalone PNG/PDF files; scribe stores
documents whose ``Figure`` blocks reference image paths. This module
ties them together: render a figure to PNG and insert it into a
named section of a scribe document in one call.

Coupling is one-directional: this module imports from both
``andamentum.figures`` and ``andamentum.scribe``, but neither of the
two depends on this glue. Removing it leaves both modules intact.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .render import figure as render_figure

if TYPE_CHECKING:  # pragma: no cover
    from andamentum.scribe.api import Document


_RASTER_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def insert_figure(
    doc: "Document",
    section: str,
    *,
    output_dir: str | Path,
    caption: str,
    label: str,
    width_in: float | None = None,
    position: str = "end",
    filename: str | None = None,
    **chart_kwargs: Any,
) -> str:
    """Render a figure and insert it into a scribe document section.

    The figure is rendered as a raster file (PNG by default) because
    scribe's docx renderer embeds via python-docx, which does not
    accept PDF. The file is written under ``output_dir`` and the
    scribe ``Figure`` block stores its path.

    Args:
        doc: Scribe ``Document`` to insert into.
        section: Name of the section to receive the figure.
        output_dir: Directory where the rendered file will be written.
            Created if missing.
        caption: Figure caption.
        label: Figure label (e.g. ``"fig:benchmark"``); also used to
            derive the default filename.
        width_in: Optional width in inches for the docx render.
        position: ``"end"`` (after the section's last block) or
            ``"start"`` (immediately after the heading).
        filename: Override the auto-derived filename. If the extension
            is not a python-docx-compatible raster format, ``.png`` is
            appended.
        **chart_kwargs: Forwarded to :func:`andamentum.figures.figure`
            (``data``, ``kind``, ``x``, ``y``, ``style``, etc.).

    Returns:
        The block id of the inserted figure.
    """
    from andamentum.scribe.api import Figure as ScribeFigure

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = filename or f"{label.replace(':', '_')}.png"
    if not fname.lower().endswith(_RASTER_EXTS):
        fname = f"{fname}.png"
    out_path = out_dir / fname

    chart_kwargs["output"] = out_path
    render_figure(**chart_kwargs)

    spec = ScribeFigure(
        path=str(out_path),
        caption=caption,
        label=label,
        width_in=width_in,
    )
    return doc.insert_into_section(section, spec, position=position)
