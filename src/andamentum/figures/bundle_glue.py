"""Glue between :mod:`andamentum.figures` and :mod:`andamentum.compose.bundle`.

The figures module renders standalone image files; a bundle keeps its
images in ``figures/`` and references them from section bodies. This module
ties them together: render a figure to PNG and add it to a section of a
bundle in one call.

Coupling is one-directional: this module imports from both
``andamentum.figures`` and ``andamentum.compose.bundle``, but neither of the two
depends on this glue. Removing it leaves both modules intact.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .render import figure as render_figure

if TYPE_CHECKING:  # pragma: no cover
    from andamentum.compose.bundle import Bundle, Section


def insert_figure(
    bundle: "Bundle",
    section_id: str,
    *,
    caption: str,
    label: str,
    expected_revision: int | None = None,
    **chart_kwargs: Any,
) -> "Section":
    """Render a figure and append it to a section of a bundle.

    The figure is rendered as PNG (python-docx cannot embed PDF) into a
    temporary directory and copied into the bundle's ``figures/`` folder by
    :meth:`Bundle.add_figure`, which names the file after *label*.

    Args:
        bundle: The ``Bundle`` to add to.
        section_id: Id of the section that receives the figure.
        caption: Figure caption.
        label: Figure label (e.g. ``"benchmark"``); also the file name.
        expected_revision: The section revision you read. ``None`` uses the
            section's current revision.
        **chart_kwargs: Forwarded to :func:`andamentum.figures.figure`
            (``data``, ``kind``, ``x``, ``y``, ``style``, etc.).

    Returns:
        The updated section.
    """
    section = bundle.section(section_id)  # fails before any rendering work
    revision = section.revision if expected_revision is None else expected_revision
    with tempfile.TemporaryDirectory(prefix="bundle-figure-") as tmp:
        rendered = Path(tmp) / "figure.png"
        chart_kwargs["output"] = rendered
        render_figure(**chart_kwargs)
        return bundle.add_figure(
            section_id,
            image=rendered,
            caption=caption,
            label=label,
            expected_revision=revision,
        )
