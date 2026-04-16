"""andamentum.typeset — beautiful documents from 7 atoms.

Three ways to build a document:

**Report builder** (easiest)::

    from andamentum.typeset import Report

    r = Report(style="article")
    r.heading("My Report", meta={"date": "2026-04-16"})
    r.callout("Key finding.")
    r.prose("## Summary\\n\\nBody text...")
    r.save("report.html")

**Builder functions** (composable)::

    from andamentum.typeset import render, heading, prose, callout

    html = render([heading("My Report"), prose("Body.")])

**Raw dicts** (most flexible)::

    from andamentum.typeset import render

    html = render([{"kind": "heading", "content": "My Report"}])

Seven atoms: ``heading``, ``prose``, ``callout``, ``items``, ``aside``,
``card``, ``reference``. Three styles: ``article``, ``cv``, ``report``.
"""

from __future__ import annotations

from .builders import (
    Report,
    aside,
    callout,
    card,
    heading,
    items,
    prose,
    reference,
)
from .renderer import render, render_to_file
from .styles import STYLES, get_style

__all__ = [
    # Report builder
    "Report",
    # Builder functions
    "heading",
    "prose",
    "callout",
    "items",
    "aside",
    "card",
    "reference",
    # Core renderer
    "render",
    "render_to_file",
    "render_pdf",
    # Styles
    "STYLES",
    "get_style",
]

try:
    from .renderer import render_pdf
except ImportError:
    pass
