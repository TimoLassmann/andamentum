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

# === Functions you can wrap as agent tools ===
# `Report` is a class — wrap its methods (`heading`, `prose`, `callout`,
# `save`, etc.) as tools, or use the standalone builder functions below.
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
from .styles import get_style

# === Result/data types (returned by the above; not tools themselves) ===
from .styles import STYLES

__all__ = [
    # Functions / callables
    "Report",
    "heading",
    "prose",
    "callout",
    "items",
    "aside",
    "card",
    "reference",
    "render",
    "render_to_file",
    "render_pdf",
    "get_style",
    # Data types
    "STYLES",
]

try:
    from .renderer import render_pdf
except ImportError:
    pass
