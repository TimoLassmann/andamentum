"""andamentum.typeset — a 7-atom document typesetting system.

The typeset package converts structured documents (lists of *atom* dicts) into
styled HTML, and optionally PDF.  A document is a flat list where every item
has a ``kind`` drawn from one of the seven atom types:

``heading``
    A section heading with a ``content`` string and optional ``level`` (1–6).
``prose``
    Flowing paragraph text in ``content``.
``callout``
    A highlighted block (``content``) with an optional ``tone`` (``"info"``,
    ``"warning"``, ``"success"``, ``"note"``, ``"quote"``).
``items``
    A list of entries — bullet, numbered, or key/value pairs depending on
    ``variant`` (``"pairs"``, ``"right"``, ``"left"``).
``aside``
    A sidebar / tangential block; accepts ``content`` or ``groups``.
``card``
    A self-contained panel with ``content`` and optional metadata.
``reference``
    A bibliographic or citation block with ``content``.

Quick-start example::

    from andamentum.typeset import render, get_style

    doc = [
        {"kind": "heading", "content": "Hello, typeset!", "level": 1},
        {"kind": "prose", "content": "This is the first paragraph."},
        {"kind": "callout", "content": "Important note.", "tone": "info"},
    ]

    html = render(doc, style="article", title="My Document")
"""

from __future__ import annotations

from .renderer import render, render_to_file
from .styles import STYLES, get_style

__all__ = [
    "render",
    "render_to_file",
    "render_pdf",
    "STYLES",
    "get_style",
]

try:
    from .renderer import render_pdf
except ImportError:  # weasyprint not installed
    pass  # render_pdf remains absent from the namespace
