"""Renderers for whetstone sharpen_document() results.

Three output formats from the same ReviewResult:
- render_docx: Word document with track changes and prepended report
- render_html: Self-contained HTML report (built on andamentum.typeset)
- render_diff: Lightweight markdown diff view

Plus a text utility:
- apply_patches: Apply accepted patches to plain text content
"""

from .diff import apply_patches, render_diff
from .docx import render_docx
from .html import render_html

__all__ = [
    "apply_patches",
    "render_diff",
    "render_docx",
    "render_html",
]
