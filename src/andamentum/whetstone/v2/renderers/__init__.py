"""Renderers turn a ``ReviewResult`` into a human-readable artifact.

Three formats:
  • ``render_markdown`` — plain markdown report (best for terminals,
    PR comments, AI agents that consume text).
  • ``render_html``     — styled HTML via ``andamentum.typeset``
    (best for sharing in a browser).
  • ``render_docx``     — Word document with edits as tracked changes
    and findings as Word comments (best for the author's own
    editing workflow). Reuses v1's ``whetstone.docx.finalization``
    machinery via a small adapter.

All three consume the same ``ReviewResult`` shape — agents that produce
review reports for downstream consumers can pick whichever format the
consumer prefers without changing the data model.
"""

from .docx import render_docx
from .html import render_html
from .markdown import render_markdown

__all__ = ["render_docx", "render_html", "render_markdown"]
