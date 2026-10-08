"""Deterministic document view shared by the panel's prompt-assembling workers.

Section-title outline + truncated markdown body — the same view
every expert receives.
"""

from __future__ import annotations

from ..model import Section

# Cap on how much markdown reaches the per-expert review prompt. 30k chars,
# the size the expert prompts were calibrated on.
DOCUMENT_VIEW_MAX_CHARS: int = 30_000


def build_document_view(source: str, sections: list[Section]) -> str:
    """Build a panel-friendly document view: section-title outline +
    truncated markdown body."""
    outline = "\n".join(f"  • {s.title}" for s in sections if s.title)
    body = source[:DOCUMENT_VIEW_MAX_CHARS]
    truncated = (
        ""
        if len(source) <= DOCUMENT_VIEW_MAX_CHARS
        else ("\n\n[...document truncated for review...]")
    )
    if outline:
        return f"DOCUMENT OUTLINE:\n{outline}\n\nDOCUMENT BODY:\n{body}{truncated}"
    return f"DOCUMENT BODY:\n{body}{truncated}"
