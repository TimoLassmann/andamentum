"""How hits are shown to a model: full passages for judging, one line per hit for tools."""

from __future__ import annotations

from .store import Hit

_SNIPPET_CHARS = 300
_PASSAGE_CHARS = 1200


def passages_text(hits: list[Hit]) -> str:
    """Each hit as its label and up to 1200 characters of text."""
    if not hits:
        return "(no passages)"
    return "\n\n".join(f"[{h.citation}]\n{h.text[:_PASSAGE_CHARS]}" for h in hits)


def hit_line(h: Hit) -> str:
    snippet = " ".join(h.text.split())[:_SNIPPET_CHARS]
    return f"{h.citation} | {h.heading} | {snippet}"


def hit_lines(hits: list[Hit]) -> str:
    return "\n".join(hit_line(h) for h in hits) if hits else "(no results)"
