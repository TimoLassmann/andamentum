"""Builder functions and Report class for andamentum.typeset.

Three ways to build documents, from simplest to most flexible:

1. **Report builder** — natural Python, reads like a document outline::

        from andamentum.typeset import Report

        r = Report(style="article")
        r.heading("My Report", meta={"date": "2026-04-16"})
        r.callout("Key finding.")
        r.prose("## Summary\\n\\nBody text...")
        r.save("report.html")

2. **Builder functions** — return plain dicts, compose into a list::

        from andamentum.typeset import render, heading, prose, callout

        html = render([heading("My Report"), prose("Body.")])

3. **Raw dicts** — most flexible, best for agents::

        render([{"kind": "heading", "content": "My Report"}])

All three produce the same output through the same renderer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .renderer import render as _render
from .renderer import render_to_file as _render_to_file


# ---------------------------------------------------------------------------
# Builder functions — one per atom, each returns a plain dict
# ---------------------------------------------------------------------------


def heading(content: str, **kwargs: Any) -> dict[str, Any]:
    """Build a ``heading`` atom dict.

    Args:
        content: Document title text (markdown).
        subtitle: Optional subtitle string.
        meta: Optional metadata — a string or dict of key/value pairs.
        id: Optional DOM id rendered on the ``<header>`` element (for anchors).
    """
    return {"kind": "heading", "content": content, **kwargs}


def prose(content: str, **kwargs: Any) -> dict[str, Any]:
    """Build a ``prose`` atom dict.

    Args:
        content: Markdown body text.
        heading: Optional section heading rendered as ``<h2>``.
        id: Optional DOM id rendered on the ``<h2>`` when ``heading`` is set,
            otherwise on the ``<section>`` (for anchors).
    """
    return {"kind": "prose", "content": content, **kwargs}


def callout(content: str, **kwargs: Any) -> dict[str, Any]:
    """Build a ``callout`` atom dict.

    Args:
        content: Markdown text for the callout.
        tone: Optional — one of ``info``, ``warning``, ``success``,
              ``note``, ``quote``.
    """
    return {"kind": "callout", "content": content, **kwargs}


def items(entries: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
    """Build an ``items`` atom dict.

    Args:
        entries: List of ``{"label": ..., "body": ...}`` dicts.
        variant: Optional — ``"pairs"`` (default), ``"right"``, or ``"left"``.
        heading: Optional section heading.
    """
    return {"kind": "items", "entries": entries, **kwargs}


def aside(**kwargs: Any) -> dict[str, Any]:
    """Build an ``aside`` atom dict.

    Pass either ``content="markdown"`` for a simple aside, or
    ``groups={"Title": {"key": "value"}}`` for a sidebar metadata grid.
    """
    return {"kind": "aside", **kwargs}


def card(content: str, **kwargs: Any) -> dict[str, Any]:
    """Build a ``card`` atom dict.

    Args:
        content: Main statement (markdown).
        badge: Optional status label (e.g., ``"supported"``, ``"challenged"``).
        refs: Optional list of citation identifiers.
        source: Optional URL.
        source_label: Optional display text for ``source`` (defaults to the URL).
        details: Optional collapsible details (markdown).
        id: Optional DOM id rendered on the card container (for anchors).
    """
    return {"kind": "card", "content": content, **kwargs}


def reference(content: str, **kwargs: Any) -> dict[str, Any]:
    """Build a ``reference`` atom dict.

    Args:
        content: Reference text (markdown).
        number: Optional sequence number.
        source: Optional URL.
        source_label: Optional display text for ``source`` (defaults to the URL).
        badge: Optional status label.
        group: Optional grouping key (consecutive references with the
               same group are clustered under a heading).
    """
    return {"kind": "reference", "content": content, **kwargs}


# ---------------------------------------------------------------------------
# Report class — fluent builder that wraps the atom list
# ---------------------------------------------------------------------------


class Report:
    """Fluent document builder.

    Each atom type is a method that appends to an internal list. Call
    :meth:`save`, :meth:`save_pdf`, or :meth:`render` when done.

    Example::

        from andamentum.typeset import Report

        r = Report(style="article")
        r.heading("Weekly Status", meta={"date": "2026-04-16"})
        r.callout("Shipped semantic routing.", tone="success")
        r.prose("## Details\\n\\nThe router is live...")
        r.save("status.html")
    """

    def __init__(self, style: str = "article", **kwargs: Any) -> None:
        self.style = style
        self._kwargs = kwargs
        self._atoms: list[dict[str, Any]] = []

    # -- Atom methods (one per kind) --

    def heading(self, content: str, **kw: Any) -> "Report":
        """Append a ``heading`` atom."""
        self._atoms.append({"kind": "heading", "content": content, **kw})
        return self

    def prose(self, content: str, **kw: Any) -> "Report":
        """Append a ``prose`` atom."""
        self._atoms.append({"kind": "prose", "content": content, **kw})
        return self

    def callout(self, content: str, **kw: Any) -> "Report":
        """Append a ``callout`` atom."""
        self._atoms.append({"kind": "callout", "content": content, **kw})
        return self

    def items(self, entries: list[dict[str, str]], **kw: Any) -> "Report":
        """Append an ``items`` atom."""
        self._atoms.append({"kind": "items", "entries": entries, **kw})
        return self

    def aside(self, **kw: Any) -> "Report":
        """Append an ``aside`` atom."""
        self._atoms.append({"kind": "aside", **kw})
        return self

    def card(self, content: str, **kw: Any) -> "Report":
        """Append a ``card`` atom."""
        self._atoms.append({"kind": "card", "content": content, **kw})
        return self

    def reference(self, content: str, **kw: Any) -> "Report":
        """Append a ``reference`` atom."""
        self._atoms.append({"kind": "reference", "content": content, **kw})
        return self

    # -- Output methods --

    @property
    def atoms(self) -> list[dict[str, Any]]:
        """The accumulated atom list (read-only copy)."""
        return list(self._atoms)

    def render(self) -> str:
        """Render to an HTML string."""
        return _render(self._atoms, style=self.style, **self._kwargs)

    def save(self, path: str | Path) -> Path:
        """Render and write to an HTML file."""
        return _render_to_file(self._atoms, path, style=self.style, **self._kwargs)

    def save_pdf(self, path: str | Path, **kw: Any) -> Path:
        """Render and write to a PDF file. Requires WeasyPrint."""
        from .renderer import render_pdf

        return render_pdf(self._atoms, path, style=self.style, **self._kwargs, **kw)

    def __len__(self) -> int:
        return len(self._atoms)

    def __repr__(self) -> str:
        return f"Report({len(self._atoms)} atoms, style={self.style!r})"
