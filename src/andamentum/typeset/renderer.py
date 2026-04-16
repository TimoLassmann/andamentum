"""HTML/PDF renderer for andamentum.typeset.

Public API
----------
render(document, *, style, custom_css, title) -> str
render_to_file(document, output, **kwargs) -> Path
render_pdf(document, output, *, style, custom_css, title) -> Path
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from markdown.extensions import Extension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension
from markdown.extensions.toc import TocExtension

from andamentum.typeset.atoms import validate_document
from andamentum.typeset.styles import get_style

logger = logging.getLogger(__name__)

# Optional extensions — guarded so the module loads without them.
_EXTRA_EXTENSIONS: list[Extension | str] = []
try:
    from markdown.extensions.codehilite import CodeHiliteExtension

    _EXTRA_EXTENSIONS.append(CodeHiliteExtension())
except ImportError:
    pass

_EXTRA_EXTENSIONS.append("md_in_html")


# ---------------------------------------------------------------------------
# Markdown helper
# ---------------------------------------------------------------------------

def _md(text: str) -> str:
    """Convert *text* from Markdown to an HTML fragment."""
    import markdown

    extensions: list[Extension | str] = [
        TableExtension(),
        FencedCodeExtension(),
        TocExtension(permalink=False),
        *_EXTRA_EXTENSIONS,
    ]
    return markdown.markdown(str(text), extensions=extensions)


def _strip_p(html: str) -> str:
    """Remove a single wrapping ``<p>…</p>`` if present."""
    stripped = html.strip()
    return re.sub(r"^<p>(.*)</p>$", r"\1", stripped, flags=re.DOTALL)


# ---------------------------------------------------------------------------
# Atom renderers
# ---------------------------------------------------------------------------

def _render_heading(atom: dict[str, object]) -> str:
    """Render a *heading* atom to an HTML ``<header>`` block."""
    content = _strip_p(_md(str(atom["content"])))
    parts: list[str] = [
        '<header class="typeset-heading">',
        f"<h1>{content}</h1>",
    ]

    subtitle = atom.get("subtitle")
    if subtitle is not None:
        parts.append(f'<p class="typeset-subtitle">{subtitle}</p>')

    meta = atom.get("meta")
    if meta is not None:
        if isinstance(meta, dict):
            meta_text = " &middot; ".join(str(v) for v in meta.values())
        else:
            meta_text = str(meta)
        parts.append(f'<p class="typeset-meta">{meta_text}</p>')

    parts.append("</header>")
    return "\n".join(parts)


def _render_prose(atom: dict[str, object]) -> str:
    """Render a *prose* atom to an HTML ``<section>`` block."""
    parts: list[str] = []

    heading = atom.get("heading")
    if heading is not None:
        parts.append(f"<h2>{heading}</h2>")

    parts.append('<section class="typeset-prose">')
    parts.append(_md(str(atom["content"])))
    parts.append("</section>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_RENDERERS: dict[str, object] = {
    "heading": _render_heading,
    "prose": _render_prose,
}


def _render_atom(atom: dict[str, object]) -> str:
    """Dispatch *atom* to the appropriate renderer."""
    kind = str(atom.get("kind", "prose"))
    renderer = _RENDERERS.get(kind)
    if renderer is None:
        logger.warning("No renderer for kind %r; falling back to prose.", kind)
        return _render_prose(atom)
    return (renderer)(atom)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# HTML shell
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
<article class="typeset-document">
{body}
</article>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render(
    document: list[dict[str, object]] | str,
    *,
    style: str = "article",
    custom_css: str | None = None,
    title: str | None = None,
) -> str:
    """Render *document* to a complete HTML string.

    Parameters
    ----------
    document:
        Either a list of atom dicts or a plain Markdown string (treated as a
        single prose atom).
    style:
        Name of the built-in style to apply (see :mod:`andamentum.typeset.styles`).
    custom_css:
        Optional CSS string that replaces the built-in style entirely.
    title:
        Optional document title used in the ``<title>`` element.  Auto-detected
        from the first heading atom's ``content`` if not provided; defaults to
        ``"Document"``.

    Returns
    -------
    str
        Complete HTML document string.
    """
    # Normalise a plain string to a single prose atom.
    raw_atoms: Sequence[Mapping[str, object]]
    if isinstance(document, str):
        raw_atoms = [{"kind": "prose", "content": document}]
    else:
        raw_atoms = document

    validated = validate_document(list(raw_atoms))

    # Auto-detect title from first heading atom.
    if title is None:
        for atom in validated:
            if atom.get("kind") == "heading":
                title = _strip_p(_md(str(atom["content"])))
                # Remove any remaining HTML tags for a plain-text title.
                title = re.sub(r"<[^>]+>", "", title).strip()
                break
        if title is None:
            title = "Document"

    css = custom_css if custom_css is not None else get_style(style)

    body = "\n\n".join(_render_atom(atom) for atom in validated)

    return _HTML_TEMPLATE.format(title=title, css=css, body=body)


def render_to_file(
    document: list[dict[str, object]] | str,
    output: str | Path,
    **kwargs: object,
) -> Path:
    """Render *document* and write the HTML to *output*.

    Parameters
    ----------
    document:
        Atom list or plain Markdown string.
    output:
        Destination file path.
    **kwargs:
        Forwarded to :func:`render`.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    html = render(document, **kwargs)  # type: ignore[arg-type]
    path = Path(output)
    path.write_text(html, encoding="utf-8")
    return path.resolve()


def render_pdf(
    document: list[dict[str, object]] | str,
    output: str | Path,
    *,
    style: str = "article",
    custom_css: str | None = None,
    title: str | None = None,
) -> Path:
    """Render *document* to a PDF file using WeasyPrint.

    WeasyPrint is an optional dependency.  This function performs a lazy
    import and raises :exc:`ImportError` with a helpful message if it is not
    installed.

    Parameters
    ----------
    document:
        Atom list or plain Markdown string.
    output:
        Destination ``.pdf`` file path.
    style:
        Built-in style name.
    custom_css:
        Optional extra CSS.
    title:
        Optional document title.

    Returns
    -------
    Path
        Resolved path of the written PDF.

    Raises
    ------
    ImportError
        If WeasyPrint is not installed.
    """
    try:
        import weasyprint  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "WeasyPrint is required for PDF rendering.  "
            "Install it with: pip install weasyprint"
        ) from exc

    html = render(document, style=style, custom_css=custom_css, title=title)
    path = Path(output)
    weasyprint.HTML(string=html).write_pdf(str(path))
    return path.resolve()
