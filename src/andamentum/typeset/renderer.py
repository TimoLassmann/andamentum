"""HTML/PDF renderer for andamentum.typeset.

Public API
----------
render(document, *, style, custom_css, title) -> str
render_to_file(document, output, **kwargs) -> Path
render_pdf(document, output, *, style, custom_css, title) -> Path
"""

from __future__ import annotations

import itertools
import logging
import os
import re
import sys
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

def _id_attr(atom: dict[str, object]) -> str:
    """Return ``id="..."`` (with leading space) when the atom has an ``id`` field."""
    atom_id = atom.get("id")
    return f' id="{atom_id}"' if atom_id is not None else ""


def _render_heading(atom: dict[str, object]) -> str:
    """Render a *heading* atom to an HTML ``<header>`` block."""
    content = _strip_p(_md(str(atom["content"])))
    parts: list[str] = [
        f'<header class="typeset-heading"{_id_attr(atom)}>',
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
        parts.append(f"<h2{_id_attr(atom)}>{heading}</h2>")
        parts.append('<section class="typeset-prose">')
    else:
        parts.append(f'<section class="typeset-prose"{_id_attr(atom)}>')

    parts.append(_md(str(atom["content"])))
    parts.append("</section>")
    return "\n".join(parts)


def _render_callout(atom: dict[str, object]) -> str:
    """Render a *callout* atom to an HTML ``<aside>`` block."""
    tone = atom.get("tone")
    if tone is not None:
        cls = f"typeset-callout tone-{tone}"
    else:
        cls = "typeset-callout"
    content = _md(str(atom["content"]))
    return f'<aside class="{cls}">\n{content}\n</aside>'


def _render_items(atom: dict[str, object]) -> str:
    """Render an *items* atom to a labelled-entries block."""
    parts: list[str] = []

    heading = atom.get("heading")
    if heading is not None:
        parts.append(f"<h2>{heading}</h2>")

    variant = atom.get("variant", "pairs")
    parts.append(f'<div class="typeset-items variant-{variant}">')

    raw_entries = atom.get("entries") or []
    entries = raw_entries if isinstance(raw_entries, list) else []
    for entry in entries:
        assert isinstance(entry, dict)
        label = entry.get("label", "")
        body = _strip_p(_md(str(entry.get("body", ""))))
        item_cls = f"typeset-item item-{variant}"
        parts.append(f'  <div class="{item_cls}">')
        parts.append(f'    <div class="typeset-item-label">{label}</div>')
        parts.append(f'    <div class="typeset-item-body">{body}</div>')
        parts.append("  </div>")

    parts.append("</div>")
    return "\n".join(parts)


def _render_aside(atom: dict[str, object]) -> str:
    """Render an *aside* atom — either a sidebar metadata grid or a simple aside."""
    groups = atom.get("groups")
    if groups is not None:
        assert isinstance(groups, dict)
        parts: list[str] = ['<aside class="typeset-aside typeset-sidebar">']
        for group_name, entries in groups.items():
            parts.append('  <div class="typeset-sidebar-group">')
            parts.append(f'    <div class="typeset-sidebar-title">{group_name}</div>')
            assert isinstance(entries, dict)
            for key, value in entries.items():
                parts.append('    <div class="typeset-sidebar-row">')
                parts.append(f'      <span class="typeset-sidebar-label">{key}</span>')
                parts.append(f'      <span class="typeset-sidebar-value">{value}</span>')
                parts.append("    </div>")
            parts.append("  </div>")
        parts.append("</aside>")
        return "\n".join(parts)

    content = atom.get("content")
    if content is not None:
        return f'<aside class="typeset-aside">{_md(str(content))}</aside>'

    return '<aside class="typeset-aside"></aside>'


def _render_card(atom: dict[str, object]) -> str:
    """Render a *card* atom."""
    parts: list[str] = [f'<div class="typeset-card"{_id_attr(atom)}>']
    parts.append('  <div class="typeset-card-body">')
    parts.append(f'    {_md(str(atom.get("content", "")))}')

    badge = atom.get("badge")
    if badge is not None:
        parts.append(f'    <span class="typeset-badge" data-value="{str(badge).lower()}">{badge}</span>')

    refs = atom.get("refs")
    if refs is not None:
        assert isinstance(refs, list)
        refs_text = ", ".join(str(r) for r in refs)
        parts.append(f'    <sup class="typeset-refs">{refs_text}</sup>')

    parts.append("  </div>")

    source = atom.get("source")
    if source is not None:
        label = atom.get("source_label", source)
        parts.append(f'  <div class="typeset-card-source"><a href="{source}">{label}</a></div>')

    details = atom.get("details")
    if details is not None:
        parts.append('  <details class="typeset-card-details">')
        parts.append("    <summary>Details</summary>")
        parts.append(f"    {_md(str(details))}")
        parts.append("  </details>")

    parts.append("</div>")
    return "\n".join(parts)


def _render_reference(atom: dict[str, object]) -> str:
    """Render a single *reference* atom."""
    parts: list[str] = ['<div class="typeset-reference">']

    number = atom.get("number")
    if number is not None:
        parts.append(f'  <div class="typeset-ref-number">{number}.</div>')

    parts.append('  <div class="typeset-ref-content">')
    parts.append(f'    <div class="typeset-ref-body">{_strip_p(_md(str(atom.get("content", ""))))}</div>')

    badge = atom.get("badge")
    if badge is not None:
        parts.append(f'    <span class="typeset-badge" data-value="{str(badge).lower()}">{badge}</span>')

    source = atom.get("source")
    if source is not None:
        label = atom.get("source_label", source)
        parts.append(f'    <div class="typeset-ref-source"><a href="{source}">{label}</a></div>')

    parts.append("  </div>")

    parts.append("</div>")
    return "\n".join(parts)


def _render_reference_group(group_label: str, refs: list[dict[str, object]]) -> str:
    """Wrap a list of reference atoms in a group container."""
    parts: list[str] = ['<div class="typeset-reference-group">']
    parts.append(f'  <div class="typeset-ref-group-label">{group_label}</div>')
    for ref in refs:
        parts.append(_render_reference(ref))
    parts.append("</div>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_RENDERERS: dict[str, object] = {
    "heading": _render_heading,
    "prose": _render_prose,
    "callout": _render_callout,
    "items": _render_items,
    "aside": _render_aside,
    "card": _render_card,
    "reference": _render_reference,
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
    footer: str = "",
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
    css = css.replace("{footer_label}", footer)

    # Assemble body, clustering consecutive reference atoms by group.
    rendered_parts: list[str] = []
    idx = 0
    while idx < len(validated):
        atom = validated[idx]
        if atom.get("kind") == "reference":
            # Collect all consecutive reference atoms.
            ref_run: list[dict[str, object]] = []
            while idx < len(validated) and validated[idx].get("kind") == "reference":
                ref_run.append(validated[idx])
                idx += 1
            # Sub-group by their `group` field (None if absent).
            for group_label, group_iter in itertools.groupby(
                ref_run, key=lambda a: a.get("group")
            ):
                group_refs = list(group_iter)
                if group_label is not None:
                    rendered_parts.append(
                        _render_reference_group(str(group_label), group_refs)
                    )
                else:
                    for ref in group_refs:
                        rendered_parts.append(_render_reference(ref))
        else:
            rendered_parts.append(_render_atom(atom))
            idx += 1

    body = "\n\n".join(rendered_parts)

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
    footer: str = "",
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
    # Ensure Homebrew libs are findable on Apple Silicon (pango, gobject).
    if sys.platform == "darwin":
        brew_lib = "/opt/homebrew/lib"
        if os.path.isdir(brew_lib):
            cur = os.environ.get("DYLD_LIBRARY_PATH", "")
            if brew_lib not in cur.split(os.pathsep):
                os.environ["DYLD_LIBRARY_PATH"] = (
                    f"{brew_lib}{os.pathsep}{cur}" if cur else brew_lib
                )

    try:
        import weasyprint  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "WeasyPrint is required for PDF rendering.  "
            "Install it with: pip install weasyprint  "
            "On macOS: brew install pango libffi"
        ) from exc

    html = render(
        document, style=style, custom_css=custom_css, title=title, footer=footer
    )
    path = Path(output)
    weasyprint.HTML(string=html).write_pdf(str(path))
    return path.resolve()
