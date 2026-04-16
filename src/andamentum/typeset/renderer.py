"""HTML/PDF renderer stubs for andamentum.typeset.

The public functions in this module are intentionally unimplemented
(``NotImplementedError``).  They define the interface that will be filled in
by a future rendering back-end.
"""

from __future__ import annotations

from pathlib import Path


def render(
    document: list[dict[str, object]] | str,
    *,
    style: str = "article",
    custom_css: str | None = None,
    title: str | None = None,
) -> str:
    """Render *document* to an HTML string.

    Parameters
    ----------
    document:
        Either a validated list of atom dicts or a pre-serialised JSON string.
    style:
        Name of the built-in style to apply (see :mod:`andamentum.typeset.styles`).
    custom_css:
        Optional CSS string appended after the base style.
    title:
        Optional document title used in the ``<title>`` element.

    Returns
    -------
    str
        HTML string.

    Raises
    ------
    NotImplementedError
        Always — this is a stub pending the rendering back-end.
    """
    raise NotImplementedError("render() is not yet implemented.")


def render_to_file(
    document: list[dict[str, object]] | str,
    output: str | Path,
    **kwargs: object,
) -> Path:
    """Render *document* and write the HTML to *output*.

    Parameters
    ----------
    document:
        Atom list or JSON string.
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
        Atom list or JSON string.
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
    NotImplementedError
        Propagated from :func:`render` (stub not yet implemented).
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
