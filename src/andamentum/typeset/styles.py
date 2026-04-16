"""CSS style stubs for andamentum.typeset.

Each constant is a placeholder CSS string that will be replaced by a real
stylesheet in a future release.
"""

from __future__ import annotations

ARTICLE: str = "/* article style placeholder */"
CV: str = "/* cv style placeholder */"
REPORT: str = "/* report style placeholder */"

STYLES: dict[str, str] = {
    "article": ARTICLE,
    "cv": CV,
    "report": REPORT,
}


def get_style(name: str) -> str:
    """Return the CSS string for *name*.

    Parameters
    ----------
    name:
        Style name (one of ``"article"``, ``"cv"``, ``"report"``).

    Returns
    -------
    str
        CSS string.

    Raises
    ------
    KeyError
        If *name* is not a known style.  The error message includes the list
        of available names.
    """
    if name not in STYLES:
        available = ", ".join(sorted(STYLES))
        raise KeyError(
            f"Unknown style {name!r}. Available styles: {available}."
        )
    return STYLES[name]
