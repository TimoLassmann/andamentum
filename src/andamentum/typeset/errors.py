"""Errors raised by andamentum.typeset."""

from __future__ import annotations


class TypesetError(ValueError):
    """Base class for typeset errors.

    A ``ValueError`` subclass, so callers that catch ``ValueError`` keep working.
    """


class PdfUnavailableError(TypesetError, ImportError):
    """PDF output was requested but WeasyPrint (the ``pdf`` extra) is not installed.

    Also an ``ImportError``, so callers that catch ``ImportError`` keep working.
    """


#: The one install hint, used in every message about PDF output.
PDF_INSTALL_HINT = (
    "Install andamentum with the pdf extra: uv tool install 'andamentum[pdf] @ git+https://github.com/timolassmann/andamentum.git' "
    "(or uv sync --extra pdf in a checkout). "
    "WeasyPrint also needs system libraries (pango and cairo; "
    "on macOS: brew install pango libffi)."
)
