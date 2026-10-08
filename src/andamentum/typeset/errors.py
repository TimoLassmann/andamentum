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
    "Install the optional extra: pip install 'andamentum[pdf]'. "
    "WeasyPrint also needs system libraries (pango and cairo; "
    "on macOS: brew install pango libffi)."
)
