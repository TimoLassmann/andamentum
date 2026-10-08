"""Errors raised by andamentum.figures."""

from __future__ import annotations


class FiguresError(ValueError):
    """Base class for figures errors.

    A ``ValueError`` subclass, so callers that catch ``ValueError`` keep working.
    """
