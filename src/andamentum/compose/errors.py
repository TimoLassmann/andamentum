"""The errors compose raises on purpose; ``ComposeError`` is the base of all of them."""

from __future__ import annotations

from .bundle.models import BundleError, ComposeError

__all__ = ["BundleError", "ComposeError", "ComposeInputError", "ComposeUsageError"]


class ComposeUsageError(ComposeError, ValueError):
    """A bad argument: an empty task, an unknown format, a round count below 1."""


class ComposeInputError(ComposeError):
    """An input could not be read: the folder is missing or is not a document
    folder, or a source file does not exist."""
