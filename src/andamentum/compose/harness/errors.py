"""The base error of the Word tools (``andamentum.word.WordError``).

Every error the Word-file tools raise on purpose derives from :class:`WordError`,
so a caller can catch one class. ``WordUsageError`` is a bad invocation (exit
code 2 on the command line); ``WordInputError`` is a file that cannot be read
(exit code 3).
"""

from __future__ import annotations


class WordError(Exception):
    """Base class of the errors raised by the Word tools."""


class WordUsageError(WordError):
    """A bad invocation: a wrong or missing argument, an impossible combination."""


class WordInputError(WordError):
    """An input file is missing, is not a .docx, or cannot be read."""
