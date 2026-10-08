"""Errors raised by andamentum.agentic_dialect."""

from __future__ import annotations


class DialectError(Exception):
    """Base class for agentic_dialect errors."""


class PathNotFoundError(DialectError, FileNotFoundError):
    """``check_code`` was given a path that does not exist."""


class UnknownLawError(DialectError, KeyError):
    """No law has this id (ids look like ``L7``)."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class UnknownRoleError(DialectError, KeyError):
    """No role has this name (see ``roles()``)."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""
