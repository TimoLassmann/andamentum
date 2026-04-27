"""Loud-failure exception hierarchy for harvest.

Per project convention: never silently return empty / wrong markdown.
Every recoverable problem is a typed exception so callers can pattern-match.
"""

from __future__ import annotations


class HarvestError(Exception):
    """Base class for all harvest failures."""


class FetchError(HarvestError):
    """URL fetch failed (HTTP error, SSRF block, timeout, DNS failure)."""


class UnsupportedFormatError(HarvestError):
    """Input format detected, but no backend can handle it."""


class ExtractionError(HarvestError):
    """All applicable backends failed (or returned no usable content) on the bytes.

    Carries diagnostic info: which backends were tried, what they returned
    (length / score), so the caller can log a useful message.
    """

    def __init__(
        self,
        message: str,
        *,
        attempted: list[str] | None = None,
        diagnostics: dict[str, str] | None = None,
    ):
        self.attempted = attempted or []
        self.diagnostics = diagnostics or {}
        detail = ""
        if self.attempted:
            detail = f"  attempted: {', '.join(self.attempted)}"
        if self.diagnostics:
            detail += "\n  diagnostics:\n    " + "\n    ".join(
                f"{k}: {v}" for k, v in self.diagnostics.items()
            )
        super().__init__(f"{message}{detail}" if detail else message)
