"""Errors of the decision-model layer.

One base class, so a caller that can fall back catches
``DecisionError`` and nothing wider.
"""

from __future__ import annotations

from ..errors import CoreError


class DecisionError(CoreError):
    """Any failure to get a decision from a local decision model."""


class DecisionModelUnavailable(DecisionError):
    """The server cannot be reached, is too old, or lacks the model."""


class DecisionRequestTooLarge(DecisionError, ValueError):
    """The request would exceed a documented limit; nothing was sent."""


class DecisionRequestRejected(DecisionError):
    """The server answered a request with a client error (for example HTTP 400
    for an over-length input, which the endpoint refuses rather than truncates)."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(
            f"the decision endpoint rejected the request ({status}): {body}"
        )
        self.status = status


class DecisionResponseError(DecisionError, ValueError):
    """The response does not match the documented shape."""
