"""Investigator registry — extension point for hypothesis-resolution methods.

The Skim agent emits hypotheses tagged (via the deterministic
``classify_hypothesis``) with an ``investigation_type``. The InvestigateLoop
node looks up the investigator function for that type in the registry
and dispatches the hypothesis to it.

This module's existence is the architectural seam that supports future
extensions like:
  • novelty (run deep_research to verify novelty claims)
  • factual (look up specific facts in an external KB)
  • statistical (re-run a calculation from the document's data)

Each new investigator type is added by:
  1. Writing a pattern in ``dispatcher.py``
  2. Implementing the investigator function (matching the
     ``Investigator`` Protocol in this file)
  3. Registering it in ``INVESTIGATORS``

For Phase 2 only ``internal`` ships — investigators that read sections
of the document itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable

from .dispatcher import classify_hypothesis
from .internal import investigate_internal

if TYPE_CHECKING:
    from ..deps import ReviewDeps
    from ..schemas import Finding, Hypothesis
    from ..state import ReviewState


# An investigator is an async function that takes a hypothesis + the live
# graph state/deps and returns either a Finding (committed), None (the
# hypothesis was unfounded), or a list of sub-hypotheses (the hypothesis
# needs to be split before it can be answered).
Investigator = Callable[
    ["Hypothesis", "ReviewState", "ReviewDeps"],
    Awaitable["InvestigationOutcome"],
]


class InvestigationOutcome:
    """Discriminated outcome returned by every investigator.

    Implemented as a frozen dataclass-like with constructor helpers so
    the InvestigateLoop can pattern-match on the type.
    """

    __slots__ = ("kind", "finding", "unfounded_reason", "sub_hypotheses", "raw_quotes")

    def __init__(
        self,
        kind: str,
        finding: "Finding | None" = None,
        unfounded_reason: str = "",
        sub_hypotheses: "list[Hypothesis] | None" = None,
        raw_quotes: list[str] | None = None,
    ):
        self.kind = kind
        self.finding = finding
        self.unfounded_reason = unfounded_reason
        self.sub_hypotheses = sub_hypotheses or []
        self.raw_quotes = raw_quotes or []

    @classmethod
    def found(
        cls, finding: "Finding", *, raw_quotes: list[str] | None = None
    ) -> "InvestigationOutcome":
        return cls(kind="finding", finding=finding, raw_quotes=raw_quotes)

    @classmethod
    def unfounded(cls, reason: str) -> "InvestigationOutcome":
        return cls(kind="unfounded", unfounded_reason=reason)

    @classmethod
    def split(cls, sub_hypotheses: "list[Hypothesis]") -> "InvestigationOutcome":
        return cls(kind="needs_subhypotheses", sub_hypotheses=sub_hypotheses)


INVESTIGATORS: dict[str, Investigator] = {
    "internal": investigate_internal,
}


def register_investigator(investigation_type: str, investigator: Investigator) -> None:
    """Add an investigator type at runtime.

    Useful for tests (inject a fake) and for downstream packages that
    want to extend whetstone without forking it (e.g., a `novelty`
    investigator that calls deep_research).
    """
    INVESTIGATORS[investigation_type] = investigator


def get_investigator(investigation_type: str) -> Investigator:
    """Look up an investigator by type. Falls back to `internal` on miss.

    The fallback is intentional: an unknown investigation_type is always
    safer to handle by reading the document than to silently skip.
    """
    return INVESTIGATORS.get(investigation_type, INVESTIGATORS["internal"])


__all__ = [
    "INVESTIGATORS",
    "Investigator",
    "InvestigationOutcome",
    "classify_hypothesis",
    "get_investigator",
    "investigate_internal",
    "register_investigator",
]
