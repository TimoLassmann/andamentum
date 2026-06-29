"""Pydantic-free dataclasses for the forge benchmark.

A :class:`Case` is one brief plus the verdict it should earn (``build`` or ``refuse``)
and, when buildable, the control-flow grammar the design must exhibit. A
:class:`RunOutcome` is what one forge run actually produced; a :class:`CaseScore`
aggregates the ``runs`` repetitions of a case into a pass rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Case:
    """One benchmark case: a brief and the verdict + shape it should earn."""

    brief: str
    expected: str  # "build" | "refuse"
    grammar: str  # "sequence" | "branch" | "loop" | "fanout" | "stateful" | "none"
    note: str = ""


@dataclass
class RunOutcome:
    """The result of one forge run over a case."""

    kind: str  # "built" | "refused" | "design_failed"
    features: set[str] = field(default_factory=set)
    error: str = ""


@dataclass
class CaseScore:
    """The aggregate of ``runs`` repetitions of one case."""

    case: Case
    runs: list[RunOutcome]
    passes: int
    total: int
    pass_rate: float
