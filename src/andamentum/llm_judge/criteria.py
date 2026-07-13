"""The default, generic, domain-agnostic criterion set.

Kept intentionally short and general — six axes that apply to almost any
task output. Callers with a specific domain (grant review, code review,
manuscript review, ...) pass their own ``criteria=[Criterion(...), ...]``;
this module does not grow domain-specific lists (see the repo-wide rule
against hard-coded domain rules — prefer general mechanisms).
"""

from __future__ import annotations

from .schemas import Criterion

DEFAULT_CRITERIA: list[Criterion] = [
    Criterion(
        name="correctness",
        description="Is the output factually and logically correct?",
    ),
    Criterion(
        name="completeness",
        description="Does the output address everything the task asked for?",
    ),
    Criterion(
        name="instruction_following",
        description="Does the output follow the explicit constraints and format of the task?",
    ),
    Criterion(
        name="sound_reasoning",
        description="Is the reasoning behind the output valid and well-supported?",
    ),
    Criterion(
        name="clarity",
        description="Is the output clear, well-organized, and easy to follow?",
    ),
    Criterion(
        name="groundedness",
        description="Are claims in the output grounded in the given context rather than fabricated?",
    ),
]

__all__ = ["DEFAULT_CRITERIA"]
