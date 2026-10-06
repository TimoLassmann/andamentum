"""The decision cache's precondition: the decision model answers deterministically.

The run store keeps decision answers by (model digest, request, state)
(``ResearchStore.cache_answers``; ``ask.Asker.decide`` reads and writes it). That
is only sound for a model that gives the same answer to the same request, so a
run checks first (``decision_determinism``): one fixed request, sent twice, one
after the other. If the two answers differ, the cache is disabled for that
model and the run says so (stats, a ``check`` item, a note); nothing is cached
on a hope.

The verdict is kept in the store per model digest, so a store that has judged a
digest does not ask again. A check that could not run (the model failed, or has
no digest) disables the cache and records why; it is never read as "fine".
"""

from __future__ import annotations

from pydantic import BaseModel

from andamentum.core.decisions import DecisionModel, Question

from .ask import MODEL_ERRORS
from .store import ResearchStore

CHECK_NAME = "decision_determinism"
# The fixed request: a synthetic statement and one yes/no question about it.
PROBE_STATE: dict[str, object] = {
    "statement": "Copper conducts electricity better than rubber does.",
}
PROBE_QUESTION = Question(
    type="noul", instructions="Does the statement describe a physical property?"
)
_KEY = "q"


class DeterminismCheck(BaseModel):
    """What the check found, and what the run does about the cache."""

    deterministic: bool
    enabled: bool
    detail: str
    probe_calls: int = 0
    reused: bool = False

    @property
    def state(self) -> str:
        """The value of ``RunStats.decision_cache``."""
        return "enabled" if self.enabled else f"disabled: {self.detail}"


async def check_determinism(
    decision_model: DecisionModel, *, store: ResearchStore
) -> DeterminismCheck:
    """Whether *decision_model* may have its answers cached in *store*."""
    digest = decision_model.digest or ""
    if not digest:
        return DeterminismCheck(
            deterministic=False,
            enabled=False,
            detail=f"{decision_model.id} has no digest to key answers by",
        )
    known = store.determinism_of(digest)
    if known is not None:
        deterministic, detail = known
        return DeterminismCheck(
            deterministic=deterministic,
            enabled=deterministic,
            detail=detail,
            reused=True,
        )
    calls = 0
    answers = []
    try:
        for _ in range(2):
            calls += 1
            answers.append(
                await decision_model.decide(PROBE_STATE, {_KEY: PROBE_QUESTION})
            )
    except MODEL_ERRORS as error:
        return DeterminismCheck(
            deterministic=False,
            enabled=False,
            detail=f"the determinism check failed ({type(error).__name__}: {error})",
            probe_calls=calls,
        )
    first, second = answers
    same = first == second
    detail = (
        "two identical requests gave identical answers"
        if same
        else "two identical requests gave different answers"
    )
    store.record_determinism(digest, deterministic=same, detail=detail)
    return DeterminismCheck(
        deterministic=same, enabled=same, detail=detail, probe_calls=calls
    )
