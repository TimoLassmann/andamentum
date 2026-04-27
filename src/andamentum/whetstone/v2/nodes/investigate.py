"""Node: InvestigateLoop.

Pops the highest-priority open hypothesis off the queue, dispatches to
the appropriate investigator (via the registry), and updates state with
the outcome. Loops until the queue is empty OR the budget is exhausted.

Each iteration is a single LLM call (the investigator's). Sub-hypotheses
emitted by an investigator are pushed back onto the queue with reduced
priority — this lets the agent break a hard question into easier ones
without unbounded recursion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_graph import BaseNode, GraphRunContext

from ..deps import ReviewDeps
from ..investigators import get_investigator
from ..schemas import Hypothesis, ReviewResult
from ..state import FailedInvestigation, ReviewState

if TYPE_CHECKING:
    from .edit_sections import EditSections


# Priority sort order. Higher numerical = investigated first.
_PRIORITY_ORDER = {"high": 3, "medium": 2, "low": 1}


@dataclass
class InvestigateLoop(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Run investigators until queue empty or budget exhausted."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "EditSections":
        ctx.state.current_phase = "investigate"

        while (
            _has_open_hypothesis(ctx.state.hypotheses)
            and ctx.state.investigations_done < ctx.state.hypothesis_budget
        ):
            hypothesis = _pop_next(ctx.state.hypotheses)
            if hypothesis is None:
                break

            hypothesis.status = "investigating"
            investigator = get_investigator(hypothesis.investigation_type)
            try:
                outcome = await investigator(hypothesis, ctx.state, ctx.deps)
            except Exception as exc:
                ctx.state.failed_investigations.append(
                    FailedInvestigation(hypothesis=hypothesis, error=str(exc))
                )
                hypothesis.status = "unfounded"
                ctx.state.investigations_done += 1
                continue

            ctx.state.investigations_done += 1

            if outcome.kind == "finding" and outcome.finding is not None:
                # Propagate perspective if this hypothesis was tagged.
                if hypothesis.perspective:
                    outcome.finding.perspective = hypothesis.perspective
                ctx.state.findings.append(outcome.finding)
                hypothesis.status = "resolved"

            elif outcome.kind == "needs_subhypotheses":
                # Push sub-hypotheses with the parent's perspective + a
                # priority cap so we don't recurse forever.
                from typing import cast, Literal as _Lit

                for sub in outcome.sub_hypotheses:
                    sub.priority = cast(
                        "_Lit['low', 'medium', 'high']",
                        _demote(hypothesis.priority),
                    )
                    sub.perspective = hypothesis.perspective
                    sub.status = "open"
                    ctx.state.hypotheses.append(sub)
                hypothesis.status = "resolved"  # the parent is "decomposed"

            else:  # "unfounded"
                hypothesis.status = "unfounded"

        from .edit_sections import EditSections

        return EditSections()


def _has_open_hypothesis(hypotheses: list[Hypothesis]) -> bool:
    return any(h.status == "open" for h in hypotheses)


def _pop_next(hypotheses: list[Hypothesis]) -> Hypothesis | None:
    """Return the highest-priority open hypothesis (does NOT remove from list)."""
    candidates = [h for h in hypotheses if h.status == "open"]
    if not candidates:
        return None
    candidates.sort(key=lambda h: -_PRIORITY_ORDER.get(h.priority, 0))
    return candidates[0]


def _demote(priority: str) -> "str":
    """Lower a priority by one tier — sub-hypotheses are less urgent than parent."""
    if priority == "high":
        return "medium"
    if priority == "medium":
        return "low"
    return "low"
