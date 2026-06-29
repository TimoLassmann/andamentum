"""Worker: the plan manager — goal-vs-plan grounding (Tier 1), deterministic + semantic.

Two pure/engine-free entry points:

- ``plan_coverage(why, areas, drafts) -> list[DesignFinding]`` — DETERMINISTIC. Every
  framed area must map to >=1 node job (NodeDraft.area). An area that produced zero jobs
  is a concrete UNCOVERED_AREA gap. (Single-sink / input-consumed structural facts are
  diagnose.py's job — not duplicated here.) Blocking: decompose() raises on any finding.

- ``review_plan(why, board_text, *, sink) -> PlanVerdict`` — the one small LLM call
  (PLAN_MANAGER) plus a deterministic rapidfuzz dedup of its uncovered_concerns against
  the existing node jobs, so only concrete, non-redundant gaps survive.

Leaf worker (dialect Law 2): pydantic + rapidfuzz + sibling schemas only; no graph engine.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from .agents import PLAN_MANAGER, AgentSink
from .schemas import (
    DesignFinding,
    FindingKind,
    ForgeWhy,
    NodeDraft,
    PlanVerdict,
)

# A concern this similar to an existing node job is already covered — drop it (dedup).
_CONCERN_COVERED_THRESHOLD = 80.0


def plan_coverage(
    why: ForgeWhy, areas: list[str], drafts: list[NodeDraft]
) -> list[DesignFinding]:
    """Every framed area must own at least one node job. An area with zero jobs is a
    deterministic UNCOVERED_AREA gap (blocking). Pure; no near-miss matching — area
    membership is the exact NodeDraft.area string set during decompose stage 1.

    ``why`` is reserved for a later purpose-level coverage check; asserting it is present
    documents that intent and keeps it in the signature per the resolved design."""
    assert why is not None
    covered = {d.area for d in drafts}
    findings: list[DesignFinding] = []
    for area in areas:
        if area not in covered:
            findings.append(
                DesignFinding(
                    kind=FindingKind.UNCOVERED_AREA,
                    detail=(
                        f"framed concern {area!r} produced no steps — the plan does not "
                        "address it"
                    ),
                    suggestion=(
                        "decompose this concern into at least one step, or drop it from the "
                        "framing if it is not a real concern"
                    ),
                )
            )
    return findings


def _concern_is_covered(concern: str, jobs: list[str]) -> bool:
    """True if a node job already covers this concern (rapidfuzz, diagnose's tool)."""
    return any(
        fuzz.partial_ratio(concern, job) >= _CONCERN_COVERED_THRESHOLD for job in jobs
    )


async def review_plan(
    why: ForgeWhy, board_text: str, jobs: list[str], *, sink: AgentSink
) -> PlanVerdict:
    """One PLAN_MANAGER call over the compact board, then dedup its concerns against the
    existing node jobs so only concrete, non-redundant gaps survive."""
    out = await sink.run(
        PLAN_MANAGER,
        purpose=why.purpose,
        boundary_in=why.boundary_in,
        boundary_out=why.boundary_out,
        board=board_text,
    )
    assert isinstance(out, PlanVerdict)
    surviving = [
        c.strip()
        for c in out.uncovered_concerns
        if c.strip() and not _concern_is_covered(c.strip(), jobs)
    ]
    return PlanVerdict(serves_goal=out.serves_goal, uncovered_concerns=surviving)


def plan_board(drafts: list[NodeDraft]) -> str:
    """A compact name -> job board for the plan manager — NO consumes/produces, NO code."""
    return "\n".join(f"- {d.id}: {d.job}" for d in drafts)


__all__ = ["plan_coverage", "review_plan", "plan_board"]
