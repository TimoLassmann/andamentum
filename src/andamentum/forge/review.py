"""Worker: deterministic per-area plan coverage (Tier 1a).

``plan_coverage(why, areas, drafts) -> list[DesignFinding]`` — every framed area must map
to >=1 node job (NodeDraft.area). An area that produced zero jobs is a concrete
UNCOVERED_AREA gap. (Single-sink / input-consumed structural facts are diagnose.py's job —
not duplicated here.) Blocking: decompose() raises on any finding.

Leaf worker (dialect Law 2): pydantic + sibling schemas only; no graph engine.
"""

from __future__ import annotations

from .schemas import (
    DesignFinding,
    FindingKind,
    ForgeWhy,
    NodeDraft,
)


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


__all__ = ["plan_coverage"]
