"""Worker: decompose the framed problem into a coherent, fully-typed node board.

Three bounded stages, all engine-free:

  Stage 1 — ``list_jobs`` per area (fan-out): name each area's steps as plain sentences.
            A list of strings is the one shape small models fill reliably.
  Stage 2 — ``type_node`` per job: the model DECLARES each node's kind and its
            ``consumes``/``produces`` variable names FREELY (it is shown the whole board
            so it reuses names, but selection is not forced). Names are canonicalised via
            :func:`naming.canonical_datum` so casing/spacing variants of one concept unify.
  Stage 3 — the refine loop (bounded by ``MAX_DESIGN_ROUNDS``): assemble the declared
            board into a data DAG (:mod:`assemble`), diagnose every structural problem
            (:mod:`diagnose`), and — for each flagged node — re-run ``type_node`` with the
            finding's detail + suggestion as feedback. Converge to a clean board, or **fail
            loud** with the full report after the cap.

The robustness comes from the determinism doing the heavy lifting: :mod:`diagnose` finds
every dangling read / orphan / near-miss / cycle and proposes a concrete fix ("reads
``bullets``; nearest produced name is ``bullet_statements`` — did you mean that?"), so the
model only applies a targeted correction. That is what makes the loop converge on small
local models, and why the closed-registry forced selection is no longer needed.

A detected problem is never silently dropped, defaulted, or rewritten: it is surfaced in
the :class:`DesignReport` and repaired, or — at the cap — raised. A system that runs but
does the wrong thing is worse than one that stops.

Stage 1 fan-out is bounded (Law 5: ``max_jobs_per_area`` / ``max_nodes``); the refine loop
is bounded by ``MAX_DESIGN_ROUNDS``.
"""

from __future__ import annotations

import asyncio

from .agents import LIST_JOBS, TYPE_NODE, AgentSink
from .assemble import assemble
from .diagnose import diagnose
from .naming import canonical_datum
from .schemas import (
    INPUT_TOKENS,
    DesignFinding,
    DesignPlan,
    DesignReport,
    ForgeWhy,
    JobList,
    NodeDraft,
    NodeTyping,
)

# How many assemble→diagnose→repair rounds before the design fails loud (Law 5).
MAX_DESIGN_ROUNDS = 4


def _board(drafts: list[NodeDraft], focus_id: str) -> str:
    """A plain-text view of the whole plan, the focus node marked ``>>>``."""
    lines: list[str] = []
    for d in drafts:
        mark = ">>>" if d.id == focus_id else "   "
        cons = ", ".join(d.consumes) or "—"
        prod = ", ".join(d.produces) or "?"
        lines.append(
            f"{mark} {d.id} [{d.kind.value}] {d.job}  (consumes: {cons}; produces: {prod})"
        )
    return "\n".join(lines)


def _canon(name: str) -> str:
    return canonical_datum(name, INPUT_TOKENS)


async def decompose(
    why: ForgeWhy,
    areas: list[str],
    *,
    sink: AgentSink,
    max_jobs_per_area: int,
    max_nodes: int,
) -> tuple[DesignPlan, DesignReport, list[str]]:
    """Return ``(plan, report, notes)`` — the typed node board, the (clean) structural
    diagnosis, and any advisory notes. Raises if the board cannot be made coherent within
    ``MAX_DESIGN_ROUNDS`` (fail loud, never a half-wired design)."""
    notes: list[str] = []

    # Stage 1: list jobs per area (bounded fan-out over areas).
    job_lists = await asyncio.gather(
        *(sink.run(LIST_JOBS, area=a, purpose=why.purpose) for a in areas)
    )

    drafts: list[NodeDraft] = []
    for area, jl in zip(areas, job_lists):
        assert isinstance(jl, JobList)
        jobs = [j.strip() for j in jl.jobs if j.strip()]
        if len(jobs) > max_jobs_per_area:
            notes.append(
                f"decompose: area {area!r} listed {len(jobs)} jobs; capped to {max_jobs_per_area}"
            )
            jobs = jobs[:max_jobs_per_area]
        for job in jobs:
            if len(drafts) >= max_nodes:
                notes.append(
                    f"decompose: hit max_nodes={max_nodes}; later jobs dropped"
                )
                break
            drafts.append(NodeDraft(id=f"n{len(drafts) + 1}", area=area, job=job))
        if len(drafts) >= max_nodes:
            break

    if not drafts:
        raise ValueError(
            "decompose produced no steps; the brief did not yield an actionable design"
        )

    # Stage 2: type every node, declaring consumes/produces FREELY (no forced selection).
    # The board is shown so names get reused; canonicalisation unifies casing/spacing.
    for d in drafts:
        await _type_node(d, drafts, sink=sink, feedback="")

    # Stage 3: assemble → diagnose → repair, bounded. Converge to clean, or fail loud with
    # the full report. Each round re-types only the flagged nodes, with the finding's
    # detail + suggestion as concrete feedback.
    report = DesignReport()
    for round_index in range(MAX_DESIGN_ROUNDS):
        graph = assemble(drafts)
        report = diagnose(drafts, graph)
        if report.clean:
            break
        if round_index == MAX_DESIGN_ROUNDS - 1:
            raise ValueError(
                "decompose could not produce a coherent design after "
                f"{MAX_DESIGN_ROUNDS} repair rounds. The structural problems remain (surfaced, "
                f"never dropped):\n{report.summary()}"
            )
        await _repair_round(drafts, report, sink=sink)

    return DesignPlan(why=why, nodes=drafts), report, notes


async def _repair_round(
    drafts: list[NodeDraft], report: DesignReport, *, sink: AgentSink
) -> None:
    """Re-type each node a finding names, feeding the finding's detail + suggestion back.

    Findings without a node (e.g. ``multiple_sinks`` / ``no_output``) are routed to the
    terminal node — the last board node — since that is where the system output lives.
    Nothing is dropped: every finding produces a targeted re-type with concrete feedback.
    """
    by_node: dict[str, list[DesignFinding]] = {}
    for finding in report.findings:
        target = finding.node or drafts[-1].id
        by_node.setdefault(target, []).append(finding)

    for d in drafts:
        flags = by_node.get(d.id)
        if not flags:
            continue
        feedback = _feedback_for(flags)
        await _type_node(d, drafts, sink=sink, feedback=feedback)


def _feedback_for(findings: list[DesignFinding]) -> str:
    """The concrete repair feedback for one node — the findings' detail + suggestion."""
    lines = [
        "Your declared inputs/outputs for this step caused a structural problem. Fix it by "
        "re-declaring `consumes`/`produces` (reuse the EXACT name an earlier step produced "
        "when you mean its output):"
    ]
    for f in findings:
        line = f"- {f.detail}"
        if f.suggestion:
            line += f"  Suggested fix: {f.suggestion}"
        lines.append(line)
    return "\n".join(lines)


async def _type_node(
    draft: NodeDraft,
    drafts: list[NodeDraft],
    *,
    sink: AgentSink,
    feedback: str,
) -> None:
    """Type ONE node from its job + the whole board, declaring reads/writes freely.

    Canonicalises the declared names so casing/spacing variants of one concept unify.
    ``feedback`` carries the diagnoser's finding + suggestion during a repair round; it is
    empty on the first pass.
    """
    typing = await sink.run(
        TYPE_NODE,
        job=draft.job,
        board=_board(drafts, draft.id),
        feedback=feedback,
    )
    assert isinstance(typing, NodeTyping)
    draft.kind = typing.kind
    draft.consumes = [_canon(c) for c in typing.consumes if c.strip()]
    draft.produces = [_canon(p) for p in typing.produces if p.strip()][:1]
    draft.produces_kind = typing.produces_kind
    draft.control = typing.control
    draft.network = typing.network


# Re-exported for callers that build a board directly (e.g. tests).
__all__ = ["decompose", "MAX_DESIGN_ROUNDS"]
