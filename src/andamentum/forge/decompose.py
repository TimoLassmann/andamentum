"""Worker: decompose the framed problem into a fully-typed node board.

This is the dialect rewrite of forge's imperative negotiation loop. Two bounded stages,
both engine-free:

  Stage 1 — ``list_jobs`` per area (fan-out): name each area's steps as plain sentences.
            A list of strings is the one shape small models fill reliably.
  Stage 2 — ``type_node`` per job, **sequentially**, against a growing **variable
            registry**: each node is shown the closed list of variables produced by
            upstream nodes (plus the graph input) and SELECTS which it reads from that
            list. It never invents read-names, so producer/consumer wiring cannot drift.

The registry is the fix for the failure mode a real model otherwise hits: one node
*produces* ``summary`` while the next *consumes* ``brief_summary`` (same idea, different
word), leaving the link unconnected. By selecting from the real list, the consumer can
only name a variable that actually exists. A selection outside the list is retried with
the list as feedback, and **fails loud** after the cap — it is never silently dropped or
rewritten to read the raw input (a system that quietly stops chaining is worse than one
that stops the build).

Stage 1 fan-out is bounded (Law 5: ``max_jobs_per_area`` / ``max_nodes``). Stage 2 is
sequential by necessity — node *k* must see what nodes *< k* produced.
"""

from __future__ import annotations

import asyncio

from .agents import LIST_JOBS, TYPE_NODE, AgentSink
from .naming import canonical_datum
from .schemas import INPUT_TOKENS, DesignPlan, ForgeWhy, JobList, NodeDraft, NodeTyping

# How many times a node may re-select its inputs before the design fails loud (Law 5).
MAX_WIRE_RETRIES = 3


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
) -> tuple[DesignPlan, list[str]]:
    """Return ``(plan, notes)`` — the typed node board and any advisory notes."""
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

    # Stage 2: SEQUENTIAL typing against a growing variable registry. Every read is
    # selected from `available` (the closed list of upstream-produced variables + input),
    # so the wiring connects by construction.
    available: list[str] = ["input"]
    available_set: set[str] = {"input"}
    for d in drafts:
        await _wire_node(
            d, drafts, sink=sink, available=available, available_set=available_set
        )
        for p in d.produces:
            if p not in available_set:
                available_set.add(p)
                available.append(p)

    return DesignPlan(why=why, nodes=drafts), notes


async def _wire_node(
    draft: NodeDraft,
    drafts: list[NodeDraft],
    *,
    sink: AgentSink,
    available: list[str],
    available_set: set[str],
) -> None:
    """Type one node, validating that every read it selects is actually available.

    Retries with the available list as feedback; raises (fail loud) if the agent keeps
    selecting variables that do not exist. Never drops a bad read or rewrites it.
    """
    feedback = ""
    problem = ""
    for _ in range(MAX_WIRE_RETRIES):
        typing = await sink.run(
            TYPE_NODE,
            job=draft.job,
            board=_board(drafts, draft.id),
            available=", ".join(available),
            feedback=feedback,
        )
        assert isinstance(typing, NodeTyping)
        consumes = [_canon(c) for c in typing.consumes if c.strip()]
        bad = [c for c in consumes if c not in available_set]
        if bad:
            problem = f"selected reads {bad}, which no earlier step produces"
            feedback = (
                f"You {problem}. Choose `consumes` ONLY from this exact list (copy verbatim): "
                f"{', '.join(available)}."
            )
            continue
        if not consumes:
            # Every step must have an input, or its body has nothing to work from and is
            # unfillable. At minimum it reads `input`. Fail loud rather than emit a dead node.
            problem = "selected no inputs at all"
            feedback = (
                "Every step MUST read at least one variable — it cannot work from nothing. "
                f"Select at least one of (copy verbatim): {', '.join(available)} (use `input` if "
                "this step is the first to touch the raw input)."
            )
            continue
        draft.kind = typing.kind
        draft.consumes = consumes
        draft.produces = [_canon(p) for p in typing.produces if p.strip()][
            :1
        ]  # exactly one new datum
        draft.produces_kind = typing.produces_kind
        draft.control = typing.control
        draft.network = typing.network
        return
    raise ValueError(
        f"node {draft.job!r}: the agent {problem} after {MAX_WIRE_RETRIES} attempts. "
        "The design is incomplete — surfaced loudly, never dropped."
    )


# Re-exported for callers that build a board directly (e.g. tests).
__all__ = ["decompose", "MAX_WIRE_RETRIES"]
