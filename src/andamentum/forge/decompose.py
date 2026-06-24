"""Worker: decompose the framed problem into a fully-typed node board.

This is the dialect rewrite of forge's imperative negotiation loop. Two bounded
stages, both engine-free:

  Stage 1 — ``list_jobs`` per area (fan-out): name each area's steps as plain
            sentences. A list of strings is the one shape small models fill reliably.
  Stage 2 — ``type_node`` per job (fan-out): fill ONE node's fields, shown the whole
            board as context — rich input, tiny output.

Both fan-outs are bounded (dialect Law 5): ``max_jobs_per_area`` and ``max_nodes``
trace to Deps values; an over-cap area truncates *loudly* via the returned notes. The
agent runner's own semaphore serialises calls for local models, so the ``gather`` here
is safe and order-preserving.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from .agents import LIST_JOBS, TYPE_NODE, AgentSink
from .schemas import DesignPlan, ForgeWhy, JobList, NodeDraft, NodeTyping


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

    # Stage 2: type every job, each seeing the WHOLE board as context (fan-out over nodes).
    typings = await asyncio.gather(
        *(sink.run(TYPE_NODE, board=_board(drafts, d.id)) for d in drafts)
    )
    for d, t in zip(drafts, typings):
        _apply_typing(d, t)

    # Reconciliation: re-type each node against the now-complete board so cross-node data
    # names line up — a consumer typed before its producer can now reuse the EXACT produced
    # name. One bounded extra pass (Law 5); the fix for small-model producer/consumer drift.
    recon = await asyncio.gather(
        *(sink.run(TYPE_NODE, board=_board(drafts, d.id)) for d in drafts)
    )
    for d, t in zip(drafts, recon):
        _apply_typing(d, t)

    return DesignPlan(why=why, nodes=drafts), notes


def _apply_typing(draft: NodeDraft, typing: BaseModel) -> None:
    assert isinstance(typing, NodeTyping)
    draft.kind = typing.kind
    draft.consumes = [c.strip() for c in typing.consumes if c.strip()]
    draft.produces = [p.strip() for p in typing.produces if p.strip()][
        :1
    ]  # exactly one datum
    draft.produces_kind = typing.produces_kind
    draft.control = typing.control
    draft.network = typing.network
