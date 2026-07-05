"""Worker: decompose the framed problem into a coherent, fully-typed node board.

Bounded, engine-free stages. Stage 1 lists jobs; Stage 2 elicits each node's I/O in
TWO passes — declare-then-select — so producer→consumer wiring is correct BY
CONSTRUCTION rather than by the model reproducing matching name strings across calls:

  Stage 1 — ``list_jobs`` per area (fan-out): name each area's steps as plain sentences.
            A list of strings is the one shape small models fill reliably.
  Stage 2a — ``declare_node`` per job (DECLARE): the model declares ONLY each node's
            kind and its ONE produced name (plus produces_kind / control / network). It
            declares NO inputs. Deterministic code then canonicalises and DEDUPES the
            produced names so the produced set P is GLOBALLY UNIQUE by construction (a
            canonical collision gets a numeric suffix, e.g. ``_2``).
  Stage 2b — ``select_consumes`` per job (SELECT): the model is shown the CLOSED,
            NUMBERED list of readable data (``input`` followed by every name in P, each
            annotated with its producer's job) and returns ``consume_indices`` — ORDINALS
            into that list, never name strings. Deterministic code maps indices → names.
            A consume can therefore never reference a name no step produces; an
            out-of-range index is dropped and recorded (never silently kept as a phantom).
  Stage 3 — the refine loop (bounded by ``MAX_DESIGN_ROUNDS``): assemble the board into a
            data DAG (:mod:`assemble`), diagnose every structural problem (:mod:`diagnose`),
            and — for each flagged node — RE-RUN ONLY pass 2b (``select_consumes``) with the
            finding as feedback. Producer names are FROZEN after 2a, so a repair can never
            reinvent a produce: the name-matching thrash is gone. Converge to a clean
            board, or **fail loud** with the full report after the cap.

A detected problem is never silently dropped, defaulted, or rewritten: it is surfaced in
the :class:`DesignReport` and repaired, or — at the cap — raised. A system that runs but
does the wrong thing is worse than one that stops.

Stage 1 fan-out is bounded (Law 5: ``max_jobs_per_area`` / ``max_nodes``); the refine loop
is bounded by ``MAX_DESIGN_ROUNDS``.
"""

from __future__ import annotations

import asyncio

from .agents import DECLARE_NODE, LIST_JOBS, SELECT_CONSUMES, AgentSink
from .assemble import assemble
from .diagnose import diagnose
from .naming import canonical_datum
from .review import plan_coverage
from .schemas import (
    INPUT_TOKENS,
    ConsumeSelection,
    DataKind,
    DesignFinding,
    DesignPlan,
    DesignReport,
    ForgeWhy,
    JobList,
    NodeDeclaration,
    NodeDraft,
)
from .spec import NodeControl

# How many assemble→diagnose→repair rounds before the design fails loud (Law 5).
MAX_DESIGN_ROUNDS = 4

# The single graph-input datum presented at ordinal 0 in every select list.
INPUT_OPTION = "input"


def _job_board(drafts: list[NodeDraft], focus_id: str) -> str:
    """A plain-text view of every step's job, the focus node marked ``>>>``."""
    lines: list[str] = []
    for d in drafts:
        mark = ">>>" if d.id == focus_id else "   "
        area = f"[{d.area}] " if d.area else ""
        lines.append(f"{mark} {d.id} {area}{d.job}")
    return "\n".join(lines)


def dedupe_names(
    raw_names: list[str], fallbacks: list[str], input_tokens: frozenset[str]
) -> list[str]:
    """Canonicalise ``raw_names`` and make them globally unique (order-preserving).

    A blank raw name falls back to the paired ``fallbacks`` entry. A canonical collision
    (with an earlier produced name OR an input token, which a produce must never shadow)
    gets a numeric suffix (``_2``, ``_3``, …). Pure and deterministic — this is what makes
    the produced-name set unique by construction, so ``duplicate_producer`` cannot arise.
    """
    used: set[str] = set(input_tokens)
    out: list[str] = []
    for raw, fallback in zip(raw_names, fallbacks):
        base = canonical_datum(raw, input_tokens) if raw.strip() else ""
        if not base:
            base = canonical_datum(fallback, input_tokens)
        unique = base
        suffix = 2
        while unique in used:
            unique = f"{base}_{suffix}"
            suffix += 1
        used.add(unique)
        out.append(unique)
    return out


def build_option_names(produced: list[str]) -> list[str]:
    """The closed, ordinal-indexed list of names for the input plus ``produced``: the graph
    input (ordinal 0) followed by every produced name. Pure ordinal→name mapping."""
    return [INPUT_OPTION, *produced]


def visible_producers(drafts: list[NodeDraft], focus_index: int) -> list[NodeDraft]:
    """The producer nodes a node at ``focus_index`` may read, in ordinal order.

    A node sees only inputs it can legitimately depend on — so an accidental cycle is
    impossible **by construction** while every real grammar stays expressible:

    - **every EARLIER node** (``j < focus_index``) — ordinary forward data flow (chain,
      fan-in, fan-out, branch);
    - **a later CHECKPOINT node** — the loop back-edge: a bounded-loop checkpoint is the one
      legitimate cycle (``diagnose`` exempts it), so a body step may read the checkpoint's
      decision;
    - **the focus node's OWN output iff it is an ENTITY** — the rung-2 read-modify-write
      self-edge (``diagnose`` exempts the entity round-trip).

    A later *signal* node and a non-entity self-read are excluded, which is exactly what
    made the unconstrained closed set explode into cycles: the model could pick any node,
    including downstream and itself.
    """
    visible: list[NodeDraft] = []
    for j, d in enumerate(drafts):
        if not d.produces:
            continue
        if j < focus_index:
            visible.append(d)
        elif j == focus_index:
            if d.produces_kind is DataKind.ENTITY:
                visible.append(d)
        elif d.control is NodeControl.CHECKPOINT:
            visible.append(d)
    return visible


def _option_names_for(drafts: list[NodeDraft], focus_index: int) -> list[str]:
    """The ordinal→name list a node at ``focus_index`` selects over (input + its visible
    producers) — the frozen, cycle-safe closed set for that node."""
    return build_option_names(
        [d.produces[0] for d in visible_producers(drafts, focus_index)]
    )


def _options_text_for(drafts: list[NodeDraft], focus_index: int) -> str:
    """The numbered list shown to ``select_consumes`` for the node at ``focus_index`` — the
    input plus each VISIBLE producer, annotated with its producing step's job."""
    lines = [f"0. {INPUT_OPTION} — the raw original text given to the whole system"]
    for i, d in enumerate(visible_producers(drafts, focus_index), start=1):
        lines.append(f"{i}. {d.produces[0]} — produced by step {d.id}: {d.job}")
    return "\n".join(lines)


def resolve_consumes(
    indices: list[int], option_names: list[str]
) -> tuple[list[str], list[int]]:
    """Map selected ordinals → data names over the closed ``option_names`` list.

    Returns ``(names, dropped)``: ``names`` are the in-range selections resolved to real
    data names (de-duplicated, order-preserving); ``dropped`` are the out-of-range ordinals
    (recorded by the caller, never silently ignored). By construction every resolved name is
    a real produced name or the input token — so ``dangling_read`` / ``near_miss`` cannot
    arise from a resolved consume.
    """
    names: list[str] = []
    dropped: list[int] = []
    seen: set[str] = set()
    for idx in indices:
        if 0 <= idx < len(option_names):
            name = option_names[idx]
            if name not in seen:
                seen.add(name)
                names.append(name)
        else:
            dropped.append(idx)
    return names, dropped


def collapse_extra_sinks(drafts: list[NodeDraft]) -> list[str]:
    """Deterministically resolve ``multiple_sinks`` from over-decomposition, no model call.

    When several steps each produce an unconsumed final signal, the system output is the
    LAST step's (matching ``diagnose._sinks`` / the compile backstop's topologically-last
    rule). Make that last step CONSUME every extra terminal signal, so exactly one sink
    remains and the earlier producers stop being dead-ends. Forward-window safe: every
    merged producer is earlier than the last step, so no cycle is introduced. Returns the
    merged names (empty when there is nothing to merge, or when the last step is not itself
    the sole terminal — that shape is left to the model)."""
    if len(drafts) < 2:
        return []
    consumed = set(assemble(drafts).readers)
    last = drafts[-1]
    last_is_terminal = any(
        n not in consumed
        for n in last.produces
        if last.produces_kind is DataKind.SIGNAL
    )
    if not last_is_terminal:
        return []
    merged = [
        n
        for d in drafts[:-1]
        if d.produces_kind is DataKind.SIGNAL
        for n in d.produces
        if n not in consumed and n not in last.consumes
    ]
    last.consumes.extend(merged)
    return merged


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

    # Stage 2a: DECLARE — each node's kind + its ONE produced name (no inputs yet).
    await asyncio.gather(*(_declare_node(d, drafts, sink=sink) for d in drafts))
    # Deterministic dedupe → the produced set is globally unique BY CONSTRUCTION.
    unique = dedupe_names(
        [d.produces[0] if d.produces else "" for d in drafts],
        [f"{d.id}_out" for d in drafts],
        INPUT_TOKENS,
    )
    for d, name in zip(drafts, unique):
        d.produces = [name]

    # Stage 2b: SELECT — each node picks its inputs BY ORDINAL from the frozen closed set.
    for d in drafts:
        await _select_consumes(d, drafts, sink=sink, feedback="", notes=notes)

    # Stage 3: assemble → diagnose → repair, bounded. A repair re-runs ONLY pass 2b for a
    # flagged node (producer names are frozen), so wiring converges instead of thrashing.
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
        # Deterministic sink-collapse first (over-decomposition merge, no model call); then
        # let the model re-select only for what determinism could not fix.
        merged = collapse_extra_sinks(drafts)
        if merged:
            notes.append(
                f"decompose: merged {len(merged)} extra terminal signal(s) into the system "
                f"output ({', '.join(merged)})"
            )
            report = diagnose(drafts, assemble(drafts))
            if report.clean:
                break
        await _repair_round(drafts, report, sink=sink, notes=notes)

    # Deterministic plan-coverage (Tier 1a): every framed concern must own >=1 step.
    # Blocking and fail-loud, consistent with the structural cap path above.
    coverage = plan_coverage(why, areas, drafts)
    if coverage:
        report = DesignReport(findings=report.findings + coverage)
        raise ValueError(
            "decompose: framed concerns produced no steps (uncovered areas). The plan does "
            f"not address part of the brief (surfaced, never dropped):\n{report.summary()}"
        )

    return DesignPlan(why=why, nodes=drafts), report, notes


async def _repair_round(
    drafts: list[NodeDraft],
    report: DesignReport,
    *,
    sink: AgentSink,
    notes: list[str],
) -> None:
    """Re-select the inputs of each node a finding names, feeding the finding back.

    Findings without a node (e.g. ``multiple_sinks`` / ``no_output``) are routed to the
    terminal node — the last board node — since that is where the system output lives.
    Only pass 2b re-runs: produced names are frozen, so a repair cannot manufacture a new
    ``duplicate_producer`` or orphan cascade — it can only re-wire what a node reads.
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
        await _select_consumes(d, drafts, sink=sink, feedback=feedback, notes=notes)


def _feedback_for(findings: list[DesignFinding]) -> str:
    """The concrete repair feedback for one node — the findings' detail + suggestion."""
    lines = [
        "The inputs you chose for this step caused a structural problem. Fix it by RE-CHOOSING "
        "the input numbers (pick the number of the earlier step whose output this step should "
        "read):"
    ]
    for f in findings:
        line = f"- {f.detail}"
        if f.suggestion:
            line += f"  Suggested fix: {f.suggestion}"
        lines.append(line)
    return "\n".join(lines)


async def _declare_node(
    draft: NodeDraft,
    drafts: list[NodeDraft],
    *,
    sink: AgentSink,
) -> None:
    """DECLARE pass: type ONE node's kind + its single produced name (no inputs).

    The raw produced name is stored as-is; the caller canonicalises + dedupes it afterward.
    """
    decl = await sink.run(
        DECLARE_NODE,
        job=draft.job,
        board=_job_board(drafts, draft.id),
    )
    assert isinstance(decl, NodeDeclaration)
    draft.kind = decl.kind
    draft.produces = [decl.produces]
    draft.produces_kind = decl.produces_kind
    draft.control = decl.control
    draft.network = decl.network


async def _select_consumes(
    draft: NodeDraft,
    drafts: list[NodeDraft],
    *,
    sink: AgentSink,
    feedback: str,
    notes: list[str],
) -> None:
    """SELECT pass: choose ONE node's inputs by ordinal from the frozen closed set.

    ``feedback`` carries the diagnoser's finding + suggestion during a repair round; it is
    empty on the first pass. Out-of-range ordinals are dropped and recorded in ``notes``.
    """
    focus_index = drafts.index(draft)
    option_names = _option_names_for(drafts, focus_index)
    selection = await sink.run(
        SELECT_CONSUMES,
        job=draft.job,
        board=_job_board(drafts, draft.id),
        options=_options_text_for(drafts, focus_index),
        feedback=feedback,
    )
    assert isinstance(selection, ConsumeSelection)
    names, dropped = resolve_consumes(selection.consume_indices, option_names)
    draft.consumes = names
    if dropped:
        notes.append(
            f"decompose: node {draft.id} selected out-of-range input ordinal(s) "
            f"{dropped} (valid 0..{len(option_names) - 1}); dropped"
        )


# Re-exported for callers that build a board directly (e.g. tests) and the offline harness.
__all__ = [
    "decompose",
    "dedupe_names",
    "build_option_names",
    "resolve_consumes",
    "collapse_extra_sinks",
    "MAX_DESIGN_ROUNDS",
]
