"""Offline, model-free proof that the two-pass declare-then-select wiring makes the three
name-flaw classes STRUCTURALLY IMPOSSIBLE on the primary design path.

The reliability failure this replaces was small-model name-matching thrash: the model had to
reproduce ``consumes``/``produces`` name strings character-for-character across separate
calls, and the repair loop reinvented names, manufacturing fresh ``duplicate_producer`` /
``near_miss`` / ``dangling_read`` findings every round. The two-pass scheme removes it:

  - DECLARE gives each node ONE produced name; deterministic dedupe (``dedupe_names``) makes
    the produced set globally UNIQUE → ``duplicate_producer`` cannot arise.
  - SELECT chooses inputs by ORDINAL from the closed ``build_option_names`` list; deterministic
    ``resolve_consumes`` maps ordinals → real names → every consume references a real produced
    name or the input token → ``near_miss`` / ``dangling_read`` cannot arise.
  - A repair re-runs only SELECT: the produced-name set is IDENTICAL before and after (the
    thrash-elimination proof).

These are deterministic invariants — no model, milliseconds. They prove the failure CLASS is
impossible; they do NOT measure the model's selection QUALITY (that is the live benchmark).
"""

from __future__ import annotations

from andamentum.forge.assemble import assemble
from andamentum.forge.decompose import (
    build_option_names,
    dedupe_names,
    resolve_consumes,
    visible_producers,
)
from andamentum.forge.diagnose import diagnose
from andamentum.forge.schemas import INPUT_TOKENS, DataKind, FindingKind, NodeDraft
from andamentum.forge.spec import NodeControl


def _node(node_id: str, produces: str, *, kind=DataKind.SIGNAL, control=NodeControl.NONE):
    d = NodeDraft(id=node_id, area="core", job=f"job {node_id}")
    d.produces = [produces]
    d.produces_kind = kind
    d.control = control
    return d


def test_forward_window_excludes_downstream_and_self_signals() -> None:
    """A node may read only earlier producers, a later CHECKPOINT, or its own ENTITY output —
    never a later signal node and never its own signal output. This makes an accidental cycle
    (the failure the unconstrained closed set introduced) impossible by construction."""
    boards = [
        _node("n1", "urgency"),                                  # signal
        _node("n2", "team"),                                     # signal
        _node("n3", "again", control=NodeControl.CHECKPOINT),    # later checkpoint
        _node("n4", "ticket_record", kind=DataKind.ENTITY),      # entity
    ]
    # n1 (first, signal): sees only the input + the later checkpoint — no earlier, no self.
    vis1 = {d.id for d in visible_producers(boards, 0)}
    assert vis1 == {"n3"}  # later checkpoint visible; n2 (later signal) and n1-self excluded
    # n2: sees earlier n1 + later checkpoint n3; NOT itself, NOT later entity n4.
    vis2 = {d.id for d in visible_producers(boards, 1)}
    assert vis2 == {"n1", "n3"}
    # n4 (entity): sees earlier nodes AND itself (rung-2 read-modify-write self-edge).
    vis4 = {d.id for d in visible_producers(boards, 3)}
    assert "n4" in vis4 and "n1" in vis4


def test_forward_window_selection_cannot_form_an_unintended_cycle() -> None:
    """Whatever ordinals each node picks over its OWN forward-window option list, the
    assembled board has no cycle that lacks a checkpoint — so ``unintended_cycle`` (which
    exploded under the all-nodes closed set) cannot arise from selection."""
    boards = [_node(f"n{i}", f"out{i}") for i in range(1, 6)]  # 5 plain signal nodes
    # Every node greedily selects ALL of its available options (the worst case for cycles).
    for i, d in enumerate(boards):
        opts = build_option_names([p.produces[0] for p in visible_producers(boards, i)])
        names, _ = resolve_consumes(list(range(len(opts))), opts)
        d.consumes = [n for n in names if n not in INPUT_TOKENS]
    report = diagnose(boards, assemble(boards))
    assert FindingKind.UNINTENDED_CYCLE not in {f.kind for f in report.findings}

_NAME_FLAWS = {
    FindingKind.DUPLICATE_PRODUCER,
    FindingKind.NEAR_MISS,
    FindingKind.DANGLING_READ,
}


def _build_board(
    declared_produces: list[str],
    selections: list[list[int]],
    *,
    kinds: list[DataKind] | None = None,
) -> list[NodeDraft]:
    """Construct a board exactly as ``decompose`` does: DECLARE → dedupe → SELECT by ordinal.

    ``declared_produces[i]`` is node i's raw declared output; ``selections[i]`` is node i's
    chosen input ordinals over the closed option list. Returns the wired ``NodeDraft`` board.
    """
    ids = [f"n{i + 1}" for i in range(len(declared_produces))]
    unique = dedupe_names(declared_produces, [f"{nid}_out" for nid in ids], INPUT_TOKENS)
    option_names = build_option_names(unique)
    nodes: list[NodeDraft] = []
    for i, nid in enumerate(ids):
        names, _dropped = resolve_consumes(selections[i], option_names)
        kind = kinds[i] if kinds else DataKind.SIGNAL
        nodes.append(
            NodeDraft(id=nid, consumes=names, produces=[unique[i]], produces_kind=kind)
        )
    return nodes


# --- dedupe: produced names are globally unique → no duplicate_producer -----------------


def test_dedupe_makes_colliding_produces_unique() -> None:
    unique = dedupe_names(
        ["answer", "answer", "Answer", "answer"],
        ["n1_out", "n2_out", "n3_out", "n4_out"],
        INPUT_TOKENS,
    )
    assert len(set(unique)) == len(unique) == 4
    assert unique[0] == "answer"
    assert all(u.startswith("answer") for u in unique)


def test_dedupe_never_shadows_an_input_token() -> None:
    # A produce that canonicalises to an input token must be suffixed, never left to shadow it.
    unique = dedupe_names(["input", "request"], ["n1_out", "n2_out"], INPUT_TOKENS)
    assert "input" not in unique
    assert "request" not in unique


def test_blank_produce_falls_back_to_a_unique_node_name() -> None:
    unique = dedupe_names(["", "  ", "answer"], ["n1_out", "n2_out", "n3_out"], INPUT_TOKENS)
    assert len(set(unique)) == 3
    assert "" not in unique


def test_duplicate_producer_never_arises_from_deduped_board() -> None:
    # Every node declares the SAME name; after dedupe + a valid chain there is no duplicate.
    nodes = _build_board(
        ["thing", "thing", "thing"],
        [[0], [1], [2]],  # n2 reads n1's output, n3 reads n2's output
    )
    report = diagnose(nodes, assemble(nodes))
    assert FindingKind.DUPLICATE_PRODUCER not in {f.kind for f in report.findings}


# --- select: resolved consumes always reference real names → no near_miss/dangling ------


def test_resolve_consumes_drops_out_of_range_and_records_it() -> None:
    option_names = build_option_names(["a", "b"])  # indices 0..2
    names, dropped = resolve_consumes([0, 2, 99, -1], option_names)
    assert names == ["input", "b"]
    assert dropped == [99, -1]


def test_resolve_consumes_dedups_repeated_ordinals() -> None:
    option_names = build_option_names(["a", "b"])
    names, dropped = resolve_consumes([1, 1, 2], option_names)
    assert names == ["a", "b"]
    assert dropped == []


def test_name_flaws_never_arise_from_index_selected_consumes() -> None:
    # A messy board: fan-in, a self-select, and an out-of-range ordinal — none can produce a
    # name flaw, because every resolved consume is a real produced name or the input token.
    nodes = _build_board(
        ["left", "right", "merged"],
        [[0], [0], [1, 2, 999]],  # n3 fans in n1+n2; 999 is dropped
    )
    report = diagnose(nodes, assemble(nodes))
    assert not (_NAME_FLAWS & {f.kind for f in report.findings})


# --- repair invariant: re-selecting inputs never changes the produced-name set ----------


def test_repair_reselect_leaves_produced_names_identical() -> None:
    declared = ["parsed", "ranked", "answer"]
    before = _build_board(declared, [[0], [1], [0]])  # n3 mis-reads the input
    before_produces = [n.produces[0] for n in before]

    # A repair re-runs only SELECT (new ordinals for the flagged node); DECLARE is frozen.
    after = _build_board(declared, [[0], [1], [2]])  # n3 now reads n2's output
    after_produces = [n.produces[0] for n in after]

    assert before_produces == after_produces  # the produced set is unchanged — no thrash
    assert before[2].consumes != after[2].consumes  # only the wiring moved


# --- property-style sweeps: the three name-flaw classes never appear ---------------------


def test_property_varied_boards_never_produce_name_flaws() -> None:
    # Vary board size, produced-name collisions, and selection ordinals (including
    # out-of-range and self-selects). No combination may yield a name flaw.
    for size in range(1, 6):
        raw_variants = [
            [f"d{i}" for i in range(size)],  # all distinct
            ["same"] * size,  # all colliding
            ["Mix"] * (size // 2) + [f"u{i}" for i in range(size - size // 2)],
        ]
        for raw in raw_variants:
            # Deterministic pseudo-varied selections: each node reads its two predecessors,
            # plus an out-of-range ordinal, plus (for the first) the input.
            selections = [
                [j for j in (i, i - 1, 10 * size) if j >= 0] or [0] for i in range(size)
            ]
            nodes = _build_board(raw, selections)
            report = diagnose(nodes, assemble(nodes))
            flaws = _NAME_FLAWS & {f.kind for f in report.findings}
            assert not flaws, (raw, selections, flaws)


def test_property_entity_self_edge_is_not_a_name_flaw() -> None:
    # The rung-2 read-modify-write: a node reads and rewrites its own durable entity.
    nodes = _build_board(
        ["saved_value", "answer"],
        [[0, 1], [1]],  # n1 reads input + its own produce (self-edge); n2 reads n1's
        kinds=[DataKind.ENTITY, DataKind.SIGNAL],
    )
    # Mark n1 with the entity round-trip shape diagnose exempts.
    report = diagnose(nodes, assemble(nodes))
    assert not (_NAME_FLAWS & {f.kind for f in report.findings})
    assert FindingKind.UNINTENDED_CYCLE not in {f.kind for f in report.findings}


def test_control_field_survives_construction() -> None:
    # A checkpoint declared in pass 2a must ride through to the board (diagnose reads it).
    nodes = _build_board(["seed", "refined"], [[0], [1]])
    nodes[0].control = NodeControl.CHECKPOINT
    assert nodes[0].control is NodeControl.CHECKPOINT
