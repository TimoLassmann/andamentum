"""Offline self-tests for the shape inspector — real specs, no model.

Each design is built as a typed :class:`DesignPlan` and run through the real
``compile_spec`` (deterministic, no LLM), so ``detect_features`` is asserted against an
actually-validated :class:`SystemSpec` — the same path the live benchmark exercises.
"""

from __future__ import annotations

from andamentum.forge import compile_spec
from andamentum.forge.schemas import DataKind, DesignPlan, ForgeWhy, NodeDraft
from andamentum.forge.spec import NodeControl, NodeKind

from .shape import detect_features, outcome_matches
from .types import Case, RunOutcome


def _why(purpose: str = "Do a thing.") -> ForgeWhy:
    return ForgeWhy(purpose=purpose, boundary_in="a request", boundary_out="an answer")


# --- detect_features over real compiled specs -----------------------------------


def test_plain_chain_has_no_features() -> None:
    plan = DesignPlan(
        why=_why(),
        nodes=[
            NodeDraft(
                id="n1",
                area="core",
                job="Parse the request.",
                kind=NodeKind.SPINE,
                consumes=["input"],
                produces=["parsed_request"],
            ),
            NodeDraft(
                id="n2",
                area="core",
                job="Format the result.",
                kind=NodeKind.SPINE,
                consumes=["parsed_request"],
                produces=["result"],
            ),
        ],
    )
    spec = compile_spec(plan)
    assert detect_features(spec) == set()


def test_decision_node_is_a_branch() -> None:
    plan = DesignPlan(
        why=_why(),
        nodes=[
            NodeDraft(
                id="n1",
                area="core",
                job="Decide the route.",
                kind=NodeKind.SPINE,
                consumes=["input"],
                produces=["route_label"],
                control=NodeControl.DECISION,
            ),
            NodeDraft(
                id="n2",
                area="core",
                job="Produce the answer.",
                kind=NodeKind.SPINE,
                consumes=["route_label"],
                produces=["answer"],
            ),
        ],
    )
    spec = compile_spec(plan)
    assert detect_features(spec) == {"branch"}


def test_checkpoint_node_is_a_loop() -> None:
    plan = DesignPlan(
        why=_why(),
        nodes=[
            NodeDraft(
                id="n1",
                area="core",
                job="Gather data.",
                kind=NodeKind.SPINE,
                consumes=["input"],
                produces=["gathered"],
            ),
            NodeDraft(
                id="n2",
                area="core",
                job="Check sufficiency.",
                kind=NodeKind.SPINE,
                consumes=["gathered"],
                produces=["checked"],
                control=NodeControl.CHECKPOINT,
            ),
        ],
    )
    spec = compile_spec(plan)
    assert detect_features(spec) == {"loop"}


def test_entity_round_trip_is_an_entity() -> None:
    # A datum a single node reads-modifies-writes is durable state (the §7 round-trip
    # signature) and compiles to a declared entity.
    plan = DesignPlan(
        why=_why("Remember and update a value across runs."),
        nodes=[
            NodeDraft(
                id="n1",
                area="core",
                job="Update the saved value.",
                kind=NodeKind.SPINE,
                consumes=["saved_value", "input"],
                produces=["saved_value"],
                produces_kind=DataKind.ENTITY,
            ),
            NodeDraft(
                id="n2",
                area="core",
                job="Format the answer.",
                kind=NodeKind.SPINE,
                consumes=["saved_value"],
                produces=["answer"],
            ),
        ],
    )
    spec = compile_spec(plan)
    assert "entity" in detect_features(spec)


def test_one_field_read_by_two_nodes_is_fanout() -> None:
    plan = DesignPlan(
        why=_why(),
        nodes=[
            NodeDraft(
                id="n1",
                area="core",
                job="Prepare the shared data.",
                kind=NodeKind.SPINE,
                consumes=["input"],
                produces=["shared"],
            ),
            NodeDraft(
                id="n2",
                area="core",
                job="Handle the first path.",
                kind=NodeKind.SPINE,
                consumes=["shared"],
                produces=["branch_a"],
            ),
            NodeDraft(
                id="n3",
                area="core",
                job="Handle the second path.",
                kind=NodeKind.SPINE,
                consumes=["shared"],
                produces=["branch_b"],
            ),
            NodeDraft(
                id="n4",
                area="core",
                job="Combine the paths.",
                kind=NodeKind.SPINE,
                consumes=["branch_a", "branch_b"],
                produces=["digest"],
            ),
        ],
    )
    spec = compile_spec(plan)
    assert detect_features(spec) == {"fanout"}


# --- outcome_matches unit cases -------------------------------------------------


def test_refuse_case_passes_only_when_refused() -> None:
    case = Case(brief="x", expected="refuse", grammar="none")
    assert outcome_matches(case, RunOutcome(kind="refused"))
    assert not outcome_matches(case, RunOutcome(kind="built"))
    assert not outcome_matches(case, RunOutcome(kind="design_failed", error="boom"))


def test_sequence_build_requires_no_feature() -> None:
    case = Case(brief="x", expected="build", grammar="sequence")
    assert outcome_matches(case, RunOutcome(kind="built", features=set()))
    assert not outcome_matches(case, RunOutcome(kind="built", features={"branch"}))
    assert not outcome_matches(case, RunOutcome(kind="refused"))


def test_grammar_build_requires_its_feature() -> None:
    for grammar, feature in (
        ("branch", "branch"),
        ("loop", "loop"),
        ("fanout", "fanout"),
        ("stateful", "entity"),
    ):
        case = Case(brief="x", expected="build", grammar=grammar)
        assert outcome_matches(case, RunOutcome(kind="built", features={feature}))
        assert not outcome_matches(case, RunOutcome(kind="built", features=set()))
